"""This script implements Spanish fine-tuning on the pre-trained English decoder-only transformer model"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import HubertForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from jiwer import wer, cer
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
import math
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('spanish_vocab.json', 'r', encoding='utf-8') as vocab_file:
    spanish_vocab_dict = json.load(vocab_file)

print(f"Spanish vocab size: {len(spanish_vocab_dict)}")
SOS_ID = spanish_vocab_dict["[SOS]"]
EOS_ID = spanish_vocab_dict["[EOS]"]
PAD_ID = spanish_vocab_dict["[PAD]"]
print(f"Spanish-specific tokens: {[k for k in spanish_vocab_dict if k in 'áéíóúñü']}")

spanish_tokenizer = Wav2Vec2CTCTokenizer("./spanish_vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
spanish_processor = Wav2Vec2Processor.from_pretrained("./hubert_model", tokenizer=spanish_tokenizer)
encoder_model = HubertForCTC.from_pretrained("./hubert_model").to(device)

class Config:
    d_model = encoder_model.config.hidden_size
    vocab_size = len(spanish_vocab_dict)
    max_len = 1024
    sos_id = SOS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID
    learning_rate = 1e-4
    num_heads = 8
    num_layers = 12
    dropout = 0.1
    batch_size = 4
    num_workers = 4
    pin_memory = True
    max_epochs = 10

class PositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pe = torch.zeros(config.max_len, config.d_model)
        position = torch.arange(config.max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        batch_size, seq_len, d_model = word_embeddings.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max allowed {self.config.max_len}")
        pe_adjusted = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return word_embeddings + pe_adjusted.to(word_embeddings.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_k = self.d_model // self.num_heads 
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, q_encodings, k_encodings, v_encodings, mask=None, is_self_attn=True):
        batch_size = q_encodings.size(0)
        q = self.W_q(q_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        similarities = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        if is_self_attn:
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(q.device)
            similarities = similarities.masked_fill(causal_mask == 0, -1e9)
        attention_weights = F.softmax(similarities, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_out(attention_output)

class DecoderOnlyTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_losses_epochs = []
        self.val_losses_epochs = []
        self.current_train_losses = []
        self.current_val_losses = []
        self.position_encoding = PositionEncoding(config)
        self.target_query_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.sep_embedding = nn.Parameter(torch.randn(1, config.d_model))
        self.self_attention_layers = nn.ModuleList([MultiHeadAttention(config) for _ in range(config.num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(2 * config.num_layers)])
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4),
                nn.ReLU(),
                nn.Linear(config.d_model * 4, config.d_model),
                nn.Dropout(config.dropout),
            ) for _ in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=config.pad_id)

    def forward(self, encoder_features, target_ids=None, max_length=None):
        batch_size = encoder_features.size(0)
        audio_prefix = encoder_features
        if target_ids is not None:
            text_embeddings = self.target_query_embedding(target_ids)
            text_embeddings = self.position_encoding(text_embeddings)
            sep = self.sep_embedding.expand(batch_size, 1, self.config.d_model)
            x = torch.cat([audio_prefix, sep, text_embeddings], dim=1)
            for i, (self_attention, feedforward) in enumerate(zip(self.self_attention_layers, self.feedforward_layers)):
                attn_output = self_attention(x, x, x, is_self_attn=True)
                x = self.layer_norms[i](x + attn_output)
                ff_output = feedforward(x)
                x = self.layer_norms[i + self.config.num_layers](x + ff_output)
            x_text = x[:, audio_prefix.size(1) + 1:]
            return self.fc(x_text)
        else:
            current_ids = torch.full((batch_size, 1), self.config.sos_id, device=self.device)
            outputs = []
            for i in range(max_length or self.config.max_len):
                text_embeddings = self.target_query_embedding(current_ids)
                text_embeddings = self.position_encoding(text_embeddings)
                sep = self.sep_embedding.expand(batch_size, 1, self.config.d_model)
                x = torch.cat([audio_prefix, sep, text_embeddings], dim=1)
                for j, (self_attention, feedforward) in enumerate(zip(self.self_attention_layers, self.feedforward_layers)):
                    attn_output = self_attention(x, x, x, is_self_attn=True)
                    x = self.layer_norms[j](x + attn_output)
                    ff_output = feedforward(x)
                    x = self.layer_norms[j + self.config.num_layers](x + ff_output)
                    del attn_output, ff_output
                logits = self.fc(x)
                last_token_index = audio_prefix.size(1) + 1 + current_ids.size(1) - 1
                next_token = torch.argmax(logits[:, last_token_index, :], dim=-1, keepdim=True)
                outputs.append(logits[:, last_token_index:last_token_index+1, :])
                if (next_token == self.config.eos_id).all():
                    break
                current_ids = torch.cat([current_ids, next_token], dim=1)
                del x, logits
            return torch.cat(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.current_train_losses.append(loss.item())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.current_val_losses.append(loss.item())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_train_losses:
            avg_train_loss = sum(self.current_train_losses) / len(self.current_train_losses)
            self.train_losses_epochs.append(avg_train_loss)
            self.current_train_losses = []
            with open("train_losses_spanish.json", "w") as f:
                json.dump(self.train_losses_epochs, f)

    def on_validation_epoch_end(self):
        if self.current_val_losses:
            avg_val_loss = sum(self.current_val_losses) / len(self.current_val_losses)
            self.val_losses_epochs.append(avg_val_loss)
            self.current_val_losses = []
            with open("val_losses_spanish.json", "w") as f:
                json.dump(self.val_losses_epochs, f)

    def _compute_loss(self, batch):
        waveform, sample_rate, ground_truth = batch
        waveform = waveform.to(self.device)
        labels = spanish_tokenizer(ground_truth, return_tensors="pt", padding=True).input_ids.to(self.device)
        with torch.no_grad():
            inputs = spanish_processor(waveform.squeeze(1), sampling_rate=sample_rate[0],
                                       return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            encoder_features = encoder_model(inputs["input_values"].squeeze(0), output_hidden_states=True).hidden_states[9]
        logits = self(encoder_features, target_ids=labels[:, :-1])
        return self.loss(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.learning_rate)

class SpanishAudioDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dirs):
        if isinstance(preprocessed_dirs, str):
            preprocessed_dirs = [preprocessed_dirs]
        self.preprocessed_dirs = [Path(dir) for dir in preprocessed_dirs]
        self.metadata = []
        for dir_path in self.preprocessed_dirs:
            with open(dir_path / "metadata.json", "r", encoding="utf-8") as f:
                dir_metadata = json.load(f)
                for item in dir_metadata:
                    if "/" not in item["audio_path"] and "\\" not in item["audio_path"]:
                        item["audio_path"] = str(dir_path / item["audio_path"])
                    else:
                        item["audio_path"] = str(Path(item["audio_path"]).absolute())
                self.metadata.extend(dir_metadata)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        waveform = torch.load(item["audio_path"])
        sample_rate = item["sample_rate"]
        transcript = item["transcript"]
        return waveform, sample_rate, transcript
   
def collate_fn(batch):
    waveforms, sample_rates, transcripts = [], [], []
    for sample in batch:
        waveform, sample_rate, transcript, *_ = sample  
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        transcripts.append(transcript)
    max_len = max(w.shape[1] for w in waveforms)
    padded_waveforms = [F.pad(w, (0, max_len - w.shape[1]), "constant", 0) for w in waveforms]
    return torch.stack(padded_waveforms), sample_rates, transcripts

def setup_training(config, preprocessed_dirs):
    dataset = SpanishAudioDataset(preprocessed_dirs)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                             num_workers=config.num_workers, pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader

def decode_prediction(logits):
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    if EOS_ID in predicted_ids:
        predicted_ids = predicted_ids[:predicted_ids.index(EOS_ID) + 1]
    return spanish_tokenizer.decode(predicted_ids)

def train_and_visualize(preprocessed_dirs):
    config = Config()
    train_loader, val_loader, test_loader = setup_training(config, preprocessed_dirs)
    decoder_model = DecoderOnlyTransformer(config).to(device)
    
    try:
        pretrained_dict = torch.load("decoder_model_weights_960h_12layers.pth", map_location=device, weights_only=True)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not (k.startswith('fc') or k.startswith('target_query_embedding'))}
        decoder_model.load_state_dict(filtered_dict, strict=False)
        print("Loaded English pretrained weights.")
    except FileNotFoundError:
        print("Pretrained weights not found. Starting from scratch.")
    
    # reinitialize final layers for spanish vocab
    decoder_model.fc = nn.Linear(config.d_model, config.vocab_size).to(device)
    decoder_model.target_query_embedding = nn.Embedding(config.vocab_size, config.d_model).to(device)
    print(f"Reinitialized fc and target_query_embedding for Spanish vocab size {config.vocab_size}.")
    
    logger = CSVLogger("logs/", name="spanish_speech_model")
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=logger,
        gradient_clip_val=1.0
    )
    trainer.fit(decoder_model, train_loader, val_loader)
    
    torch.save(decoder_model.state_dict(), "decoder_model_weights_spanish.pth")
    print("Model weights saved.")
    
    plt.plot(decoder_model.train_losses_epochs, label="Training Loss")
    plt.plot(decoder_model.val_losses_epochs, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("spanish_finetune_losses.png")
    plt.close()
    return decoder_model, train_loader, val_loader, test_loader

def evaluate(decoder_model, test_loader, processor, tokenizer, encoder_model):
    decoder_model.eval()
    total_wer = 0
    total_cer = 0
    total_loss = 0
    total_samples = 0
    all_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            waveforms, sample_rates, transcripts = batch
            waveforms = waveforms.to(device)
            labels = tokenizer(transcripts, return_tensors="pt", padding=True).input_ids.to(device)
            sampling_rate = sample_rates[0] if isinstance(sample_rates, list) else sample_rates
            inputs = processor(waveforms.squeeze(1), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_values = inputs["input_values"].squeeze(0)
            outputs = encoder_model(input_values, output_hidden_states=True)
            encoder_features = outputs.hidden_states[9].to(device)
            decoder_model = decoder_model.to(device)
            logits = decoder_model(encoder_features, target_ids=labels[:, :-1])
            loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            total_loss += loss.item()
            predicted_texts = [decode_prediction(logit) for logit in logits]
            for i, (pred, truth) in enumerate(zip(predicted_texts, transcripts)):
                pred_clean = pred.replace("[SOS]", "").replace("[EOS]", "").strip()
                truth_clean = truth.replace("[SOS]", "").replace("[EOS]", "").strip()
                sample_wer = wer(truth_clean, pred_clean)
                sample_cer = cer(truth_clean, pred_clean)
                total_wer += sample_wer
                total_cer += sample_cer
                total_samples += 1
                all_results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'ground_truth': truth_clean,
                    'prediction': pred_clean,
                    'wer': sample_wer,
                    'cer': sample_cer
                })
    avg_wer = total_wer / total_samples
    avg_cer = total_cer / total_samples
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average Cross-Entropy Loss: {avg_loss:.4f}")
    with open('spanish_finetune_results', 'w') as f:
        f.write(str(all_results))
    return avg_wer, avg_cer, avg_loss

if __name__ == "__main__":
    preprocessed_dirs = ["spanish_train_00", "spanish_train_01", "spanish_train_02", "spanish_train_03",
                          "spanish_train_04", "spanish_train_05", "spanish_train_06", "spanish_train_07"]
    decoder_model, train_loader, val_loader, test_loader = train_and_visualize(preprocessed_dirs)
    avg_wer, avg_cer, avg_loss = evaluate(decoder_model, test_loader, spanish_processor, spanish_tokenizer, encoder_model)
