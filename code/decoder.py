"""This script implements the decoder-only transformer architecture for 
training the model on 960 hours of English Librispeech data"""

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import HubertForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from jiwer import wer, cer
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")

# HuBERT Base Model downloaded from: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
model = HubertForCTC.from_pretrained("./hubert_model").to(device)
processor = Wav2Vec2Processor.from_pretrained("./hubert_model")

with open('vocab.json', 'r') as vocab_file:
    vocab_dict = json.load(vocab_file)

# get special token indices
SOS_ID = vocab_dict["[SOS]"]
EOS_ID = vocab_dict["[EOS]"]
PAD_ID = vocab_dict["[PAD]"]

# define config class
class Config:
    d_model = model.config.hidden_size
    vocab_size = len(vocab_dict)
    max_len = 1024
    learning_rate = 1e-4  # learning rate for the optimizer
    num_heads = 8         # number of attention heads for each multi-head attention layer
    num_layers = 12        # number of transformer layers
    dropout = 0.1         # dropout rate for feedforward layers
    batch_size = 4  
    num_workers = 16
    pin_memory = True
    persistent_workers = True
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
        batch_size, seq_len, d_model = word_embeddings.shape  # ensure proper shape

        # ensure positional encodings match the input sequence length
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max allowed {self.config.max_len}")

        # slice the positional encoding correctly
        pe_adjusted = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        return word_embeddings + pe_adjusted.to(word_embeddings.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_k = self.d_model // self.num_heads 
        
        # linear layers for q, k, v
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)
        
        # final linear layer
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, q_encodings, k_encodings, v_encodings, mask=None, is_self_attn=True):
        batch_size = q_encodings.size(0)
    
        # project into q, k, v spaces
        q = self.W_q(q_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v_encodings).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
        # compute scaled dot product attention
        similarities = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
    
        # causal mask for self-attention
        if is_self_attn:
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(q.device)
            similarities = similarities.masked_fill(causal_mask == 0, -1e9)

        # softmax over the last dimension (attention scores)
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

        # positional encoding for text tokens
        self.position_encoding = PositionEncoding(config)

        # learned text token embedding
        self.target_query_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # separator embedding indicating separation between audio and text
        self.sep_embedding = nn.Parameter(torch.randn(1, config.d_model))
        
        self.self_attention_layers = nn.ModuleList([MultiHeadAttention(config) for _ in range(config.num_layers)])
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4),
                nn.ReLU(),
                nn.Linear(config.d_model * 4, config.d_model),
                nn.Dropout(config.dropout),
            ) for _ in range(config.num_layers)
        ])

        # two layer norms per transformer layer (one after attention and one after feedforward)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(2 * config.num_layers)])
        
        # final projection layer to vocabulary logits
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    def forward(self, encoder_features, target_ids=None, max_length=None):
        batch_size = encoder_features.size(0)

        # hubert features as the audio prefix
        audio_prefix = encoder_features  # shape: (batch_size, seq_len_audio, d_model)

        if target_ids is not None:
            # training mode: teacher forcing
            text_embeddings = self.target_query_embedding(target_ids)
            text_embeddings = self.position_encoding(text_embeddings)

            # create a separator token embedding for the batch.
            sep = self.sep_embedding.expand(batch_size, 1, self.config.d_model)

            # concatenate audio features, separator, and text embeddings
            x = torch.cat([audio_prefix, sep, text_embeddings], dim=1)

            # process through transformer layers
            for i in range(self.config.num_layers):
                attn_output = self.self_attention_layers[i](x, x, x, is_self_attn=True)
                x = self.layer_norms[i](x + attn_output)
                ff_output = self.feedforward_layers[i](x)
                x = self.layer_norms[i + self.config.num_layers](x + ff_output)

            # only keep the outputs corresponding to the text portion skip audio_prefix tokens and the separator token
            x_text = x[:, audio_prefix.size(1) + 1:]
            return self.fc(x_text)
        
        else:
            # inference mode: autoregressive generation
            current_ids = torch.full((batch_size, 1), SOS_ID, device=self.device)
            outputs = []
            for i in range(max_length or self.config.max_len):
                text_embeddings = self.target_query_embedding(current_ids)
                text_embeddings = self.position_encoding(text_embeddings)
                sep = self.sep_embedding.expand(batch_size, 1, self.config.d_model)
                x = torch.cat([audio_prefix, sep, text_embeddings], dim=1)
                for j in range(self.config.num_layers):
                    attn_output = self.self_attention_layers[j](x, x, x, is_self_attn=True)
                    x = self.layer_norms[j](x + attn_output)
                    ff_output = self.feedforward_layers[j](x)
                    x = self.layer_norms[j + self.config.num_layers](x + ff_output)
                    del attn_output, ff_output # free up memory
                logits = self.fc(x)
                last_token_index = audio_prefix.size(1) + 1 + current_ids.size(1) - 1
                next_token = torch.argmax(logits[:, last_token_index, :], dim=-1, keepdim=True)
                outputs.append(logits[:, last_token_index: last_token_index+1, :])
                if (next_token == EOS_ID).all():
                    break
                current_ids = torch.cat([current_ids, next_token], dim=1)
                del x, logits # free up memory
            return torch.cat(outputs, dim=1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.current_train_losses.append(loss.item())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.current_val_losses.append(loss.item())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_train_losses:
            self.train_losses_epochs.append(np.mean(self.current_train_losses))
            self.current_train_losses = []
            with open("train_losses.json", "w") as f:
                json.dump(self.train_losses_epochs, f)

    def on_validation_epoch_end(self):
        if self.current_val_losses:
            self.val_losses_epochs.append(np.mean(self.current_val_losses))
            self.current_val_losses = []
            with open("val_losses.json", "w") as f:
                json.dump(self.val_losses_epochs, f)

    def _compute_loss(self, batch):
        waveform, sample_rate, ground_truth = batch
        waveform = waveform.to(self.device)

        labels = tokenizer(ground_truth, return_tensors="pt", padding=True).input_ids.to(self.device)

        with torch.no_grad():
            waveform = waveform.squeeze(1)
            sampling_rate = sample_rate[0] if isinstance(sample_rate, list) else sample_rate

            inputs = processor(
                waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            input_values = inputs["input_values"].squeeze(0)
            outputs = model(input_values, output_hidden_states=True)
            encoder_features = outputs.hidden_states[9]

        # forward pass with teacher forcing
        logits = self.forward(encoder_features, target_ids=labels[:, :-1])  # exclude EOS token for input
        
        # compute loss against shifted targets (include EOS token in target)
        return self.loss(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
    
def decode_prediction(logits):
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    if EOS_ID in predicted_ids:
        predicted_ids = predicted_ids[:predicted_ids.index(EOS_ID) + 1]  # include EOS in output

    return tokenizer.decode(predicted_ids)
   
def collate_fn(batch):
    waveforms, sample_rates, transcripts = [], [], []

    for sample in batch:
        waveform, sample_rate, transcript, *_ = sample  
        waveforms.append(waveform)
        sample_rates.append(sample_rate)

        # add SOS & EOS tokens to transcript
        transcripts.append(f"[SOS] {transcript.lower()} [EOS]")

    max_len = max(w.shape[1] for w in waveforms)
    padded_waveforms = [F.pad(w, (0, max_len - w.shape[1]), "constant", 0) for w in waveforms]

    return torch.stack(padded_waveforms), sample_rates, transcripts

def setup_training(config):
    dataset_100h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-clean-100", download=True)
    dataset_360h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-clean-360", download=True)
    dataset_500h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-other-500", download=True)
    
    # combine into one dataset (960 hours)
    full_dataset = ConcatDataset([dataset_100h, dataset_360h, dataset_500h])
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = config.batch_size 
    num_workers = config.num_workers
    pin_memory = config.pin_memory
    persistent_workers = config.persistent_workers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    return train_loader, val_loader, test_loader


def train_and_visualize():
    config = Config()
    train_loader, val_loader, test_loader = setup_training(config)
    decoder_model = DecoderOnlyTransformer(config).to(device)

    logger = CSVLogger("logs/", name="speech_model")
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    
    trainer.fit(decoder_model, train_loader, val_loader)

    # save model weights
    model_save_path = "decoder_model_weights.pth"
    torch.save(decoder_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")
    
    # training vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(decoder_model.train_losses_epochs, label='Training Loss', alpha=0.7)
    plt.plot(decoder_model.val_losses_epochs, label='Validation Loss', color='orange', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_validation_losses_plot.png')
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

            # process inputs through the hubert
            sampling_rate = sample_rates[0] if isinstance(sample_rates, list) else sample_rates
            inputs = processor(waveforms.squeeze(1), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            input_values = inputs["input_values"].squeeze(0)

            outputs = encoder_model(input_values, output_hidden_states=True)
            encoder_features = outputs.hidden_states[9].to(device)

            # ensure decoder model is on the correct device
            decoder_model = decoder_model.to(device)
            
            # generate predictions
            logits = decoder_model(encoder_features, target_ids=labels[:, :-1])  # exclude EOS in input
            
            # compute cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))  # shift labels by 1
            total_loss += loss.item()

            # decode predictions
            predicted_texts = [decode_prediction(logit) for logit in logits]

            # calculate WER and CER for each sample
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

    # save all results to a JSON file
    with open('speech_recognition_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to 'speech_recognition_results.json'")
    return avg_wer, avg_cer, avg_loss


if __name__ == "__main__":
    decoder_model, train_loader, val_loader, test_loader = train_and_visualize()
    avg_wer, avg_cer, avg_loss= evaluate(decoder_model, test_loader, processor, tokenizer, model)