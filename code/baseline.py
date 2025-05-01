""" This script implements the baseline for comparing the fine-tuned decoder-only transformer"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from transformers import HubertForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from jiwer import wer, cer
import torchaudio
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from torch.optim import Adam
import torch.nn.functional as F

torch.cuda.empty_cache()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vocabularies
with open('vocab.json', 'r', encoding='utf-8') as f:
    english_vocab_dict = json.load(f)
with open('spanish_vocab.json', 'r', encoding='utf-8') as f:
    spanish_vocab_dict = json.load(f)

# special token IDs
ENGLISH_SOS_ID = english_vocab_dict.get("[SOS]", -1)
ENGLISH_EOS_ID = english_vocab_dict.get("[EOS]", -1)
ENGLISH_PAD_ID = english_vocab_dict.get("[PAD]", -1)
SPANISH_SOS_ID = spanish_vocab_dict.get("[SOS]", -1)
SPANISH_EOS_ID = spanish_vocab_dict.get("[EOS]", -1)
SPANISH_PAD_ID = spanish_vocab_dict.get("[PAD]", -1)

# tokenizers and processors
english_tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", 
    unk_token="[UNK]", 
    pad_token="[PAD]", 
    word_delimiter_token=" "
)
spanish_tokenizer = Wav2Vec2CTCTokenizer(
    "./spanish_vocab.json", 
    unk_token="[UNK]", 
    pad_token="[PAD]", 
    word_delimiter_token=" "
)
english_processor = Wav2Vec2Processor.from_pretrained("./hubert_model", tokenizer=english_tokenizer)
spanish_processor = Wav2Vec2Processor.from_pretrained("./hubert_model", tokenizer=spanish_tokenizer)

class HubertCTCTrainer(pl.LightningModule):
    def __init__(self, processor, tokenizer, vocab_size, pad_id, learning_rate=1e-4, pretrained_path="./hubert_model"):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.learning_rate = learning_rate
        
        # load pre-trained HuBERT model
        self.model = HubertForCTC.from_pretrained(
            pretrained_path, 
            vocab_size=vocab_size, 
            pad_token_id=pad_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            ignore_mismatched_sizes=True
        )
        
        # projector output matchesshould match the vocab size
        if self.model.lm_head.out_features != vocab_size:
            self.model.lm_head = nn.Linear(self.model.config.hidden_size, vocab_size)
            
        self.model = self.model.to(device)

        for param in self.model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        
        self.train_losses_epochs = []
        self.val_losses_epochs = []
        self.current_train_losses = []
        self.current_val_losses = []

    def forward(self, input_values, attention_mask=None, labels=None):
        return self.model(input_values, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self._process_batch(batch)
        loss = outputs.loss
        self.current_train_losses.append(loss.item())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._process_batch(batch)
        loss = outputs.loss
        self.current_val_losses.append(loss.item())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_train_losses:
            avg_train_loss = sum(self.current_train_losses) / len(self.current_train_losses)
            self.train_losses_epochs.append(avg_train_loss)
            self.current_train_losses = []

    def on_validation_epoch_end(self):
        if self.current_val_losses:
            avg_val_loss = sum(self.current_val_losses) / len(self.current_val_losses)
            self.val_losses_epochs.append(avg_val_loss)
            self.current_val_losses = []

    def _process_batch(self, batch):
        waveforms, sample_rates, transcripts = batch
        
        # process audio inputs
        with torch.no_grad():
            inputs = self.processor(
                waveforms, 
                sampling_rate=sample_rates[0], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_values = inputs["input_values"].squeeze(0)
            attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
        
        # process text labels
        with torch.no_grad():
            # remove special tokens for CTC loss
            clean_transcripts = [text.replace("[SOS]", "").replace("[EOS]", "").strip() for text in transcripts]
            
            # convert transcripts to token IDs
            batch_tokens = self.tokenizer(
                clean_transcripts,
                padding="longest",
                return_tensors="pt"
            )
            labels = batch_tokens.input_ids.to(self.device)
        
        # forward pass with CTC loss calculation
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

def collate_fn_lib(batch):
    waveforms, sample_rates, transcripts = [], [], []
    for waveform, sample_rate, transcript, *_ in batch:
        waveforms.append(waveform.squeeze(0))  # [1, seq_len] -> [seq_len]
        sample_rates.append(sample_rate)
        transcripts.append(transcript.lower())
    max_len = max(w.shape[-1] for w in waveforms)
    padded_waveforms = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]
    
    return torch.stack(padded_waveforms), sample_rates, transcripts

def setup_librispeech():
    dataset_100h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-clean-100", download=True)
    dataset_360h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-clean-360", download=True)
    dataset_500h = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-other-500", download=True)
    full_dataset = ConcatDataset([dataset_100h, dataset_360h, dataset_500h])
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    batch_size = 8  
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_lib, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_lib, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_lib, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

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
        waveform = torch.load(item["audio_path"])  # [1, seq_len]
        sample_rate = item["sample_rate"]
        transcript = item["transcript"]
        return waveform.squeeze(0), sample_rate, transcript  # [seq_len]

def collate_fn_mls(batch):
    waveforms, sample_rates, transcripts = [], [], []
    for waveform, sample_rate, transcript in batch:
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        
        # remove SOS and EOS tokens from transcripts
        clean_transcript = transcript.replace("[SOS]", "").replace("[EOS]", "").strip()
        transcripts.append(clean_transcript)
    max_len = max(w.shape[-1] for w in waveforms)
    padded_waveforms = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]
    
    return torch.stack(padded_waveforms), sample_rates, transcripts

def setup_spanish(preprocessed_dirs):
    dataset = SpanishAudioDataset(preprocessed_dirs)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 8 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_mls, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_mls, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_mls, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_and_visualize(model, train_loader, val_loader, name, max_epochs=3):
    logger = CSVLogger("logs/", name=f"{name}_hubert_ctc")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=logger,
        gradient_clip_val=1.0,
        precision=16,  
        accumulate_grad_batches=1 
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Save model
    torch.save(model.state_dict(), f"hubert_ctc_{name}.pth")
    print(f"Model weights saved to hubert_ctc_{name}.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses_epochs, label="Training Loss")
    plt.plot(model.val_losses_epochs, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{name.capitalize()} Fine-tuning Loss")
    plt.savefig(f"{name}_finetune_losses.png")
    plt.close()

def evaluate(model, test_loader, processor, tokenizer, name):
    model = model.to(device)
    model.eval()
    total_wer, total_cer, total_samples = 0, 0, 0
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            waveforms, sample_rates, transcripts = batch
            batch_size = len(transcripts)
            
            # preprocess audio
            inputs = processor(
                waveforms, 
                sampling_rate=sample_rates[0], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_values = inputs["input_values"].squeeze(0)
            attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
            
            # forward pass (without labels for inference)
            outputs = model.model(
                input_values=input_values,
                attention_mask=attention_mask
            )
            
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            predicted_texts = tokenizer.batch_decode(predicted_ids)
            
            # ensure sure the number of predictions matches the batch size
            if len(predicted_texts) != len(transcripts):
                print(f"Warning: Number of predictions ({len(predicted_texts)}) doesn't match batch size ({len(transcripts)})")
                min_size = min(len(predicted_texts), len(transcripts))
                predicted_texts = predicted_texts[:min_size]
                transcripts = transcripts[:min_size]
            
            # calculate metrics
            for i, (pred, truth) in enumerate(zip(predicted_texts, transcripts)):
                # clean up predictions and ground truth
                pred_clean = pred.replace("[SOS]", "").replace("[EOS]", "").replace("[PAD]", "").replace("[UNK]", "").strip()
                truth_clean = truth.replace("[SOS]", "").replace("[EOS]", "").strip()
                
                try:
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
                except ValueError as e:
                    print(f"Error calculating WER/CER for sample {i}: {e}")
                    print(f"Truth: '{truth_clean}'")
                    print(f"Prediction: '{pred_clean}'")
            
            # free memory
            torch.cuda.empty_cache()
    
    if total_samples == 0:
        print(f"Warning: No valid samples were processed for {name}")
        return 1.0, 1.0  # no valid samples then return worst possible score
    
    # calculate average metrics
    avg_wer = total_wer / total_samples
    avg_cer = total_cer / total_samples
    
    print(f"{name.capitalize()} - Average WER: {avg_wer:.4f}")
    print(f"{name.capitalize()} - Average CER: {avg_cer:.4f}")
    
    # save results
    with open(f'{name}_ctc_finetune_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return avg_wer, avg_cer

if __name__ == "__main__":

    # fine-tune on LibriSpeech
    english_train_loader, english_val_loader, english_test_loader = setup_librispeech()
    
    english_model = HubertCTCTrainer(
        processor=english_processor,
        tokenizer=english_tokenizer,
        vocab_size=len(english_vocab_dict),
        pad_id=ENGLISH_PAD_ID,
        learning_rate=1e-4
    )
    
    train_and_visualize(english_model, english_train_loader, english_val_loader, "english", max_epochs=3)
    english_wer, english_cer = evaluate(english_model, english_test_loader, english_processor, english_tokenizer, "english")
    
    # fine-tune on spanish data using the english weights
    preprocessed_dirs = [
        "spanish_train_00", "spanish_train_01", "spanish_train_02", "spanish_train_03",
        "spanish_train_04", "spanish_train_05", "spanish_train_06", "spanish_train_07"
    ]
    spanish_train_loader, spanish_val_loader, spanish_test_loader = setup_spanish(preprocessed_dirs)
    
    # initialize Spanish model with english weights
    spanish_model = HubertCTCTrainer(
        processor=spanish_processor,
        tokenizer=spanish_tokenizer,
        vocab_size=len(spanish_vocab_dict),
        pad_id=SPANISH_PAD_ID,
        learning_rate=1e-4 
    )
    
    # load English weights except for the projection layer
    english_state_dict = english_model.model.state_dict()
    spanish_state_dict = spanish_model.model.state_dict()
    
    transferred_keys = []
    for key in english_state_dict:
        if 'lm_head' not in key: 
            spanish_state_dict[key] = english_state_dict[key]
            transferred_keys.append(key)

    print(f"Transferred {len(transferred_keys)} parameters from English model to Spanish model")
    spanish_model.model.load_state_dict(spanish_state_dict, strict=False)
    
    # train and evaluate on spanish
    train_and_visualize(spanish_model, spanish_train_loader, spanish_val_loader, "spanish", max_epochs=10)
    spanish_wer, spanish_cer = evaluate(spanish_model, spanish_test_loader, spanish_processor, spanish_tokenizer, "spanish")
    
    print(f"Final results - English: WER={english_wer:.4f}, CER={english_cer:.4f}")
    print(f"Final results - Spanish: WER={spanish_wer:.4f}, CER={spanish_cer:.4f}")

