"""This script tests the decoder-only transformer fine-tuned on Spanish data"""

import torch
import torch.nn.functional as F
from jiwer import wer, cer
from spanish_finetune import DecoderOnlyTransformer, Config
from torch.utils.data import DataLoader, random_split
from transformers import HubertForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Wav2Vec2CTCTokenizer("./spanish_vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
model = HubertForCTC.from_pretrained("./hubert_model").to(device)
processor = Wav2Vec2Processor.from_pretrained("./hubert_model")

with open('spanish_vocab.json', 'r') as vocab_file:
    spanish_vocab_dict = json.load(vocab_file)

SOS_ID = spanish_vocab_dict["[SOS]"]
EOS_ID = spanish_vocab_dict["[EOS]"]
PAD_ID = spanish_vocab_dict["[PAD]"]

config = Config()

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
                    # Only prepend dir_path if audio_path is a simple filename (no slashes)
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

def decode_prediction(logits):
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    if EOS_ID in predicted_ids:
        predicted_ids = predicted_ids[:predicted_ids.index(EOS_ID) + 1]  

    return tokenizer.decode(predicted_ids)


config = Config()
decoder_model = DecoderOnlyTransformer(config).to(device)

preprocessed_dirs = ["spanish_test_00"]

dataset = SpanishAudioDataset(preprocessed_dirs)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def evaluate_decoder(decoder_model, processor, tokenizer, encoder_model, dataloader, output_file="results_decoder_spanish.txt"):
    try:
        decoder_model.load_state_dict(torch.load("decoder_model_weights_spanish.pth"), strict=False)
        decoder_model.eval()
        print("Loaded trained decoder weights.")
    except FileNotFoundError:
        print("No trained model found. The decoder will run with random weights.")

    total_wer = 0.0
    total_cer = 0.0
    num_samples = 0

    with open(output_file, "w") as file:
        file.write("Speech Recognition Evaluation Results\n")
        file.write("=" * 60 + "\n")

        for batch_idx, batch in enumerate(dataloader):
            waveforms, sample_rates, transcripts = batch
            waveforms = waveforms.to(device)

            # tokenize transcripts (same as in training)
            labels = tokenizer(transcripts, return_tensors="pt", padding=True).input_ids.to(device)

            # process waveform with HuBERT processor
            sampling_rate = sample_rates[0] if isinstance(sample_rates, list) else sample_rates
            inputs = processor(waveforms.squeeze(1), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_values = inputs["input_values"].squeeze(0)

            # extract features from HuBERT
            with torch.no_grad():
                print(f"Batch {batch_idx} - Before Hubert: {torch.cuda.memory_allocated()/1024**2:.2f} MiB")
                outputs = encoder_model(input_values, output_hidden_states=True)
                encoder_features = outputs.hidden_states[9]  # take the ninth layer output
                print(f"Batch {batch_idx} - After Hubert: {torch.cuda.memory_allocated()/1024**2:.2f} MiB")
                max_len = max(100, labels.size(1))
                logits = decoder_model(encoder_features, max_length=max_len)
                print(f"Batch {batch_idx} - After Decoder: {torch.cuda.memory_allocated()/1024**2:.2f} MiB")

            # process each sample in batch
            for i in range(len(transcripts)):
                # get the original and predicted text (clean up special tokens for WER/CER calculation)
                original_text = transcripts[i].replace("[SOS]", "").replace("[EOS]", "").strip()
                
                # prediction
                sample_logits = logits[i:i+1]
                predicted_text = decode_prediction(sample_logits)
                predicted_text = predicted_text.replace("[SOS]", "").replace("[EOS]", "").strip()
                
                sample_wer = wer(original_text, predicted_text)
                sample_cer = cer(original_text, predicted_text)
                total_wer += sample_wer
                total_cer += sample_cer

                file.write(f"🔹 Sample {num_samples + 1}:\n")
                file.write(f"   Ground Truth: {original_text}\n")
                file.write(f"   Predicted: {predicted_text}\n")
                file.write(f"   WER: {sample_wer:.4f} | CER: {sample_cer:.4f}\n")
                file.write("-" * 60 + "\n")
                
                num_samples += 1

        avg_wer = total_wer / num_samples
        avg_cer = total_cer / num_samples
        file.write(f"Evaluation Completed: Avg WER: {avg_wer:.4f} | Avg CER: {avg_cer:.4f}\n")

    print(f"Results saved to {output_file}")
    return avg_wer, avg_cer

avg_wer, avg_cer = evaluate_decoder(decoder_model, processor, tokenizer, model, dataloader)
print(f"Final Results: Avg WER: {avg_wer:.4f} | Avg CER: {avg_cer:.4f}")