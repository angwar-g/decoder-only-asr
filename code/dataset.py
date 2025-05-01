"""This script processes Spanish parquet files, downloaded from Hugging Face, 
(https://huggingface.co/datasets/facebook/multilingual_librispeech/tree/main/spanish) by converting 
audio from OGG to WAV, and saving the processed audio along with 
corresponding metadata (transcripts with SOS/EOS tokens)."""

import json
import torch
import torchaudio
import pandas as pd
import soundfile as sf
import io
import ffmpeg
from pathlib import Path

# load Spanish vocab (for SOS/EOS tokens)
with open('spanish_vocab.json', 'r', encoding='utf-8') as vocab_file:
    spanish_vocab_dict = json.load(vocab_file)
SOS_ID = spanish_vocab_dict["[SOS]"]
EOS_ID = spanish_vocab_dict["[EOS]"]

def load_ogg_audio(audio_data):
    out, err = (
        ffmpeg
        .input('pipe:0', format='ogg')
        .output('pipe:1', format='wav')
        .run(input=audio_data, capture_stdout=True, capture_stderr=True)
    )
    waveform, sample_rate = sf.read(io.BytesIO(out))
    return waveform, sample_rate

def preprocess_and_save(parquet_files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]
    
    # load and concatenate all parquet files
    df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    
    metadata = []
    for idx in range(len(df)):
        audio_data = df.loc[idx, "audio"]["bytes"]
        transcript = df.loc[idx, "transcript"]
        transcript = f"[SOS] {transcript.lower()} [EOS]"
        
        # process audio
        waveform, sample_rate = load_ogg_audio(audio_data)
        if waveform.ndim == 1:
            waveform = waveform[None, :]  # Add channel dimension
        waveform = torch.from_numpy(waveform).float()
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # save processed waveform
        audio_path = output_dir / f"audio_{idx}.pt"
        torch.save(waveform, audio_path)
        
        # store metadata
        metadata.append({
            "audio_path": str(audio_path),
            "transcript": transcript,
            "sample_rate": sample_rate
        })
    
    # save metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Preprocessed dataset saved to {output_dir}")

if __name__ == "__main__":
    parquet_files = [
        'data/test-00000-of-00001.parquet'
    ]
    preprocess_and_save(parquet_files, "spanish_test_00")