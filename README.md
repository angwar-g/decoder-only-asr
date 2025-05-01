# Decoder-Only Transformers for Cross-Lingual Transfer Learning in Automatic Speech Recognition

This repository accompanies the final-year dissertation ["Decoder-Only Transformers for Cross-Lingual Transfer Learning in ASR"](./report.pdf) at the University of Edinburgh. It explores the potential of decoder-only transformer architectures for speech recognition across languages, focusing on the transfer from English (a high-resource language) to Spanish (a lower-resource language).

## Project Overview
Traditional ASR systems often use encoder-only or encoder-decoder transformer models. In contrast, this project investigates decoder-only transformers, which offer a unified architecture for language modelling and speech transcription. This dissertation examines whether this architecture can generalise across languages using cross-lingual transfer learning.

## Research Question
Can a decoder-only transformer model, pre-trained on English ASR data and fine-tuned on limited Spanish data, achieve competitive cross-lingual performance compared to encoder-based baselines?

## Methodology
- Feature Extraction: Pre-trained [HuBERT](https://ieeexplore.ieee.org/abstract/document/9585401) Base (frozen) extracts contextualised speech embeddings.
- Decoder-Only Transformer: 12-layer transformer model with 8 attention heads.
- Pre-training: English ASR using the [LibriSpeech](https://ieeexplore.ieee.org/abstract/document/7178964) dataset (960 hours).
- Fine-tuning: Spanish ASR using [Multilingual LibriSpeech](https://arxiv.org/abs/2012.03411) (9-240 hours).

## Author
Amishi Gangwar <br>
4th Year Artificial Intelligence and Computer Science | University of Edinburgh <br>
Supervisor: [Dr. Hao Tang](https://homepages.inf.ed.ac.uk/htang2/)

