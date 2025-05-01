# Decoder-Only Transformers for Cross-Lingual Transfer Learning in Automatic Speech Recognition

This zip folder contains the Python scripts for training and evaluating the decoder-only transformer architecture for cross-lingual transfer learning in Automatic Speech Recognition. The project investigates the model's performance when pre-trained on English data and fine-tuned on a low-resource language (Spanish).

The contents of the folder are as follows:
- requirements.txt (Contains all the Python libraries required to run the scripts)
- decoder.py (Implementation of pre-training the decoder-only transformer on English ASR Data)
- vocab.json (English Vocabulary needed to train the model on English ASR Data)
- spanish_finetune.py (Implementation of fine-tuning the pre-trained decoder-only transformer on Spanish ASR Data)
- spanish_vocab.json (Spanish Vocabulary needed to fine-tune the model on Spanish ASR Data)
- decoder_test.py (Python script to evaluate the pre-trained English decoder)
- spanish_test.py (Python script to evaluate the final fine-tuned Spanish decoder)
- dataset.py (Python script to pre-process the Spanish parquet files to use them for fine-tuning)
- baseline.py (Implementation of the baseline for comparing the final model)

Additional Resources Needed:
- HuBERT Base Model - https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
- Spanish Parquet Files - https://huggingface.co/datasets/facebook/multilingual_librispeech/tree/main/spanish


