# English–French Neural Machine Translation

LSTM Encoder–Decoder with dot-product attention for English → French neural machine translation. This repository is a cleaned, portfolio-ready version of HW7 while preserving the assignment's core model and experimental settings.

## Model
- LSTM Encoder–Decoder
- Dot-product attention implemented with `torch.bmm`
- 300-dimensional pre-trained Word2Vec source embeddings
- Hidden size: 1000
- Teacher forcing ratio: 0.5
- Batch size: 128
- Adam, learning rate 0.001
- 10 epochs
- BLEU-4 evaluation

The submitted assignment reported **BLEU-4 = 0.416** and approximately **642.13 minutes** of training time.

## Project structure
```text
.
├── config/nmt_rnn.yaml
├── data/
│   └── README.md
├── models/nmt_rnn.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── data_prepro.py
│   └── text_prepro.py
├── requirements.txt
└── README.md
```

## Setup
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the English–French train/test TSV files at `data/eng-fra_train.txt` and `data/eng-fra_test.txt`. Download `GoogleNews-vectors-negative300.bin` separately and either place it in the repository root or update `embedding_file` in `config/nmt_rnn.yaml`. Dataset files, the binary embedding file, and trained checkpoints are intentionally excluded from Git.

## Train
```bash
python -m scripts.train --config config/nmt_rnn.yaml
```

## Evaluate
```bash
python -m scripts.evaluate --config config/nmt_rnn.yaml --run-dir runs/nmt_attention
```

## What was cleaned from the coursework version
- Removed course-specific `DL_Lecture.*` import paths.
- Replaced absolute local Windows paths with relative project paths.
- Removed unrelated image/mask dataset utilities from the translation data module.
- Removed hard-coded run timestamps from evaluation.
- Added a reproducible repository structure, dependency file, and Git ignore rules.
- Kept the original LSTM + dot-product attention logic and assignment hyperparameters.


