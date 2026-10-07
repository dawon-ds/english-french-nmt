# English–French Neural Machine Translation

LSTM Encoder–Decoder with dot-product attention for English → French neural machine translation. The project implements attention directly and compares generated translations with reference sentences using BLEU-4.

[Portfolio](https://incredible-march-0ef.notion.site/English-French-Neural-Machine-Translation-3e968564df5a817f92a4f275a892ee8e)

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

## Implementation Organization
- Removed course-specific `DL_Lecture.*` import paths.
- Replaced absolute local Windows paths with relative project paths.
- Removed unrelated image/mask dataset utilities from the translation data module.
- Removed hard-coded run timestamps from evaluation.
- Added a reproducible repository structure, dependency file, and Git ignore rules.
- Kept the original LSTM + dot-product attention logic and assignment hyperparameters.

## Data & Implementation

The assignment reports **108,674 training sentence pairs** and **27,168 test pairs**. The training script reserves 10% of the supplied training pairs for validation after shuffling.

Source and target vocabularies are built from the supplied training file. The encoder uses pretrained source Word2Vec embeddings; the decoder computes dot-product attention over encoder outputs to build a context vector at each step.

Training uses teacher forcing with ratio 0.5, ignores padding labels in NLL loss, and saves the checkpoint with lowest validation loss. Evaluation uses greedy decoding and NLTK corpus BLEU-4.

Artifacts are written under `runs/nmt_attention/`: `best.pth`, `source_vocab.pkl`, `target_vocab.pkl`, and `source_embedding.npy`.

## Evaluation Limitations & Review

The reported BLEU-4 of **0.416** and training time are historical assignment values, not results reproduced in this update.

The current evaluator drops the final incomplete test batch, uses reference lengths to determine the decoding budget, and does not stop each sentence at its first EOS token. BLEU therefore needs to be interpreted with those evaluation choices in mind. Vocabularies are also constructed before the validation split.

A stronger evaluation would retain all test examples, stop decoding at EOS with a reference-independent maximum length, and report the precise tokenization and scoring setup.

Implementing attention directly provided practice with encoder/decoder states, context-vector construction, teacher forcing, and sequence-level evaluation.
