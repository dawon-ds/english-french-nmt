# English–French Neural Machine Translation

[Portfolio Page](https://app.notion.com/p/3e968564df5a817f92a4f275a892ee8e)

LSTM Encoder–Decoder with dot-product attention for English → French neural machine translation. The project implements attention directly and compares generated translations with reference sentences using BLEU-4.


## Project Overview

- **Task:** English–French neural machine translation
- **Model:** LSTM encoder–decoder with dot-product attention
- **Framework:** PyTorch
- **Evaluation:** BLEU-4

## Data

### Data & Implementation

The assignment reports **108,674 training sentence pairs** and **27,168 test pairs**. The training script reserves 10% of the supplied training pairs for validation after shuffling.

Source and target vocabularies are built from the supplied training file. The encoder uses pretrained source Word2Vec embeddings; the decoder computes dot-product attention over encoder outputs to build a context vector at each step.

Training uses teacher forcing with ratio 0.5, ignores padding labels in NLL loss, and saves the checkpoint with lowest validation loss. Evaluation uses greedy decoding and NLTK corpus BLEU-4.

Artifacts are written under `runs/nmt_attention/`: `best.pth`, `source_vocab.pkl`, `target_vocab.pkl`, and `source_embedding.npy`.

## Approach

### Model

- LSTM Encoder–Decoder
- Dot-product attention implemented with `torch.bmm`
- 300-dimensional pre-trained Word2Vec source embeddings
- Hidden size: 1000
- Teacher forcing ratio: 0.5
- Batch size: 128
- Adam, learning rate 0.001
- 10 epochs
- BLEU-4 evaluation

## Results

**Historically reported BLEU-4: 0.416** · **Training time: 642.13 minutes**

This is the submitted experiment result under the evaluation conditions described in Limitations, not a newly reproduced score.

## Repository Structure

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

## Usage

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the English–French train/test TSV files at `data/eng-fra_train.txt` and `data/eng-fra_test.txt`. Download `GoogleNews-vectors-negative300.bin` separately and either place it in the repository root or update `embedding_file` in `config/nmt_rnn.yaml`. Dataset files, the binary embedding file, and trained checkpoints are intentionally excluded from Git.

### Training

```bash
python -m scripts.train --config config/nmt_rnn.yaml
```

### Evaluation

```bash
python -m scripts.evaluate --config config/nmt_rnn.yaml --run-dir runs/nmt_attention
```

## Limitations

The updated evaluator includes the final incomplete test batch and uses autoregressive decoding with EOS stopping and a configurable, reference-independent maximum length (`--max-decode-length`, default 100). **The historical BLEU-4 result was reported with the earlier evaluation procedure and has not been recalculated under this corrected protocol.** Vocabularies are still constructed before the validation split. A stronger comparison would also report the exact tokenization and scoring setup.

## Review

Implementing attention directly provided practice with encoder/decoder states, context-vector construction, teacher forcing, and sequence-level evaluation.

## Data and Third-party Resources

- **Data / embeddings:** the English–French sentence pairs and pretrained Word2Vec files are external inputs, not included in the repository. Their original distribution and usage conditions must be checked before redistribution.
- **Evaluation:** the README's historical BLEU value should not be interpreted as newly reproduced using the updated evaluator.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
