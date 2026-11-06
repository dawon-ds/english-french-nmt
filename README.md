# English–French Neural Machine Translation

[Portfolio Page](https://app.notion.com/p/3e968564df5a817f92a4f275a892ee8e)

An LSTM Encoder–Decoder for English → French translation, using pre-trained Word2Vec embeddings and dot-product attention to address the limitations of a fixed-length sentence representation.

## Project Overview

- **Task:** English–French neural machine translation
- **Model:** LSTM encoder–decoder with dot-product attention
- **Framework:** PyTorch
- **Evaluation:** BLEU-4

## Data & Implementation

The assignment reports **108,674 training sentence pairs** and **27,168 test pairs**. The training script reserves 10% of the supplied training pairs for validation after shuffling.

Source and target vocabularies are built from the supplied training file. The encoder uses pretrained source Word2Vec embeddings; the decoder computes dot-product attention over encoder outputs to build a context vector at each step.

Training uses teacher forcing with ratio 0.5, ignores padding labels in NLL loss, and saves the checkpoint with lowest validation loss. Evaluation uses greedy decoding and NLTK corpus BLEU-4.

Artifacts are written under `runs/nmt_attention/`: `best.pth`, `source_vocab.pkl`, `target_vocab.pkl`, and `source_embedding.npy`.

## Experiment Settings

- **Architecture:** LSTM Encoder–Decoder with dot-product attention (`torch.bmm`)
- **Source embedding:** 300-dimensional pre-trained Word2Vec
- **Hidden size:** 1,000
- **Teacher forcing ratio:** 0.5
- **Batch size:** 128
- **Optimizer:** Adam · learning rate 0.001
- **Epochs:** 10
- **Evaluation:** BLEU-4

## Results

- **Reported BLEU-4:** 0.416
- **Training time:** approximately 642.13 minutes

The BLEU-4 score is from the original assignment and was not recalculated using the updated evaluator (see Limitations).

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

## Data and Third-party Resources

- **Data / embeddings:** the English–French sentence pairs and pretrained Word2Vec files are external inputs, not included in the repository. Their original distribution and usage conditions must be checked before redistribution.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
