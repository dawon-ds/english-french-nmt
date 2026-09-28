import argparse
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from gensim.models import KeyedVectors
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.nmt_rnn import EncoderLSTM_Att, DecoderLSTM_Att
from utils.data_prepro import TranslationDataset, collate_translation_batch
from utils.text_prepro import build_vocab, load_nmt_pair_data, text_to_indices


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_embedding_matrix(vocab, embedding_file, embedding_dim):
    vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab["<pad>"]] = 0
    for word, idx in vocab.items():
        key = word if word in vectors else word.lower()
        clean_key = re.sub(r"[^0-9a-zA-Z]+", "", key)
        if key in vectors:
            matrix[idx] = vectors[key]
        elif clean_key in vectors:
            matrix[idx] = vectors[clean_key]
        elif clean_key.isdigit() and "1" in vectors:
            matrix[idx] = vectors["1"]
    return matrix


def run_epoch(encoder, decoder, loader, criterion, device, teacher_forcing_ratio, optimizers=None):
    training = optimizers is not None
    encoder.train(training)
    decoder.train(training)
    total_loss = 0.0

    for source, target, source_lengths, target_lengths in tqdm(loader, leave=False):
        source, target = source.to(device), target.to(device)
        if training:
            for optimizer in optimizers:
                optimizer.zero_grad()

        encoder_output, encoder_hidden, encoder_cell = encoder(source, source_lengths)
        decoder_input = torch.full((source.size(0),), 2, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden.unsqueeze(0)
        decoder_cell = encoder_cell
        loss = 0.0
        max_length = int(target_lengths.max())
        use_teacher_forcing = training and random.random() < teacher_forcing_ratio

        for step in range(max_length):
            output, decoder_hidden, decoder_cell = decoder(
                encoder_output, decoder_input, decoder_hidden, decoder_cell
            )
            loss = loss + criterion(output, target[:, step])
            decoder_input = target[:, step] if use_teacher_forcing else output.argmax(dim=1).detach()

        if training:
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        total_loss += loss.item() / source.size(0)

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmt_rnn.yaml")
    parser.add_argument("--output", default="runs/nmt_attention")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_text, tgt_text = load_nmt_pair_data(cfg["data"]["train_file"], clean=True)
    src_vocab = build_vocab(src_text, cfg["model"]["vocab_size"])
    tgt_vocab = build_vocab(tgt_text, cfg["model"]["vocab_size"])
    src_ids = text_to_indices(src_text, src_vocab)
    tgt_ids = text_to_indices(tgt_text, tgt_vocab)

    pairs = list(zip(src_ids, tgt_ids))
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - cfg["training"]["validation_ratio"]))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    train_ds = TranslationDataset(*zip(*train_pairs))
    val_ds = TranslationDataset(*zip(*val_pairs))
    batch = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_translation_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, collate_fn=collate_translation_batch, drop_last=True)

    embedding = build_embedding_matrix(src_vocab, cfg["embedding_file"], cfg["model"]["embedding_dim"])
    encoder = EncoderLSTM_Att(len(src_vocab), cfg["model"]["embedding_dim"], cfg["model"]["hidden_size"], embedding).to(device)
    decoder = DecoderLSTM_Att(cfg["model"]["embedding_dim"], cfg["model"]["hidden_size"], len(tgt_vocab)).to(device)
    criterion = nn.NLLLoss(ignore_index=0)
    enc_opt = Adam(encoder.parameters(), lr=cfg["training"]["learning_rate"])
    dec_opt = Adam(decoder.parameters(), lr=cfg["training"]["learning_rate"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "source_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(output / "target_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    np.save(output / "source_embedding.npy", embedding)

    best = float("inf")
    start = time.time()
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        train_loss = run_epoch(encoder, decoder, train_loader, criterion, device, cfg["model"]["teacher_forcing_ratio"], (enc_opt, dec_opt))
        with torch.no_grad():
            val_loss = run_epoch(encoder, decoder, val_loader, criterion, device, 0.0)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, output / "best.pth")
    print(f"Training time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()

