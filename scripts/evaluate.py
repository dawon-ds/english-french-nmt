import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.nmt_rnn import EncoderLSTM_Att, DecoderLSTM_Att
from utils.data_prepro import TranslationDataset, collate_translation_batch
from utils.text_prepro import load_nmt_pair_data, text_to_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/nmt_rnn.yaml")
    p.add_argument("--run-dir", default="runs/nmt_attention")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(run / "source_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(run / "target_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    id2word = {v: k for k, v in tgt_vocab.items()}
    embedding = np.load(run / "source_embedding.npy")

    src_text, tgt_text = load_nmt_pair_data(cfg["data"]["test_file"], clean=True)
    dataset = TranslationDataset(text_to_indices(src_text, src_vocab), text_to_indices(tgt_text, tgt_vocab))
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], collate_fn=collate_translation_batch, drop_last=True)

    encoder = EncoderLSTM_Att(len(src_vocab), cfg["model"]["embedding_dim"], cfg["model"]["hidden_size"], embedding).to(device)
    decoder = DecoderLSTM_Att(cfg["model"]["embedding_dim"], cfg["model"]["hidden_size"], len(tgt_vocab)).to(device)
    ckpt = torch.load(run / "best.pth", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    hypotheses = []

    with torch.no_grad():
        for source, target, lengths, target_lengths in tqdm(loader):
            source = source.to(device)
            enc_out, enc_hidden, enc_cell = encoder(source, lengths)
            token = torch.full((source.size(0),), 2, dtype=torch.long, device=device)
            hidden, cell = enc_hidden.unsqueeze(0), enc_cell
            words = [[] for _ in range(source.size(0))]
            for _ in range(int(target_lengths.max())):
                out, hidden, cell = decoder(enc_out, token, hidden, cell)
                token = out.argmax(1)
                for i, idx in enumerate(token.tolist()):
                    word = id2word.get(idx, "<unk>")
                    if word not in {"<pad>", "<s>", "</s>"}:
                        words[i].append(word)
            hypotheses.extend(words)

    references = [[sentence.split()] for sentence in tgt_text[:len(hypotheses)]]
    bleu = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4: {bleu:.4f}")
    for ref, hyp in list(zip(references, hypotheses))[:10]:
        print("REF:", " ".join(ref[0]))
        print("HYP:", " ".join(hyp))
        print("-" * 40)


if __name__ == "__main__":
    main()


