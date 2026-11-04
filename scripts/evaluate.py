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
    p.add_argument("--max-decode-length", type=int, default=100, help="Maximum generated tokens per sentence")
    args = p.parse_args()
    if args.max_decode_length < 1:
        p.error("--max-decode-length must be positive")

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
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], collate_fn=collate_translation_batch, drop_last=False)

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
            finished = torch.zeros(source.size(0), dtype=torch.bool, device=device)
            eos_id = tgt_vocab["</s>"]
            for _ in range(args.max_decode_length):
                out, hidden, cell = decoder(enc_out, token, hidden, cell)
                predicted = out.argmax(1)
                for i, idx in enumerate(predicted.tolist()):
                    if finished[i]:
                        continue
                    if idx == eos_id:
                        finished[i] = True
                    elif idx not in {tgt_vocab["<pad>"], tgt_vocab["<s>"]}:
                        words[i].append(id2word.get(idx, "<unk>"))
                if finished.all():
                    break
                token = torch.where(finished, torch.full_like(predicted, eos_id), predicted)
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



