import collections
import re
import unicodedata


def clean_text(text, allow_french=False):
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    if allow_french:
        return re.sub(r"[^a-zA-ZÀ-ÿ0-9\s'?!]", "", text)
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def load_nmt_pair_data(file_path, clean=False):
    source_texts, target_texts = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            source, target = parts
            if clean:
                source = clean_text(source, allow_french=False)
                target = clean_text(target, allow_french=True)
            source_texts.append(source)
            target_texts.append(target)
    return source_texts, target_texts


def build_vocab(sentences, vocab_size):
    words = [word for sentence in sentences for word in sentence.split()]
    counts = collections.Counter(words)
    vocab = {word: i + 4 for i, (word, _) in enumerate(counts.most_common(vocab_size))}
    vocab.update({"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3})
    return vocab


def text_to_indices(texts, word_to_id, use_unk=True):
    sequences = []
    for text in texts:
        ids = [word_to_id["<s>"]]
        for word in text.split():
            if word in word_to_id:
                ids.append(word_to_id[word])
            elif use_unk:
                ids.append(word_to_id["<unk>"])
        ids.append(word_to_id["</s>"])
        sequences.append(ids)
    return sequences
