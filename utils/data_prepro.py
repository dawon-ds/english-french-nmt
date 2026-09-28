import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return torch.tensor(self.source[index], dtype=torch.long), torch.tensor(
            self.target[index], dtype=torch.long
        )


def collate_translation_batch(batch):
    source, target = zip(*batch)
    source_lengths = torch.tensor([len(x) for x in source], dtype=torch.long)
    target_lengths = torch.tensor([len(x) for x in target], dtype=torch.long)
    source = pad_sequence(source, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)
    return source, target, source_lengths, target_lengths


TranslateDataset = TranslationDataset
collate = collate_translation_batch


