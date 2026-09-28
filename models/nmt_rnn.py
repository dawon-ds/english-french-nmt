import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

        weights = torch.tensor(embedding_matrix.astype(np.float32))
        with torch.no_grad():
            self.embedding.weight.copy_(weights)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)
        encoder_output, _ = pad_packed_sequence(packed_output)
        hidden = hidden.permute(1, 0, 2).reshape(inputs.size(0), -1)
        return encoder_output, hidden, cell


class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, encoder_output, token, hidden, cell):
        embedded = self.embedding(token)
        _, (hidden, cell) = self.lstm(embedded.unsqueeze(0), (hidden, cell))

        # Dot-product attention
        encoder_states = encoder_output.permute(1, 0, 2)
        attention_scores = torch.bmm(
            encoder_states, hidden.permute(1, 2, 0)
        )
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(encoder_states * attention_weights, dim=1).unsqueeze(0)

        attended_hidden = torch.cat((hidden, context), dim=-1)
        attended_hidden = torch.tanh(self.attention_combine(attended_hidden))
        logits = self.log_softmax(self.output(attended_hidden[0]))
        return logits, attended_hidden, cell


EncoderLSTM_Att = EncoderLSTMWithAttention
DecoderLSTM_Att = DecoderLSTMWithAttention

