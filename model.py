import torch.nn as nn
from torch import tanh, sigmoid
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, tgt_size):
        super(classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.encoder_lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, bias=False, batch_first=True)
        self.projection_r = nn.Linear(in_features=embedding_size, out_features=tgt_size)

    def forward(self, texts, t_lengths):
        x = self.embedding(texts)
        t_lengths_sorted, indices = torch.sort(t_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        x_sorted = x[indices]
        packed = pack_padded_sequence(x_sorted, t_lengths_sorted, batch_first=True)
        _, (last_hidden, _) = self.encoder_lstm(packed)
        output = sigmoid(self.projection_r(last_hidden.squeeze()[desorted_indices]))
        return output


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.encoder_lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, bias=False, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size, bias=False)
        self.projection_y = nn.Linear(in_features=2*embedding_size, out_features=vocab_size, bias=False)
        self.projection_s = nn.Linear(in_features=2*embedding_size, out_features=embedding_size, bias=False)
        self.projection_att1 = nn.Linear(in_features=embedding_size, out_features=1, bias=False)
        self.projection_att2 = nn.Linear(in_features=embedding_size, out_features=1, bias=False)

    def forward(self, mr, ref, mr_lengths):
        x = self.embedding(mr)
        tgt = self.embedding(ref)

        # encoder
        packed = pack_padded_sequence(x, mr_lengths, batch_first=True, enforce_sorted=False)
        packed_hidden_states, (last_hidden, last_cell) = self.encoder_lstm(packed)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)

        # decoder
        mask = torch.zeros((len(mr_lengths), max(mr_lengths), 1))
        for i in range(len(mr_lengths)):
            mask[i, mr_lengths[i]:, :] = 1

        s0 = last_hidden.squeeze()
        m0 = last_cell.squeeze()
        output = []
        for y_t in torch.split(tgt, 1, dim=1):
            y = y_t.squeeze()
            s, m, y_ = self.decode_step(y, hidden_states, mask, s0, m0)
            output.append(y_)
            s0 = s
            m0 = m
        result = torch.stack(output, 1)
        return result

    def generate(self, mr, lengths):
        # encoder
        x = self.embedding(mr)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden_states, (last_hidden, last_cell) = self.encoder_lstm(packed)
        hidden_states, _ = pad_packed_sequence(packed_hidden_states, batch_first=True)

        # beam search decoder
        mask = torch.zeros((len(lengths), max(lengths), 1))
        for i in range(len(lengths)):
            mask[i, lengths[i]:, :] = 1

        s0 = last_hidden.squeeze()
        m0 = last_cell.squeeze()
        ###

    def decode_step(self, y, hidden_states, att_mask, s, m):
        e = tanh(self.projection_att1(hidden_states)) * tanh(self.projection_att2(s))[:, None, :]
        e_ = e.masked_fill(att_mask.bool(), -1e9)
        alpha = softmax(e_, dim=-2)
        c = torch.matmul(hidden_states.transpose(-1, -2), alpha).squeeze()
        d = self.projection_s(torch.cat((y, c), -1))
        s_, m_ = self.decoder_cell(d, (s, m))
        y_ = softmax(self.projection_y(torch.cat((s_, c), -1)), -1)
        return s_, m_, y_
