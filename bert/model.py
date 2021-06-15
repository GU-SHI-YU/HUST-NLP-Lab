from math import e
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AlbertModel, BertTokenizer


class CWS(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        self.tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
        self.word_embeds = AlbertModel.from_pretrained("clue/albert_chinese_tiny")

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(4, batch_first=True)

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, input_ids, input_mask):
        batch_size = input_ids.size(0)
        embeds = self.word_embeds(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True, output_attentions=True)
        # idx->embedding
        all_hidden_states, all_attention = embeds[-2:]
        embeds = all_hidden_states[-2]

        # LSTM forward
        self.hidden = self.init_hidden(batch_size, input_ids.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, input_ids, label, input_mask, output_mask):
        # print(input_ids)
        # print(label)
        # print(input_mask)
        # print(output_mask)
        emissions = self._get_lstm_features(input_ids, input_mask)
        loss = -self.crf(emissions, label, output_mask, reduction='mean')
        return loss

    def infer(self, input_ids, input_mask, output_mask):
        emissions = self._get_lstm_features(input_ids, input_mask)
        return self.crf.decode(emissions, output_mask)
