import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from collections import Counter
import json
import re
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
# import torchtext
from torchtext.vocab import GloVe



import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        return outputs, (hidden.unsqueeze(0), cell.unsqueeze(0))

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = encoder_outputs.mean(dim=1).unsqueeze(0)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        
        return output, (hidden, cell)



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.6):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_outputs, (hidden, cell)= self.encoder(src)
        output = trg[0, :]  
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(output, hidden, cell, encoder_outputs)
            outputs[t] = output.squeeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t, :] if teacher_force else top1)
        return outputs
    

# # Maximum sequence lengths
# max_input_length = train_dataset.preprocessed_data[0][0].size(0)  # Assuming the first input tensor contains the maximum sequence length
# max_output_length = train_dataset.preprocessed_data[0][1].size(0)  # Assuming the first output tensor contains the maximum sequence length

# Define hyperparameters

device = torch.device('cpu')

# Initialize the encoder
# encoder = Encoder(input_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix)
def make_model_1(input_vocab_size, output_vocab_size, weights_path, embedding_matrix):
    enc_hid_dim = 256
    dec_hid_dim = 512
    dropout = 0.5
    embedding_dim = 100
    encoder = Encoder(input_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix)
    decoder = Decoder(output_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout)
    seq2seq_model = (torch.load(weights_path, map_location='cpu'))
    # seq2seq_model = Seq2Seq(encoder, decoder)
    return seq2seq_model

# encoder = Encoder(input_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout)
# # Initialize the decoder
# decoder = Decoder(output_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout)

# # Initialize the Seq2Seq model
# seq2seq_model = Seq2Seq(encoder, decoder, device)

# # Move the model to the appropriate device
# model = seq2seq_model.to(device)

# # Print model summary
# print(seq2seq_model)

def beam_search(model, src, k, max_len, sos_idx, eos_idx, device):
    src = src.unsqueeze(1)
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)
        beam = [(torch.tensor([sos_idx], device=device), 0, (hidden, cell))]
        for _ in range(max_len):
            candidates = []
            for seq, score, (hidden, cell) in beam:
                if seq[-1] == eos_idx:
                    candidates.append((seq, score, (hidden, cell)))
                    continue
                output, (hidden, cell) = model.decoder(seq[-1].unsqueeze(0), hidden, cell, encoder_outputs)
                output_probs = F.softmax(output, dim=1)
                topk_scores, topk_idxs = output_probs.topk(k)
                for i in range(k):
                    next_token = topk_idxs[0][i].unsqueeze(0)
                    next_score = score + topk_scores[0][i].item()
                    candidates.append((torch.cat([seq, next_token]), next_score, (hidden, cell)))
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
            if all(candidate[0][-1] == eos_idx for candidate in beam):
                break
        best_seq, _, _ = beam[0]
        return best_seq

def translate_beam(model, data_loader, output_vocab, get_token_from_index, device):
    model.eval()
    model.to(device)
    translated_sentences = []
    targeted_sentences = []
    with torch.no_grad():
        for src_batch in data_loader:
            src_batch = src_batch.to(device)  # Move batch to device
            translated_batch = []
            for src_sent in src_batch:
                translated_sent = beam_search(model, src_sent, k=2, max_len=20, sos_idx=output_vocab['<SOS>'], eos_idx=output_vocab['<EOS>'], device=device)
                translated_batch.append(translated_sent)
            translated_batch = [[get_token_from_index(idx) for idx in sent] for sent in translated_batch]
            translated_sentences.extend(translated_batch)
    return translated_sentences


