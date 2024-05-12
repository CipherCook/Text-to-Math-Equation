import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size_encoder, hidden_size_decoder, dropout):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embedding_size)
#         self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=False)
        self.rnn = nn.LSTM(embedding_size, hidden_size_encoder, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, input_seq):
        embedded = self.dropout_layer(self.embedding(input_seq))
        outputs, (hidden_states_encoder, cell_states_encoder) = self.rnn(embedded)
        hidden_states_encoder = torch.cat((hidden_states_encoder[-2,:,:], hidden_states_encoder[-1,:,:]), dim=1)
        cell_states_encoder = torch.cat((cell_states_encoder[-2,:,:], cell_states_encoder[-1,:,:]), dim=1)
        return outputs, (hidden_states_encoder.unsqueeze(0), cell_states_encoder.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self, hidden_size_encoder, hidden_size_decoder):
        super().__init__()
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        
        self.attn = nn.Linear((hidden_size_encoder * 2) + hidden_size_decoder, hidden_size_decoder)
        self.v = nn.Linear(hidden_size_decoder, 1, bias=False)
        
    def forward(self, hidden_states_decoder, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden_states_decoder = hidden_states_decoder.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden_states_decoder, encoder_outputs), dim=2))) 
        attention_scores = self.v(energy).squeeze(2)
        
        return F.softmax(attention_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size_encoder, hidden_size_decoder, dropout, attention):
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.dropout = dropout
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        self.rnn = nn.LSTM((hidden_size_encoder * 2) + embedding_size, hidden_size_decoder)
        self.out = nn.Linear((hidden_size_encoder * 2) + hidden_size_decoder + embedding_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, input, hidden_states_decoder, cell_states_decoder, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout_layer(self.embedding(input))
        
        attention_weights = self.attention(hidden_states_decoder[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(attention_weights, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden_states_decoder, cell_states_decoder) = self.rnn(rnn_input, (hidden_states_decoder, cell_states_decoder))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        
        return output, (hidden_states_decoder, cell_states_decoder)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.6):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden_states_encoder, cell_states_encoder) = self.encoder(src)
        output = trg[0, :]  
        for t in range(1, max_len):
            output, (hidden_states_decoder, cell_states_decoder) = self.decoder(output, hidden_states_encoder, cell_states_encoder, encoder_outputs)
            outputs[t] = output.squeeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t, :] if teacher_force else top1)
        return outputs


device = torch.device('cpu')
# Initialize the encoder
# encoder = Encoder(input_vocab_size, embedding_dim, enc_hid_dim, dec_hid_dim, dropout, embedding_matrix)
def make_model_2(input_vocab_size, output_vocab_size, weights_path, embedding_matrix):
    enc_hidden_dim = 256
    dec_hidden_dim = 512
    dropout = 0.5
    embedding_dim = 100
    attention = Attention(enc_hidden_dim, dec_hidden_dim)
    encoder = Encoder(input_vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout)
    decoder = Decoder(output_vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout, attention)
    seq2seq_model = Seq2Seq(encoder, decoder, device)
    seq2seq_model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    # seq2seq_model = Seq2Seq(encoder, decoder)
    return seq2seq_model

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
                    next_score = score * topk_scores[0][i].item()
                    candidates.append((torch.cat([seq, next_token]), next_score, (hidden, cell)))
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
            if all(candidate[0][-1] == eos_idx for candidate in beam):
                break
        best_seq, _, _ = beam[0]
        return best_seq


def translate_beam2(model, data_loader, device, k, get_key_output, output_vocab):
    model.eval()
    model.to(device)
    translated_sentences = []
    targeted_sentences = []
    with torch.no_grad():
        i = 0
        for src_batch in data_loader:
            print(i)
            i+=1
            src_batch = src_batch.to(device)  # Move batch to device
            translated_batch = []

            for src_sent in src_batch:
                translated_sent = beam_search(model, src_sent, k, max_len=20, sos_idx=output_vocab['<SOS>'], eos_idx=output_vocab['<EOS>'], device=device)
                translated_batch.append(translated_sent)
            print(".2")
            translated_batch = [[get_key_output(idx) for idx in sent] for sent in translated_batch]
            translated_sentences.extend(translated_batch)
    return translated_sentences, targeted_sentences
