import random

import torch 
import torch.nn as nn

from scripts.config import *

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, p):
        super(Encoder, self).__init__()
        
        self.embedding_question = nn.Embedding(input_size, embedding_dim)
        self.embedding_context = nn.Embedding(input_size, embedding_dim)

        self.dropout_q = nn.Dropout(p)
        self.dropout_c = nn.Dropout(p)

        self.bilstm_q = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=1-p, bidirectional=True)
        self.bilstm_c = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=1-p, bidirectional=True)

        self.fc_hidden_q = nn.Linear(hidden_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size*2, hidden_size)

        self.attn_bigru = AttnBiGRU(embedding_dim, hidden_size, num_layers, p)

    def forward(self, context, question):
        out_c = self.dropout_c(self.embedding_context(context))
        out_q = self.dropout_q(self.embedding_question(question))

        out_q, h_q = self.bilstm_q(out_q)

        h_q = self.fc_hidden_q(torch.cat((h_q[:1], h_q[1:2]), dim=2))

        out_attn, hidden_attn = self.attn_bigru(out_c, out_q, h_q)

        out_c, h_c = self.bilstm_c(out_c)

        hidden = torch.cat((hidden_attn, h_c), dim=2)
        hidden = self.linear(hidden)

        return hidden

class AttnBiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, p):
        super(AttnBiGRU, self).__init__()
        self.attn_linear = nn.Linear(hidden_size + embedding_dim, 1)
        self.attn_softmax = nn.Softmax(dim=0)

        self.bigru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=1-p, bidirectional=True)

        self.num_layers = num_layers

    def forward(self, context, hidden_outputs, hidden_question):
        
        sequence_length = context.size(0)
        h_reshape = hidden_question.repeat(sequence_length, 1, 1)

        energy = self.attn_softmax(self.attn_linear(torch.cat((h_reshape, context), dim=2)))
        
        energy = energy.permute(1, 2, 0)
        context = context.permute(1, 0, 2)

        cv = torch.bmm(energy, context).permute(1, 0, 2)

        hidden_question = hidden_question.repeat(self.num_layers * 2, 1, 1)

        out, hidden = self.bigru(cv, hidden_question)

        return out, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, embedding_dim, p):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(p)
        self.bigru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=1-p, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden):

        out = self.dropout(self.embedding(input.unsqueeze(0)))

        out, hidden = self.bigru(out, hidden)

        out = self.fc(out.squeeze(0))

        return out, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, vocab):
        super(Seq2Seq, self).__init__()
        
        self.vocab = vocab
        self.encoder = Encoder(input_size, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_DIM, P)
        self.decoder = Decoder(input_size, output_size, HIDDEN_SIZE, NUM_LAYERS, EMBEDDING_DIM, P)

    def forward(self, context, question, answer, teacher_forcing_ratio=0.5):

        answr_len, N = answer.size()
        vocab_size = len(self.vocab.index2word)

        outputs = torch.zeros(answr_len, N, vocab_size).to(device=answer.device)

        hidden = self.encoder(context, question)

        input = answer[0]

        for t in range(1, answr_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            best_guess = output.argmax(1)
            input = answer[t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs
