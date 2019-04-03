# -*- coding: utf-8 -*-

#importing the dependcies

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

#Defining the architecture of the model
class RNNMod(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.n_hidden)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq):
        bs = seq.size(1) # batch size
        self.h = self.init_hidden(bs) # initialize hidden state of GRU
        embs = self.emb(seq)
        rnn_out, self.h = self.rnn(embs, self.h) # rnn returns hidden state of all timesteps as well as hidden state at last timestep
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(self.h[-1]) # h[-1] contains hidden state of last timestep
        return F.log_softmax(outp,dim=-1)
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1,batch_size,self.n_hidden)))
