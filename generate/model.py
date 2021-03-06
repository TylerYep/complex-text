import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn

class GenerateModel(nn.Module):
    def __init__(self, n_chars, hidden_dim=30, n_layers=2, embed_dim=64, lvl=2):
        super(GenerateModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_chars = n_chars
        self.embed_dim = embed_dim
        self.param_names = ['lvl', 'hidden_dim', 'n_layers', 'embed_dim', 'n_chars']
        self.param_vals = [lvl, hidden_dim, n_layers, embed_dim, n_chars]
        self.params = dict(zip(self.param_names, self.param_vals))

        self.fname = '_'.join([n + str(p) for n, p in self.params.items()]) + 'model.pth.tar'

        self.best_dev = 0
        self.epoch = 0

        self.embedding = nn.Embedding(n_chars, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers)
        self.hidden_to_out = nn.Linear(hidden_dim, n_chars)


    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))

    def forward(self, X, h, bs=1):
        X = self.embedding(X)
        X = X.view(-1, bs, self.embed_dim)
        X, hidden = self.lstm(X, h)
        X = self.hidden_to_out(X)
        X = nn.functional.log_softmax(X, dim=2)
        return X, hidden

    def save(self):
        state={'state_dict': self.state_dict(), 'params': self.params, 'epoch':self.epoch}
        torch.save(state, self.fname)



