import pandas as pd
from model import GenerateModel
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from data import DataG
import sys
sys.path.insert(0,"/home/harry/Dropbox/Random_Code/harrytools")
import harrytools
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
import util

char2ind, ind2char = util.load_pkl('deep_learn/vocab.pkl')
one_hot = OneHotEncoder(len(char2ind), sparse=False)

PS = False
LR = 0.0001
N_HIDDEN = 200
N_LAYERS = 3
EMBED_DIM = 64
SHOW = 1

def decode(inds, ind2char):
    return ''.join([ind2char[i] for i in inds])


def load_model(fname):
    model = GenerateModel(len(ind2char), N_HIDDEN, N_LAYERS)
    try:
        checkpoint = torch.load(fname)
        model = GenerateModel(**checkpoint['params'])
        model.load_state_dict(checkpoint['state_dict'])
        model.epoch = checkpoint['epoch']
        return model
    except:
        print('Starting new model')
        model = GenerateModel(len(ind2char), N_HIDDEN, N_LAYERS, EMBED_DIM)
        return model


class Trainer():
    def __init__(self, model, lr=0.001):
        self.data = DataLoader(lyrics)
        self.lr = lr
        self.model = model
        self.loss_fn = nn.NLLLoss()
        self.optimizer = Adam(model.parameters(), lr=self.lr)

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            losses = []
            for lyric in self.data:
                X = lyric[:-1]
                Y = lyric[1:]
                model_in = torch.tensor(X, dtype=torch.long)
                targets = torch.tensor(Y, dtype=torch.long)

                self.model.zero_grad()
                hidden = self.model.init_hidden()
                prediction, _ = self.model(model_in, hidden)
                loss = self.loss_fn(prediction, targets)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            sample(self.model, n=300)
            self.model.epoch += 1
            print('Loss after epoch {}: {}'.format(self.model.epoch, np.mean(losses)))
            self.model.save()

class Trainer2(Trainer):
    def __init__(self, model, lr=0.001, seq_len=25):
        Trainer.__init__(self, model, lr)
        self.seq_len = seq_len


    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            losses = []
            for lyric in tqdm(self.data):
                X = lyric[:-1]
                Y = lyric[1:]
                hidden = self.model.init_hidden()
                start_ind = 0
                while True:
                    end_ind = min(start_ind + self.seq_len, len(X))
                    model_in = torch.tensor(X[start_ind: end_ind], dtype=torch.long)
                    targets = torch.tensor(Y[start_ind: end_ind], dtype=torch.long)

                    self.model.zero_grad()
                    prediction, hidden = self.model(model_in, hidden)
                    loss = self.loss_fn(prediction, targets)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

                    hidden = (hidden[0].detach(), hidden[1].detach())

                    if end_ind == len(X): break
                    start_ind = end_ind

            sample(self.model, n=300)
            self.model.epoch += 1
            print('Loss after epoch {}: {}'.format(self.model.epoch, np.mean(losses)))
            self.model.save()




class Trainer3(Trainer):
    def __init__(self, model, lr=0.001, seq_len=25, bs=50):
        Trainer.__init__(self, model, lr)
        self.seq_len = seq_len
        self.data = DataLoader(lyrics, bs, collate_fn=lambda x: np.array(x), drop_last=True)
        self.bs = bs    


    def get_array(self, batch):
        lens = np.array([len(sentence) for sentence in batch])
        sort = np.argsort(lens)

        lens = lens[sort]
        batch = batch[sort]

        mask = np.arange(lens[-1]) < lens[:,None]
        sentences = np.zeros((self.bs, lens[-1]))
        sentences[mask]=np.concatenate(batch)
        return sentences, lens

    
    def repackage_hidden(self, h):
        if type(h) == torch.Tensor:
            return torch.tensor(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def remove_first(self, hidden):
        return (hidden[0][:,1:,:], hidden[1][:, 1:, :])

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            losses = []
            for batch in tqdm(self.data):
                working_bs = self.bs
                batch_start, seq_start = 0, 0
                sentences, lens = self.get_array(batch)
                hidden = self.model.init_hidden(working_bs)
                while True:
                    model.zero_grad()
                    for i in range(len(lens)):
                        if 0 < lens[i] <= self.seq_len:
                            working_bs -= 1
                            batch_start += 1
                            if working_bs > 0: hidden = self.remove_first(hidden)
                    if working_bs == 0:
                        break
                    
                    seq_end = seq_start + self.seq_len
                    this_X = torch.tensor(sentences[batch_start:, seq_start:seq_end].T, dtype=torch.long)
                    out, hidden = model(this_X, hidden, working_bs, verbose=0)
                    out = torch.transpose(out, 0,1).contiguous()
                    this_Y = torch.tensor(sentences[batch_start:, seq_start+1:seq_end+1], dtype=torch.long)
                    loss = self.loss_fn(out.view((-1, model.n_chars)), this_Y.view(-1))
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    hidden = self.repackage_hidden(hidden)
                    lens -= self.seq_len

            sample(self.model, n=300)
            self.model.epoch += 1
            print('Loss after epoch {}: {}'.format(self.model.epoch, np.mean(losses)))
            self.model.save()


def sample(model, start='<', n=100):
    model.eval()
    encode = [char2ind[s] for s in start]
    model_in = torch.tensor(encode, dtype=torch.long).view(-1)
    h = model.init_hidden()

    if len(start)>1:
        out, h = model(model_in[:-1], h)

    model_in = model_in[-1].view(-1)
    string = start
    for i in range(n):
        out, h = model(model_in, h)
        prob = np.exp(out.data.numpy().flatten())
        ch = np.random.choice(len(ind2char), p=prob)
        string += ind2char[ch]
        model_in = torch.tensor(ch, dtype=torch.long)
    print(string)

    model.train()


if __name__ == "__main__":


    fname = 'hidden_dim200_n_layers3_embed_dim64_n_chars78model.pth.tar' 
    model = load_model(fname)
    sample(model, n=500)
    #trainer = Trainer2(model, seq_len=40)
    trainer = Trainer3(model, seq_len=40, bs=150, lr=0.0001)
    trainer.train(200)


