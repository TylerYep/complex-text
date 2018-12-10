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

char2ind, ind2char = util.load_pkl('generate/vocab_g.pkl')
one_hot = OneHotEncoder(len(char2ind), sparse=False)

LEVEL = 4
N_HIDDEN = 100
N_LAYERS = 3
EMBED_DIM = 64
SHOW = 1
temp = 0.5
texts = DataG(LEVEL)

np.random.seed()

def decode(inds, ind2char):
    return ' '.join([ind2char[i] for i in inds])

def load_model(fname):
    model = GenerateModel(len(ind2char), N_HIDDEN, N_LAYERS, EMBED_DIM, LEVEL)
    if int(fname[3]) != LEVEL:
        raise Exception('check the level')
    checkpoint = torch.load(fname)
    model = GenerateModel(**checkpoint['params'])
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch = checkpoint['epoch']
    model.fname = fname
    return model
    #except:
        #print('Starting new model')
        #model = GenerateModel(len(ind2char), N_HIDDEN, N_LAYERS, EMBED_DIM)
        #return model


class Trainer():
    def __init__(self, model, lr=0.001):
        self.data = DataLoader(texts, 1, collate_fn=lambda x: np.array(x), drop_last=True, shuffle=True)
        self.lr = lr
        self.model = model
        self.loss_fn = nn.NLLLoss()
        self.optimizer = Adam(model.parameters(), lr=self.lr)

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            losses = []
            for text in tqdm(self.data):
                text = text[0]
                X = text[:-1]
                Y = text[1:]
                model_in = torch.tensor(X, dtype=torch.long)
                targets = torch.tensor(Y, dtype=torch.long)

                self.model.zero_grad()
                hidden = self.model.init_hidden()

                prediction, _ = self.model(model_in, hidden)
                loss = self.loss_fn(prediction.squeeze(), targets)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

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
    output = []
    for i in range(n):
        out, h = model(model_in, h)
        prob = np.exp(out.data.numpy().flatten())
        expon = [p**(1/temp) for p in prob]
        prob = [p/sum(expon) for p in expon]
        ch = np.random.choice(len(ind2char), p=prob)
        output.append(ind2char[ch])
        model_in = torch.tensor(ch, dtype=torch.long)
    print(' '.join(output))

    model.train()


if __name__ == "__main__":
    fname = 'lvl4_hidden_dim100_n_layers3_embed_dim64_n_chars43model.pth.tar' 
    model = load_model(fname)
    #sample(model)
    #model = GenerateModel(len(ind2char), N_HIDDEN, N_LAYERS, EMBED_DIM, LEVEL)
    # DID YOU CHANGE THE LEVEL?

    #trainer = Trainer2(model, seq_len=40)
    #trainer = Trainer3(model, seq_len=200, bs=10, lr=0.0001)
    trainer = Trainer(model, lr=0.0001)
    trainer.train(300)

