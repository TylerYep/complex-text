import pandas as pd
from model import GenerateModel
from train import *
import numpy as np 
import matplotlib.pyplot as plt
from data import DataG
import sys
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
import util

char2ind, ind2char = util.load_pkl('generate/vocab_g.pkl')

if __name__ == "__main__":
    levels = [2, 3, 4]
    fnames = ['lvl{}_hidden_dim100_n_layers3_embed_dim64_n_chars43model.pth.tar'.format(i) for i in levels]
    models = [load_model(f, False) for f in fnames]
    for m in models:
        sample(m, temp=0.3, n=300)
        print()

