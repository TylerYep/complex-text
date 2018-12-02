"""
File: train.py
Author: Harry Sha
Email: harryshahai@gm*
Description: Trains LSTM
"""

import pandas as pd
import numpy as np 
from data import Data
import sys
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
import util
from model import Model

word2ind, ind2word = util.load_pkl('deep_learn/vocab.pkl')

# ==== PARAMETERS ============
N_EMBS = len(word2ind)
EMBEDDING_DIM = 64
NUM_LAYERS = 1
HIDDEN_DIM = 256
BATCH_SIZE = 32
OUTPUT_DIM = 3
LR = 0.001
# ============================

def evaluate(model, loader, analyze=False):
    """
    Evaluates the model on the data in loader. 
    If analyze is True, the predictions and
    actual labels are returned as well.

    Arguments
        model (Tagger): Model to be evaluated
        loader (DataLoader):    Data for the model to be 
                                evaluated on

    Returns
        if analyze: (float):    Accuracy of the model on the data
        else: (float, array, array): Accuracy, predictions, labels
    """
    model.eval()
    preds, labels = [],[]
    for sentence, label in tqdm(loader):
        sentence, lens, label = prepare_texts(sentence, label, model.batch_size)
        model.hidden = model.init_hidden()
        activation = model(sentence, lens).data.numpy()
        preds.append(activation.argmax(1))
        label = label.numpy()
        labels.append(label)
    preds, labels = np.array(preds).flatten(), np.array(labels).flatten()
    correct = preds == labels
    
    accuracy = sum(correct)/len(correct)
    to_return = accuracy
    if analyze:
        to_return = [accuracy, preds, labels]

    return to_return

def prepare_texts(sentences, labels=None, bs=1, return_sorts=False):
    """
    Prepares sentences for input into the LSTM model. Note that when 
    batch size (bs) > 1, the returned sentences and labels will be sorted
    by length (this is a requirement for pack_padded_sequences). Sentences
    will be padded with zeros to the right.

    Arguments
        sentences (list of str):    List of strings that are indices of the 
                                    words in the vocab separated by spaces
        labels (list of int):   List of the labels. Ignored if none (e.g, 
                                when there is not a label)
        bs (int):   Batch size to proces (typically just the length of 
                    sentences)

        return_sorts (bool):    If true, return the result of argsorting by 
                                length. (argsort this again to get back the 
                                original list). This is useful to retrieve 
                                the original order of items.
    Returns
        (tensor):   Input to the LSTM model
        (list): List of lengths of sentences 
        (list): List of labels
        (list): Output of argsort by length 
    """
    sentence_lengths = np.array([len(s) if len(s) > 1 else 1 for s in sentences])
    max_len = max(sentence_lengths)
    encoding = np.zeros((bs, max_len))

    sort = np.argsort(-sentence_lengths)
    sentence_lengths = sentence_lengths[sort]
    sentences = np.array(sentences)[sort]
    if labels is not None:
        labels = labels[sort]

    for i, s in enumerate(sentences):
        encoding[i,:len(s)] = s

    to_return = [torch.tensor(encoding, dtype=torch.long), sentence_lengths, labels]
    if return_sorts:
        to_return.append(sort)

    return to_return

def train(model, train_loader, dev_loader, fname=None, num_epochs=1000):
    """
    Trains model with the data in train_loader, saving checkpoints after
    the epoch in which the best dev-accuracy is obtained.

    Arguments
        model (Model object):  The model to be trained (see models.py)
        train_loader (DataLoader):  The data loader containing training data
        dev_loader (DataLoader):    The data loader containing training data
        fname (str) or None:    The filename from which to load a checkpoint.
                                If it is None, starts a new model
        num_epochs (int):   Number of epochs to train for

    Returns
        (Model object) the trained model 
    """
    model.train()
    loss_function = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    epoch, best_dev = load_model(model, optimizer, fname, model.batch_size) if fname else 0, 0
    for epoch in range(epoch, epoch + num_epochs):
        losses = []
        for sentence, label in tqdm(train_loader):
            #print(sentence)
            sentence, lens, label= prepare_texts(sentence, label, model.batch_size)
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model(sentence, lens)
            loss = loss_function(tag_scores, label)
            losses.append(loss.data.item())
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_loader)
        dev_acc = evaluate(model, dev_loader)
        model.save_checkpoint(optimizer.state_dict(), dev_acc, epoch)

        print("Loss after epoch {}: {}".format(epoch, np.mean(losses)))
        print("Dev Accuracy: {}, Train Accuracy: {}".format(dev_acc, train_acc))
        print("Best Dev Accuracy: ", model.best_dev)

    return model

def _experiments(num_layers=[1, 2, 3], batch_size=[128, 64],
                embedding_dim=[32, 64], hidden_dim=[64, 128, 256]):
    """
    Runs some experiments (used for model selection)
    and prints some results.

    Arguments
        num_layers (list): list of num layers to try
        batch_size (list): list of batch sizes to try
        embedding_dim (list): list of embedding dims to try
        hidden_dim (list): list of hidden dims to try

    Returns: None
    """

    results = [['num layers','batch size', 'embedding dim', 'hidden dim', 'best dev']]
    for b, n, e, h in itertools.product(batch_size, num_layers, embedding_dim, hidden_dim):
        print('num layers', n)
        print('batch size', b)
        print('embedding dim', e)
        print('hidden dim', h)
        print()

        model = Model(n, e, h, b, N_EMBS, OUTPUT_DIM)
        train_loader = DataLoader(JobPostingData('train'), batch_size=b, shuffle=True, num_workers=0, drop_last=True)
        dev_loader = DataLoader(JobPostingData('dev'), batch_size=b, shuffle=True, num_workers=0, drop_last=True)
        train(model, train_loader, dev_loader, num_epochs=10)
        results.append([n, b, e, h, model.best_dev])
    print(results)

def collate(batch):
    X = [b[0] for b in batch]
    Y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return X, Y

if __name__ == "__main__":
    #_experiments()
    model=Model(NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, N_EMBS, OUTPUT_DIM)
    train_loader = DataLoader(Data(0), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    dev_loader = DataLoader(Data(1), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate)
    train(model, train_loader, dev_loader)