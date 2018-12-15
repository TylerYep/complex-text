import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
sys.path.append('preprocess')
import read_data
import util

def preprocess(text):
    text.replace('<', '')
    text.replace('>', '')
    text += '>'
    text = '<' + text
    return text

def encode(seq, word2ind):
    return [word2ind[w] if w in word2ind else 0 for w in seq]

def decode(seq, ind2word):
    return [ind2word[w] for w in seq]

def make_vocab_and_dataset():
    wb = read_data.load_weebit()
    doc_objs = util.load_pkl('preprocess/doc_objs.pkl')

    dataset = []
    vocab = Counter()

    for i, doc in enumerate(doc_objs):
        doc = doc[:-1]
        x = ['<'] + [w.pos_ if w.pos_ not in  ['PUNCT'] else w.text for w in doc] + ['>']
        dataset.append(x)
        vocab += Counter(x)

    words = ['UNK'] + [w for w, c in vocab.items() if c > 5]

    word2ind = {w : i for i, w in enumerate(words)}
    ind2word = {i : w for i, w in enumerate(words)}

    dataset = [(encode(x, word2ind), wb.level[i], wb.split[i]) for i, x in enumerate(dataset)]

    util.save_pkl('generate/vocab_g.pkl', (word2ind, ind2word))
    util.save_pkl('generate/weebit_generate.pkl', dataset)


class DataG(Dataset):
    def __init__(self, level, size=None, pos=True):
        # level \in {2, 3, 4}
        if pos: self.texts = util.load_pkl('generate/weebit_generate.pkl')
        self.texts = [x for x in self.texts if x[1] == level]
        self.texts = [x for x in self.texts if len(x[0]) > 2]
        if size is not None:
            self.texts = self.texts[:size]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, ix):
        return np.array(self.texts[ix][0])

if __name__ == "__main__":
    # make_vocab_and_dataset()
    d = DataG(2, 1)
    print(d.texts)
    word2ind, ind2word = util.load_pkl('generate/vocab_g.pkl')
    print(word2ind)
    for x in d:
        print(x)
        print(decode(x, ind2word))

