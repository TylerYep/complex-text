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
import random
import spacy
import regex as re

default_words = {
    # 'ADJ': ['new', 'good', 'high', 'special', 'big', 'local'],
    # 'ADP': ['on', 'of', 'at', 'with', 'by', 'into', 'under'],
    # 'ADV': ['really', 'already', 'still', 'early', 'now'],
    'CONJ': ['and', 'or', 'but', 'if', 'while', 'although'],
    'CCONJ': ['and', 'or', 'but'],
    'DET': ['the', 'a', 'some', 'most', 'every', 'no', 'which'],
    # 'NOUN': [''],
    'NUM': ['1', '10'],
    # 'PRT': ['at', 'on', 'out', 'over', 'per', 'that', 'up', 'with'],
    'PRON': ['he', 'their', 'her', 'its', 'my', 'I', 'us'],
    'PROPN': ['New'],
    'PART': ['\'s']
    # 'VERB':['is', 'say', 'told', 'given', 'playing', 'would']
}

nlp = spacy.load('en_core_web_sm')

lvl2 = []
with open('lvl2.txt') as f:
    for line in f:
        if line.find("Loss after epoch") >= 0:
            continue
        else:
            lvl2.append(line)

lvl3 = []
with open('lvl3.txt') as f:
    for line in f:
        if line.find("Loss after epoch") >= 0:
            continue
        else:
            lvl3.append(line)

lvl4 = []
with open('lvl4.txt') as f:
    for line in f:
        if line.find("Loss after epoch") >= 0:
            continue
        else:
            lvl4.append(line)

doc_objs = util.load_pkl('preprocess/doc_objs.pkl')
# data = read_data.load_weebit()

def add_to_word_dict(pos_dict, doc):
    for w in doc:
        if w.pos_ in pos_dict:
            pos_dict[w.pos_].append(w)
        else:
            pos_dict[w.pos_] = [w]


def create_data(lvl, lvl_text):
    data = []
    for doc in doc_objs:
        lvl_i = random.randint(0, len(lvl_text)-1)
        generated_text = re.sub('[^\P{P}\-.]+', '', lvl_text[lvl_i]).replace('>', '')
        pos_dict = default_words.copy()
        add_to_word_dict(pos_dict, doc)
        pos_dict['SPACE'] = ['']
        pos_dict['PUNCT'] = ['.']

        new_sentence = ''
        for w in generated_text.split():
            if w in pos_dict:
                new_sentence += str(random.choice(pos_dict[w])) + ' '
            else:
                new_sentence += w + ' '
        # print(new_sentence)
        data.append((new_sentence, lvl))
    return np.array(data)

lvl_2_gen_data = create_data(2, lvl2[-100:])
lvl_3_gen_data = create_data(3, lvl3[-100:])
lvl_4_gen_data = create_data(4, lvl4[-100:])
print(lvl_2_gen_data, lvl_3_gen_data, lvl_4_gen_data)

df = pd.DataFrame(np.concatenate((lvl_2_gen_data, lvl_3_gen_data, lvl_4_gen_data), axis=0))
df.columns = ['text', 'level']
df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1, 0.1])
df.to_csv('data/weebit/weebit_generate.csv', index=False)

