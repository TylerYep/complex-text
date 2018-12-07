import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk

import os, sys
sys.path.append('preprocess')
sys.path.append('algorithms')
import read_data
import util

def load_one_stop():
    return pd.read_csv('data/OneStop/tyler-processed.csv')

def load_aligned_data():
    ''' Reads aligned data. '''
    data = []
    path = 'data/OneStop/Sentence-Aligned'

    texts = os.listdir(path)
    for t in texts:
        with open(os.path.join(path, t),  'r', encoding="ISO-8859-1") as myfile:
            this_text = 'A'
            while this_text:
                this_text = myfile.readline().replace('\n', ' ')
                data.append((this_text, t[:3], t))
                this_text = myfile.readline().replace('\n', ' ')
                data.append((this_text, t[4:7], t))
                this_text = myfile.readline() # *****

    df = pd.DataFrame(data)
    df.columns = ['text', 'level', 'fname']
    # df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1,0.1])
    df.to_csv('data/OneStop/tyler-processed.csv', index=False)
    return df


def load_parsed_data():
    ''' Reads parsed data. '''
    data = []
    path = 'data/OneStop/Processed-AllLevels-AllFiles/Parsed'

    texts = os.listdir(path)
    for t in tqdm(texts): # TODO
        with open(os.path.join(path, t),  'r', encoding="ISO-8859-1") as myfile:
            this_text = myfile.read().replace('\n', ' ')
            level = t[t.index('-')+1:t.index('-')+4]
            while level not in ['ele', 'int', 'adv']:
                chop_string = t[t.index('-')+1:]
                level = chop_string[t.index('-')+1:t.index('-')+4]
            data.append((this_text, level, t))

    df = pd.DataFrame(data)
    df.columns = ['text', 'level', 'fname']
    df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1,0.1])
    df.to_csv('data/OneStop/ty-processed.csv', index=False)
    return df

def ntlk_stuff():
    df = load_one_stop()
    grammar = ('NP: {<DT>?<JJ>*<NN>}')
    chunkParser = nltk.RegexpParser(grammar)
    sentence = df['text'][3]
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # print(tagged)

    tree = chunkParser.parse(tagged)
    for subtree in tree.subtrees():
        print(subtree)

    sentence2 = df['text'][5]




ntlk_stuff()
