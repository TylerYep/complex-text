import pickle
import numpy as np
import sklearn 
import pandas as pd
import ftfy
import os, sys
sys.path.append('../')
import util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataFeatures:
    def __init__(self, dataset):
        self.raw = load_weebit()
        # TODO logic for switching datset
        self.get_tfidf()
        self.get_wc()
        self.get_nlfeatures()

        self.fname = dataset + 'features.pkl'
        self.save()

    def save(self):
        util.save_pkl(self.fname, self)


    def get_tfidf(self, tfidf_params={}):
        '''
        Returns
            tfidf array
        '''
        # Operate on training only for fitting
        df = self.raw
        train = df[df.split == 0]
        tfidf = TfidfVectorizer(**tfidf_params)
        tfidf.fit(train.text)
        self.tfidf = tfidf
        self.tfidf_matrix = tfidf.transform(df.text)
        return self.tfidf_matrix

    def get_wc(self, count_params={}):
        '''
        Returns
            word count array
        '''
        df = self.raw
        train = df[df.split == 0]
        count = CountVectorizer(**count_params)
        count.fit(train.text)
        self.wc = count
        self.count_matrix = count.transform(df.text)
        return self.count_matrix

    def get_nlfeatures(self):
        '''

        Arguments

        Returns
        Dictionary of feature name to value.
        '''
        pass

# Extract features (use spacy)
def feature_extract():
    pass

def prep_weebit():
    data = []
    weebit_dir = '../data/weebit/WeeBit-TextOnly/'
    difficulty_levels = ['WRLevel2', 'WRLevel3', 'WRLevel4']

    for difficulty in  difficulty_levels:
        path = weebit_dir + difficulty
        texts = os.listdir(path)
        for t in texts:
            with open(os.path.join(path, t),  'r', encoding = "ISO-8859-1") as myfile:
                this_text = myfile.read().replace('\n', ' ')
                this_text = this_text.replace('All trademarks and logos are property of Weekly Reader Corporation.', '')
                this_text = ftfy.fix_text(this_text)
                data.append((this_text, int(difficulty[-1]), t))

    df = pd.DataFrame(data)
    df.columns = ['text', 'level', 'fname']
    df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1,0.1])
    df.to_csv('../data/weebit/weebit.csv', index=False)

    return df

def load_weebit():
    return pd.read_csv('../data/weebit/weebit.csv')

def load_one_stop():
    # TODO 
    return pd.read_csv('')

def prep_onestop():
    pass

if __name__ == "__main__":

    #x = prep_weebit()
    #print(x)
    x = DataFeatures('weebit')


