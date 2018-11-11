import pickle
import numpy as np
import sklearn 
import os 
import pandas as pd
import ftfy

def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class DataFeatures:
    def __init__(self, dataset):
        self.raw = load_weebit()
        #self.raw = load_onestop()
        # TODO logic for switching datset
        self.fname = dataset + 'features.pkl'
        self.save()

    def save(self):
        save_pickle(self.fname, self)


    def get_tfidf(self):
        '''
        Returns
            tfidf array
        '''
        pass

    def get_wc(self):
        '''
        Returns
            word count array
        '''
        
        pass

    def get_nlp(self):
        '''

        Arguments

        Returns
        Dictionary of feature name to value.
        '''
        pass

# Extract features (use spacy)
def feature_extract():
    pass

def load_weebit():
    data = []
    weebit_dir = '../data/weebit/WeeBit-TextOnly/'
    difficulty_levels = ['WRLevel2', 'WRLevel3', 'WRLevel4']

    for difficulty in  difficulty_levels:
        path = weebit_dir + difficulty
        texts = os.listdir(path)
        for t in texts:
            with open(os.path.join(path, t),  'r', encoding = "ISO-8859-1") as myfile:
                this_text = myfile.read().replace('\n', ' ')
                this_text = ftfy.fix_text(this_text)
                data.append((this_text, int(difficulty[-1]), t))

    df = pd.DataFrame(data)
    df.columns = ['text', 'level', 'fname']
    return df


def load_onestop():
    pass

if __name__ == "__main__":
    x = DataFeatures('weebit')
    print(x)

