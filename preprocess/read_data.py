import pickle
import numpy as np
import sklearn
import pandas as pd
import ftfy
import os, sys
import spacy
sys.path.append('../')
import util
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

features = ['word count', 'tfidf', 'nl']

class DataFeatures:
    def __init__(self, dataset):
        self.raw = load_weebit()
        # TODO logic for switching datset

        self.count_matrix, self.tfidf_matrix = None, None

        #self.get_tfidf()   # These two are fast and should just be called everytime
        #self.get_wc()      # with different options.

        #self.nl_matrix = self.get_nlfeatures()

        self.fname = dataset + '_features.pkl'
        self.get_indices()
        self.labels = self.raw.level
        self.save()

    def get_f_dict(self):
        return dict(zip(features, [self.count_matrix, self.tfidf_matrix, self.nl_matrix]))

    def get_indices(self):
        self.train_indices = self.raw[self.raw.split == 0].index
        self.val_indices = self.raw[self.raw.split == 1].index
        self.test_indices = self.raw[self.raw.split == 2].index

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
        # Function to check if the token is a noise or not
        def isNoise(token, noisy_pos_tags = ['PROP'], min_token_length = 2):
            return token.pos_ in noisy_pos_tags or token.is_stop or len(token.string) <= min_token_length

        def cleanup(token, lower = True):
            if lower:
               token = token.lower()
            return token.strip()


        nlp = spacy.load('en')
        documents = self.raw['text'].apply(nlp)
        # df = pd.DataFrame(documents)
        # df.columns = ['nl-features']
        # df.to_csv('../data/tyler-bit.csv', index=False)
        # df = pd.read_csv('../data/tyler-bit.csv')
        # print(self.raw)
        feature_matrix = []
        for doc in documents: # df['nl-features']:
            feats = dict()
            noun_chunks = list(doc.noun_chunks)
            sentences = list(doc.sents)
            all_tag_counts = defaultdict(int) # {w.pos_: w.pos for w in doc}
            for w in doc:
                all_tag_counts[w.pos_] += 1
            cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word)]
            Counter(cleaned_list).most_common(5)

            feats['num_nouns'] = len(noun_chunks)
            feats['num_sents'] = len(sentences)
            for tag, count in all_tag_counts.items():
                feats[tag] = count
            feature_matrix.append(feats)

        return feature_matrix

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
#   train, test = train_test_split(df, test_size=0.33, random_state=42)
    return df

def load_weebit():
    return pd.read_csv('../data/weebit/weebit.csv')

def load_one_stop():
    # TODO
    return pd.read_csv('')

def prep_onestop():
    pass

if __name__ == "__main__":
    pass
    # prep_weebit()
    #x = DataFeatures('weebit')
