import pickle
import numpy as np
import sklearn
import pandas as pd
import ftfy
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

import os, sys
sys.path.append('preprocess')
sys.path.append('algorithms')
import util

class DataFeatures:
    def __init__(self, dataset):
        self.raw = load_weebit()
        # TODO logic for switching datset

        self.count_matrix, self.tfidf_matrix = None, None

        # self.get_tfidf()   # These two are fast and should just be called everytime
        # self.get_wc()      # with different options.
        self.nl_matrix = self.get_nlfeatures()

        self.fname = 'preprocess/' + dataset + '_features.pkl'
        self.get_indices()
        self.labels = self.raw.level.values
        self.save()

    def get_f_dict(self):
        return dict(zip(util.features, [self.count_matrix, self.tfidf_matrix, self.nl_matrix]))

    def get_joint_matrix(self, features, wc_params, tfidf_params):
        if 'word count' in features: self.get_wc(wc_params)
        if 'tfidf' in features: self.get_tfidf(tfidf_params)
        f_dict = self.get_f_dict()
        X = [f_dict[f] for f in features]
        X = np.concatenate(tuple(X), axis=1)
        return X

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
        self.tfidf_matrix = tfidf.transform(df.text).todense()
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
        self.count_matrix = count.transform(df.text).todense()
        return self.count_matrix

    def get_nlfeatures(self, save_nl=False):
        '''
        Arguments
        Returns
        Dictionary of feature name to value.
        '''
        # Function to check if the token is a noise or not
        def isNoise(token, noisy_pos_tags=['PROP'], min_token_length=2):
            return token.pos_ in noisy_pos_tags or token.is_stop or len(token.string) <= min_token_length

        def cleanup(token, lower=True):
            if lower:
                return token.lower().strip()
            return token.strip()

        print('Getting NLP Features...')
        nlp = spacy.load('en')
        # documents = self.raw['text'].apply(nlp)

        num_docs = 0
        feature_matrix = []
        if save_nl: doc_objs = []

        for text in self.raw.text:
            doc = nlp(text)

            num_docs += 1
            POS_TAGS = [
                "", "ADJ", "ADP", "ADV", "AUX", "CONJ",
                "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
                "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
                "X", "EOL", "SPACE"
            ]
            noun_chunks = list(doc.noun_chunks)
            sentences = list(doc.sents)
            avg_sent_length = sum([len(sent) for sent in sentences]) / len(sentences)
            all_tag_counts = defaultdict(int)
            for w in doc:
                all_tag_counts[w.pos_] += 1
            # cleaned_list = [cleanup(word.string) for word in doc if not isNoise(word)]
            # Counter(cleaned_list).most_common(5)
            feats = []
            for tag in POS_TAGS:
                feats.append(all_tag_counts[tag])
            feats.append(len(noun_chunks))          # num_noun_chunks
            feats.append(len(sentences))            # num_sentences
            feats.append(avg_sent_length)
            feature_matrix.append(feats)
            if save_nl: doc_objs.append(doc)

        if save_nl: util.save_pkl('preprocess/doc_objs.pkl', doc_objs)


        return np.array(feature_matrix)


def prep_weebit():
    data = []
    weebit_dir = 'data/weebit/WeeBit-TextOnly/'
    difficulty_levels = ['WRLevel2', 'WRLevel3', 'WRLevel4']

    for difficulty in difficulty_levels:
        path = weebit_dir + difficulty
        texts = os.listdir(path)
        for t in texts:
            with open(os.path.join(path, t),  'r', encoding="ISO-8859-1") as myfile:
                this_text = myfile.read().replace('\n', ' ')
                this_text = this_text.replace(
                    'All trademarks and logos are property of Weekly Reader Corporation.', '')
                this_text = ftfy.fix_text(this_text)
                data.append((this_text, int(difficulty[-1]), t))

    df = pd.DataFrame(data)
    df.columns = ['text', 'level', 'fname']
    df['split'] = np.random.choice(3, len(df), p=[0.8, 0.1,0.1])
    df.to_csv('data/weebit/weebit.csv', index=False)
    return df


def load_weebit():
    return pd.read_csv('data/weebit/weebit.csv')


def load_one_stop():
    # TODO
    return pd.read_csv('')

def fix_non_ascii():
    wb = load_weebit()
    c_1, c_2 = 0, 0
    for text in wb.text:
        for w in text.split():
            if not len(w) == len(w.encode()):
                print(w)
    # TODO Tyler, can you figure this thing out, and fix it? 


def prep_onestop():
    pass


if __name__ == "__main__":
    fix_non_ascii()
