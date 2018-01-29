# coding: utf-8
from typing import Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split


class DataGenerator(object):
    def __init__(self, positive_dataset: pd.DataFrame, negative_dataset: pd.DataFrame, test_size: float, random_state: int = 123,
                 max_sentence_length: int = None):
        self.dataset = pd.concat([positive_dataset, negative_dataset], axis=0)
        self.dataset['review'] = self.dataset['review'].apply(lambda x: x.strip().split())
        self.dictionary = Dictionary(self.dataset['review'].values)
        self.dataset['review'] = self.dataset['review'].apply(self.dictionary.doc2idx)
        self.max_sentence_length = max_sentence_length
        if self.max_sentence_length is not None:
            self.dataset['review'] = self.dataset['review'].apply(lambda x: x[:self.max_sentence_length])
        else:
            self.max_sentence_length = max(self.dataset['review'].apply(len))

        # padding
        eos_id = len(self.dictionary.keys())
        self.dataset['review'] = self.dataset['review'].apply(
            lambda x: np.pad(x, (0, self.max_sentence_length - len(x)), 'constant', constant_values=(0, eos_id)))

        # change type
        self.dataset['review'] = self.dataset['review'].apply(lambda x: x.astype(np.int32))
        self.dataset['label'] = self.dataset['label'].astype(np.int32)

        # split
        self.train, self.test = train_test_split(self.dataset, test_size=test_size, random_state=random_state)

    def get_train_dataset(self):
        return list(zip(self.train['review'].values, self.train['label'].values))

    def get_test_dataset(self):
        return list(zip(self.test['review'].values, self.test['label'].values))

    def get_max_sentence_length(self):
        return self.max_sentence_length

    def get_n_word(self):
        return len(self.dictionary.keys()) + 1
