# coding: utf-8
from typing import List

import chainer
import chainer.functions as functions
import chainer.links as links


class CNNRand(chainer.Chain):
    def __init__(self, filter_windows: List[int], max_sentence_length, n_word, n_factor, n_out_channel=100, n_class=2, dropout_ratio=0.5):
        super(CNNRand, self).__init__()
        # hyperparameters
        self.filter_windows = filter_windows
        self.max_sentence_length = max_sentence_length
        self.n_word = n_word
        self.n_factor = n_factor
        self.n_in_channel = 1
        self.n_out_channel = n_out_channel
        self.n_class = n_class
        self.dropout_ratio = dropout_ratio

        # model architecture
        with self.init_scope():
            self.embedId = links.EmbedID(self.n_word, self.n_factor)
            self.convolution_links = [links.Convolution2D(self.n_in_channel, self.n_out_channel, (window, self.n_factor), nobias=False, pad=0)
                                      for window in self.filter_windows]
            self.fully_connected = links.Linear(self.n_out_channel * len(self.filter_windows), self.n_class)

    def __call__(self, x, t=None, train=True):
        # item embedding
        embedding = functions.expand_dims(self.embedId(x), axis=1)
        convolutions = [functions.relu(c(embedding)) for c in self.convolution_links]
        poolings = functions.concat([functions.max_pooling_2d(c, ksize=(c.shape[-2])) for c in convolutions], axis=2)
        y = functions.dropout(self.fully_connected(poolings), ratio=self.dropout_ratio)

        if train:
            loss = functions.softmax_cross_entropy(y, t)
            chainer.reporter.report({'loss': loss, 'accuracy': functions.accuracy(y, t)}, self)
            return loss
        else:
            return functions.softmax(y)
