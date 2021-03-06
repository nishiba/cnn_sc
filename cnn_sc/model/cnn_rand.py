# coding: utf-8
from enum import Enum
from typing import List

import chainer
import chainer.functions as functions
import chainer.links as links


class TaskType(Enum):
    Classification = 'Classification'
    Embedding = 'Embedding'


class ConvolutionList(chainer.ChainList):
    def __init__(self, n_in_channel: int, n_out_channel: int, n_factor: int, filter_windows: List[int]):
        link_list = [links.Convolution2D(n_in_channel, n_out_channel, (window, n_factor), nobias=False, pad=0) for window in filter_windows]
        super(ConvolutionList, self).__init__(*link_list)


class CNNRand(chainer.Chain):
    def __init__(self, filter_windows: List[int], max_sentence_length, n_word, n_factor, n_out_channel=100, n_class=2, dropout_ratio=0.5,
                 mode=TaskType.Classification):
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
        self.loss_function = {TaskType.Classification: functions.softmax_cross_entropy,
                              TaskType.Embedding: functions.mean_squared_error}[mode]
        self.output_function = {TaskType.Classification: functions.softmax,
                                TaskType.Embedding: lambda x: x}[mode]
        self.accuracy_function = {TaskType.Classification: functions.accuracy,
                                  TaskType.Embedding: functions.mean_squared_error}[mode]

        # model architecture
        with self.init_scope():
            self.embedId = links.EmbedID(self.n_word, self.n_factor)
            self.convolution_links = ConvolutionList(n_in_channel=self.n_in_channel, n_out_channel=self.n_out_channel, n_factor=self.n_factor,
                                                     filter_windows=self.filter_windows)
            self.fully_connected = links.Linear(self.n_out_channel * len(self.filter_windows), self.n_class)

    def __call__(self, x, t=None, train=True):
        # item embedding
        embedding = functions.expand_dims(self.embedId(x), axis=1)
        convolutions = [functions.relu(c(embedding)) for c in self.convolution_links]
        poolings = functions.concat([functions.max_pooling_2d(c, ksize=(c.shape[2])) for c in convolutions], axis=2)
        y = functions.dropout(self.fully_connected(poolings), ratio=self.dropout_ratio)

        if train:
            loss = self.loss_function(y, t)
            chainer.reporter.report({'loss': loss, 'accuracy': self.accuracy_function(y, t)}, self)
            return loss
        else:
            return self.output_function(y)
