# coding: utf-8
import argparse
import os

import chainer
import pandas as pd
from chainer import training, iterators, optimizers, serializers
from chainer.training import extensions

from model.cnn_rand import CNNRand
from preprocessing.data_generator import DataGenerator


def _read_file(filepath: str):
    with open(filepath, 'rb') as f:
        return pd.DataFrame({'review': [str(l) for l in f.readlines()]})


def create_data(test_size: float = 0.1, max_sentence_length: int = 100) -> DataGenerator:
    positive_dataset = _read_file(os.path.join('data', 'rt-polaritydata', 'rt-polarity.pos'))
    positive_dataset['label'] = 1
    negative_dataset = _read_file(os.path.join('data', 'rt-polaritydata', 'rt-polarity.neg'))
    negative_dataset['label'] = 0
    return DataGenerator(positive_dataset, negative_dataset, test_size=test_size, max_sentence_length=max_sentence_length)


def train_model(max_sentence_length,
                n_factor,
                batch_size,
                decay,
                gpu,
                n_epoch,
                n_out_channel):

    dataset = create_data(test_size=0.1, max_sentence_length=max_sentence_length)
    filter_windows = [3, 4, 5]
    max_sentence_length = dataset.get_max_sentence_length()
    n_word = dataset.get_n_word()

    model = CNNRand(filter_windows=filter_windows, max_sentence_length=max_sentence_length, n_word=n_word, n_factor=n_factor, n_out_channel=n_out_channel)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    train_iter = iterators.SerialIterator(dataset.get_train_dataset(), batch_size, shuffle=True)
    test_iter = iterators.SerialIterator(dataset.get_test_dataset(), batch_size, repeat=False, shuffle=False)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu), name='test')
    trainer.extend(
        extensions.PrintReport(entries=[
            'epoch',
            'main/loss', 'test/main/loss',
            'main/accuracy', 'test/main/accuracy',
            'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    model.to_cpu()
    serializers.save_npz('./result/cnn_rand_model.npz', model)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sentence_length', type=int, default=60)
    parser.add_argument('--n_factor', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_out_channel', type=int, default=1)
    args = parser.parse_args()
    print(args)
    train_model(**vars(args))
