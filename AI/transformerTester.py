from FXTMdataset import MarketDataGenerator
from Attention3 import DeepPredictor
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
from NASDAQ100dataset import NasdaqGenerator
from Luong import Luong
from Transformer import Transformer

sys.setrecursionlimit(1500)


def show_rand_sample(validY, validPredict, num_samples):
    for i in range(num_samples):
        rand = random.randrange(validY.shape[0])
        rand2 = random.randrange(validY.shape[2])
        validY_slice = validY[rand, :, rand2, 0]
        validPredict_slice = validPredict[rand, :, rand2, 0]
        validY_delta = []
        validPredict_delta = []
        for j in range(len(validY_slice) - 1):
            validY_delta.append(validY_slice[j + 1] - validY_slice[j])
            validPredict_delta.append(validPredict_slice[j + 1] - validPredict_slice[j])
        plt.subplot(211)
        plt.plot(validY_slice)
        plt.plot(validPredict_slice)
        plt.subplot(212)
        plt.plot(validY_delta)
        plt.plot(validPredict_delta)
        plt.show()


def trend_accuracy(Y, Predict):
    count = 0
    total = 0

    def get_delta(Y):
        Y_shiftright = np.concatenate(([Y[0]], Y), axis=0)
        Y_shiftright = np.delete(Y_shiftright, len(Y) - 1, axis=0)
        print(Y.shape)
        print(Y_shiftright.shape)
        return np.subtract(Y_shiftright, Y)

    Y_delta = get_delta(Y)
    Predict_delta = get_delta(Predict)

    Y_delta = np.reshape(Y_delta, [-1])
    Predict_delta = np.reshape(Predict_delta, [-1])

    for i in range(len(Y_delta)):
        if Y_delta[i] * Predict_delta[i] > 0:  # if they are the same
            count += 1
        total += 1

    return count / total


# tensorboard --host 127.0.0.1 --logdir=D:/Projects/tensor2/summary/Transformer
train_ratio = 0.8
seq_length = 32
output_count = 8
batch_size = 8
N = 6
filter_num = 12
kernel_size = 64
ffn_size = kernel_size * 4
epoch = 3000
learning_rate = 0.00001

mfile = './models/en2de.model.h5'
mfile_arch = './models/Transformer/en2de.model_arch.json'


with open(mfile_arch, 'r') as f:
    model = tf.keras.models.model_from_json(f.read())