from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
from scipy import stats
from MT5DataGetter import MT5DataGetter
from datetime import datetime
import MetaTrader5


class MarketDataGenerator(object):

    @staticmethod
    def createTestData_nparray(data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += predLength

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def createTrainData_nparray(data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += 1

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def normalize(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    def normalize_per_symbol(self, data):
        dataset = []
        for i in range(len(data[0])):
            temp = data[:, i]
            temp = self.normalize(temp)
            dataset.append(temp)
        dataset = np.stack(dataset, axis=1);
        return dataset

    @staticmethod
    def standardize(data):
        m = np.mean(data)
        stdev = np.std(data)
        return (data - m) / stdev

    @staticmethod
    def deStandardize(prevData, currentData):
        m = np.mean(prevData)
        stdev = np.std(prevData)
        return currentData * stdev + m

    @staticmethod
    def DeNormalize(prevData, currentData):
        min = np.min(prevData, 0)
        denominator = np.max(prevData, 0) - np.min(prevData, 0)
        return currentData * denominator + min

    @staticmethod
    def getMinTimeStep(data):
        min = data[0].shape[0]
        for i in range(len(data)):
            if (min > data[i].shape[0]):
                min = data[i].shape[0]
        return min

    @staticmethod
    def get_delta(Y):
        Y_shiftright = np.concatenate(([Y[0]], Y), axis=0)
        Y_shiftright = np.delete(Y_shiftright, len(Y) - 1, axis=0)
        return np.subtract(Y_shiftright, Y)

    @staticmethod
    def moving_avg(Y, timestep):
        Y_new = []
        for i in range(len(Y) - timestep):
            Y_chunk = Y[i:i + timestep]
            mean = np.mean(Y_chunk, axis=0)
            Y_new.append(mean)
        return np.stack(Y_new)

    @staticmethod
    def exp_moving_avg(data, timestep):
        ema = []
        k = 2 / (timestep + 1)
        base = np.mean(data[0:timestep], axis=0)
        ema.append(base)
        for symbols in data[timestep:]:
            res = symbols * k + ema[-1] * (1 - k)
            ema.append(res)
        return np.stack(ema)

    @staticmethod
    def remove_outliers(data, threshold=7):
        z = np.abs(stats.zscore(data))
        points = np.where(z > threshold)
        xs = points[0]
        ys = points[1]
        for i in range(len(points[0])):
            x = xs[i]
            y = ys[i]
            data[x, y] = np.mean(data[x - 20:x, y])

    def __init__(self, train_ratio, seq_length, output_count, batch_size):
        symbol_list = ["EURUSD"]
        raw_datasets = MT5DataGetter(symbol_list).getcandledata(datetime(2019, 4, 20), 99999, MetaTrader5.MT5_TIMEFRAME_M1)
        dataset = []

        for raw_dataset in raw_datasets:
            close = raw_dataset['close'].values
            close = np.expand_dims(close,axis=-1)
            dataset.append(close)
        dataset = np.stack(dataset, axis=1) # (timesteps, markets, features)
        print(dataset.shape)

        dataset = self.exp_moving_avg(dataset, 10)
        shape = dataset.shape
        dataset = np.reshape(dataset, [dataset.shape[0], dataset.shape[1] * dataset.shape[2]])

        dataset = self.get_delta(dataset)
        plt.plot(dataset)
        plt.show()
        self.remove_outliers(dataset)
        plt.plot(dataset)
        plt.show()
        dataset = self.normalize_per_symbol(dataset)
        plt.plot(dataset)
        plt.show()
        dataset = np.reshape(dataset, [shape[0], shape[1], shape[2]])

        self.og_dataset = dataset

        train_size = int(len(dataset) * train_ratio)

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        self.trainX, self.trainY = self.createTrainData_nparray(train_dataset, seq_length, output_count)
        self.validX, self.validY = self.createTrainData_nparray(test_dataset, seq_length, output_count)
        self.testX, self.testY = self.createTestData_nparray(test_dataset, seq_length, output_count)

        self.batch_size = batch_size
        self.input_dim = self.trainX.shape[1:]  # dimension of inputs
        self.output_dim = self.trainY.shape[1:]
        self.iteration = int((self.trainX.shape[0] - self.trainX.shape[0] % batch_size) / batch_size)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.trainX, self.trainY)).batch(batch_size).repeat().prefetch(buffer_size=1)


if __name__ == "__main__":
    test = MarketDataGenerator(0.8, 16, 8, 4)
