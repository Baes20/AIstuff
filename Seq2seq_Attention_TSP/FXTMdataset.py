from __future__ import absolute_import, division, print_function

import os
from Attention3 import DeepPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
import tensorflow as tf
import random
from tensorflow import keras
import pandas as pd

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MarketDataGenerator(object):

    def createTestData_nparray(self, data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += predLength

        return np.array(dataX), np.array(dataY)

    def createTrainData_nparray(self, data, seqLength, predLength=1):
        i = 0
        dataX = []
        dataY = []
        while (i < (len(data) - seqLength - predLength)):
            dataX.append(data[i:i + seqLength])
            dataY.append(data[i + seqLength:(i + seqLength + predLength)])
            i += 1

        return np.array(dataX), np.array(dataY)

    def normalize(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    def standardize(self, data):
        m = np.mean(data)
        stdev = np.std(data)
        return (data - m) / stdev

    def deStandardize(self, prevData, currentData):
        m = np.mean(prevData)
        stdev = np.std(prevData)
        return currentData * stdev + m

    def DeNormalize(self, prevData, currentData):
        min = np.min(prevData, 0)
        denominator = np.max(prevData, 0) - np.min(prevData, 0)
        return currentData * denominator + min

    def getMinTimeStep(self, data):
        min = data[0].shape[0]
        for i in range(len(data)):
            if (min > data[i].shape[0]):
                min = data[i].shape[0]
        return min

    def get_delta(self, Y):
        Y_shiftright = np.concatenate(([Y[0]], Y), axis=0)
        Y_shiftright = np.delete(Y_shiftright, len(Y) - 1, axis=0)
        return np.subtract(Y_shiftright, Y)

    def __init__(self, train_ratio, seq_length, output_count, batch_size):

        column_names = ['Open', 'High', 'Low', 'Close']
        GBPUSD1D_raw = pd.read_csv(filepath_or_buffer="GBPUSD1D.csv", names=column_names)
        EURGBP1D_raw = pd.read_csv(filepath_or_buffer="EURGPB1D.csv", names=column_names)
        EURUSD1D_raw = pd.read_csv(filepath_or_buffer="EURUSD1D.csv", names=column_names)
        # RATIO_raw = pd.read_csv(filepath_or_buffer="RATIO.csv", names=column_ratio_names)
        # load history
        raw_datasets = []
        # and put all of them in a list of datasets
        raw_datasets.append(GBPUSD1D_raw)
        raw_datasets.append(EURGBP1D_raw)
        raw_datasets.append(EURUSD1D_raw)
        # raw_datasets.append(RATIO_raw)
        dataset = []
        min = self.getMinTimeStep(raw_datasets)  # truncate to the minimum timestep
        for raw_dataset in raw_datasets:
            # raw_dataset.pop('Date')  # pop out the date
            # raw_dataset.pop('Time')  # pop out the time as well
            raw_dataset.pop('Open')  # pop out Open
            raw_dataset.pop('High')  # pop out High
            raw_dataset.pop('Low')  # pop out Low
            dataset.append(raw_dataset.values[:min])
        dataset = np.stack(dataset, axis=0)  # (markets, timesteps, features)
        dataset = np.transpose(dataset, (1, 0, 2))  # (timesteps, markets, features)
        # dataset_delta = self.get_delta(dataset)
        # dataset = np.concatenate((dataset, dataset_delta), axis= 2)
        # print(dataset.shape)
        self.originalData = dataset  # save the dataset as originaldata in order to de-normalize
        # dataset = normalize(dataset)  # normalize the data
        # dataset = standardize(dataset) # standardize the data
        self.dataset = dataset

        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        trainX, trainY = self.createTrainData_nparray(train_dataset, seq_length, output_count)
        self.testX, self.testY = self.createTestData_nparray(test_dataset, seq_length, output_count)
        self.trainX = trainX
        self.trainY = trainY
        self.batch_size = batch_size
        self.input_dim = trainX.shape[1:]  # dimension of inputs
        self.output_dim = trainY.shape[1:]
        self.iteration = int((trainX.shape[0] - trainX.shape[0] % batch_size) / batch_size)
        self.dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY)). \
            shuffle(len(trainX)).batch(batch_size).repeat().prefetch(buffer_size=1)
