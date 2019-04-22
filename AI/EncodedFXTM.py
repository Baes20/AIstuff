import tensorflow as tf
import pandas as pd

# Helper libraries
import numpy as np
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
import random
import os
from FXTMdataset import MarketDataGenerator
from VAE2 import SeqVAE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualize(Xvalid, Yvalid, Predictor, decoder):
    for i in range(10):
        rand = int(random.uniform(0, Xvalid.shape[0]))
        rand2 = int(random.uniform(0, Xvalid.shape[2]))

        sampleX = Xvalid[rand:rand + 1]
        sampleY = Yvalid[rand:rand + 1]
        sample_pred = Predictor.predict(sampleX)

        sampleY = np.squeeze(sampleY, axis=1)
        sample_pred = np.squeeze(sample_pred, axis=1)

        sampleY = decoder.decode(sampleY)
        sample_pred = decoder.decode(sample_pred)

        sampleY = np.squeeze(sampleY, axis=0)
        sampleY = np.squeeze(sampleY)
        sample_pred = np.squeeze(sample_pred, axis=0)
        sample_pred = np.squeeze(sample_pred)

        plt.plot(sampleY)
        plt.plot(sample_pred)
        plt.show()


class FXTMEncoded:
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
    @staticmethod
    def normalize(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)
    @staticmethod
    def DeNormalize(prevData, currentData):
        min = np.min(prevData, 0)
        denominator = np.max(prevData, 0) - np.min(prevData, 0)
        return currentData * denominator + min

    def __init__(self, train_ratio, seq_length, output_count, batch_size):

        raw = pd.read_csv("./Datasets/3FXTM1M_exp_moving10_delta_norm_encoded.csv")

        raw = raw.iloc[:, 1:]

        self.dataset = raw.to_numpy()

        self.prev_dataset = self.dataset
        self.dataset = self.normalize(self.dataset)

        self.dataset = np.expand_dims(self.dataset, axis=-1)

        train_size = int(len(self.dataset) * train_ratio)

        train_dataset = self.dataset[:train_size]
        test_dataset = self.dataset[train_size:]

        self.trainX, self.trainY = self.createTrainData_nparray(train_dataset, seq_length, output_count)
        self.validX, self.validY = self.createTrainData_nparray(test_dataset, seq_length, output_count)
        self.testX, self.testY = self.createTestData_nparray(test_dataset, seq_length, output_count)
        print(self.trainX.shape)
        print(self.testX.shape)

        self.batch_size = batch_size
        self.input_dim = self.trainX.shape[1:]  # dimension of inputs
        self.output_dim = self.trainY.shape[1:]


if __name__ == "__main__":
    test = FXTMEncoded(0.8, 40, 1, 1)
    plt.plot(np.squeeze(test.dataset))
    plt.show()



    # input_dim = [16, 3, 1]
    # encoder_layers = [400]
    # decoder_layers = [400, input_dim[1] * input_dim[2]]
    # batch_size = 32
    # epoch = 10
    # mfile = '.\SavedModel\VAE/VAE.h5'
    # latent_dim = 30


