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

        sampleX = Xvalid[rand:rand+1]
        sampleY = Yvalid[rand:rand+1]
        sample_pred = Predictor.predict(sampleX)

        sampleY = np.squeeze(sampleY,axis=1)
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

class FXTMDecoded:
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

    def __init__(self, train_ratio, seq_length, output_count, batch_size):

        raw = pd.read_csv("./Datasets/3FXTM1D_moving10_delta_norm_encoded.csv")

        raw = raw.iloc[:,1:]

        self.dataset = raw.to_numpy()


        self.dataset = np.expand_dims(self.dataset,axis=-1)


        train_size = int(len(self.dataset) * train_ratio)

        train_dataset = self.dataset[:train_size]
        test_dataset = self.dataset[train_size:]


        self.trainX, self.trainY = self.createTrainData_nparray(train_dataset, seq_length, output_count)
        self.testX, self.testY = self.createTrainData_nparray(test_dataset, seq_length, output_count)
        print(self.trainX.shape)
        print(self.testX.shape)

        self.batch_size = batch_size
        self.input_dim = self.trainX.shape[1:]  # dimension of inputs
        self.output_dim = self.trainY.shape[1:]

if __name__ == "__main__":
    test = FXTMDecoded(0.8, 40, 1, 1)
    gen = MarketDataGenerator
    trainX = test.trainX
    trainY = test.trainY
    testX = test.testX
    testY = test.testY

    input_dim = [16, 3, 1]
    encoder_layers = [500]
    decoder_layers = [500, input_dim[1] * input_dim[2]]
    batch_size = 32
    epoch = 10
    mfile = '.\SavedModel\VAE/VAE.h5'
    latent_dim = 25

    print(trainX.shape)
    print(trainY.shape)

    vae = SeqVAE(input_dim, encoder_layers, decoder_layers, latent_dim)
    vae.compile(input_dim, keras.optimizers.adam(epsilon=0.0001))

    vae.model.load_weights(mfile)

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(test.input_dim[0],test.input_dim[1])))
    model.add(keras.layers.CuDNNLSTM(800))
    model.add(keras.layers.Dense(test.input_dim[1], use_bias=False))
    model.add(keras.layers.Reshape(target_shape=(test.output_dim[0], test.output_dim[1])))

    model.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)
    model.summary()
    model.fit(trainX, trainY, batch_size=test.batch_size, epochs=15, validation_data=(testX, testY))



    visualize(testX, testY, model, vae)