from FXTMdataset import MarketDataGenerator
from Attention3 import DeepPredictor
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
from NASDAQ100dataset import NasdaqGenerator
from Luong import Luong


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


# tensorboard --host 127.0.0.1 --logdir=D:/Projects/tensor2/summary/Attention3
train_ratio = 0.8
seq_length = 32
output_count = 8
batch_size = 8
encoder_layers = decoder_layers = [64, 64, 64]
epoch = 3000
learning_rate = 0.0001

tf.reset_default_graph()
generator = MarketDataGenerator(train_ratio, seq_length, 8, batch_size)
predictor = Luong(generator, encoder_layers, decoder_layers, isAttending=True)
print(predictor.count_all_trainable_parameters())
#predictor.train(epoch, learning_rate, continue_from_ckpt=False)

trainPredict = predictor.predict(generator.trainX)
testPredict = predictor.predict(generator.testX)
test_acc = trend_accuracy(generator.testY, testPredict)
train_acc = trend_accuracy(generator.trainY, trainPredict)

testY = generator.testY
testPredict_reshaped = np.reshape(testPredict, (testPredict.shape[0] * testPredict.shape[1],
                                                testPredict.shape[2], testPredict.shape[3]))
testY_reshaped = np.reshape(testY, (testY.shape[0] * testY.shape[1],
                                    testY.shape[2], testY.shape[3]))
plt.plot(testY_reshaped[:, 0, 0])
plt.plot(testPredict_reshaped[:, 0, 0])
plt.show()

print(test_acc)
show_rand_sample(generator.testY, testPredict, 5)
print(train_acc)
show_rand_sample(generator.trainY, trainPredict, 5)
