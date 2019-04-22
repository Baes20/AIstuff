from WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from FXTMdataset import MarketDataGenerator
import os
from DecodedFXTM import FXTMDecoded
from datetime import datetime
import random
import matplotlib.pyplot as plt
from Wavenet_noRes import WaveNetMK0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#######################HyperParams#########################
train_ratio = 0.9
seq_length = 512
output_count = 8
batch_size = 32

n_pp = 16
n_filter = 64
n_fc = 128
n_layer = 8

epoch = 3
###########################################################

mfile = './SavedModel/WaveNet/WaveNet.h5'
summarydir = "./Summary/Wavenet/"
tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\Wavenet\

gen = MarketDataGenerator(train_ratio, seq_length, output_count, batch_size, ["EURUSD"], datetime(2019, 4, 20), 80000)

trainX = gen.trainX
trainY = gen.trainY
validX = gen.validX
validY = gen.validY
testX = gen.testX
testY = gen.testY

#test = WaveNetMK0(n_filter, n_fc, n_layer)
test = WaveNet(n_filter, n_pp, n_fc, n_layer)
test.compile(gen.input_dim, gen.output_dim, optimizer=Adam(lr=0.001, decay=0.01), mode=0, default_loss='mse')
# test.model_train.fit([trainX, trainY], trainY, batch_size=batch_size, epochs=epoch, callbacks=[model_saver],
#                      validation_data=([validX, validY], validY))
test.model_train.load_weights(mfile)

for i in range(20):
    rand = int(random.uniform(0, len(testX)))
    sampleX = testX[rand:rand + 1]
    sampleY = testY[rand:rand + 1]

    pred = test.predict(sampleX)
    pred = np.reshape(pred, [test.output_dim[0], test.dec_features])
    pred = gen.DeNormalize(gen.og_dataset, pred)
    pred = np.cumsum(pred,axis=0)

    sampleY = np.squeeze(sampleY)
    sampleY = gen.DeNormalize(gen.og_dataset, sampleY)
    sampleY = np.cumsum(sampleY, axis=0)

    plt.plot(sampleY)
    plt.plot(pred)
    plt.show()
