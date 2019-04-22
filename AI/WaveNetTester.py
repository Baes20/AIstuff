from WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import TimeDistributed
from FXTMdataset import MarketDataGenerator
import os
from EncodedFXTM import FXTMEncoded
from datetime import datetime
import random
import matplotlib.pyplot as plt
from Wavenet_noRes import WaveNetMK0
from VAE2 import SeqVAE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#######################HyperParams#########################
train_ratio = 0.9
seq_length = 256
output_count = 8
batch_size = 16

n_pp = 128
n_filter = 256
n_fc = 512
n_layer = 8

epoch = 5
###########################################################

mfile = './SavedModel/WaveNet/WaveNet.h5'
vae_mfile = '.\SavedModel\VAE/VAE.h5'
summarydir = "./Summary/Wavenet/"
tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\Wavenet\

gen = MarketDataGenerator(train_ratio, seq_length, output_count, batch_size, ["EURGBP", "EURUSD", "GBPUSD"]
                          , datetime(2019, 4, 20), 90000)
# gen = FXTMEncoded(train_ratio, seq_length, output_count, batch_size)

trainX = gen.trainX
trainY = gen.trainY
validX = gen.validX
validY = gen.validY
testX = gen.testX
testY = gen.testY

# test = WaveNetMK0(n_filter, n_fc, n_layer)
test = WaveNet(n_filter, n_pp, n_fc, n_layer)
test.compile(gen.input_dim, gen.output_dim, optimizer=Adam(lr=0.001, decay=0.01), mode=0, default_loss='mse')
test.model_train.fit([trainX, trainY], trainY, batch_size=batch_size, epochs=epoch, callbacks=[model_saver],
                    validation_data=([validX, validY], validY))
test.model_train.load_weights(mfile)

# input_dim = [16, 3, 1]
# encoder_layers = [400]
# decoder_layers = [400, input_dim[1] * input_dim[2]]
#
# vae = SeqVAE([16,3,1], encoder_layers, decoder_layers, 30)
# vae.compile(input_dim, Adam(epsilon=0.0001))
# vae.model.load_weights(vae_mfile)
# symbol_list = ["EURGBP", "EURUSD", "GBPUSD"]
# gen2 = MarketDataGenerator(0.8, 16, 8, batch_size, symbol_list, datetime(2019, 4, 20), num_samples=90000)

for i in range(20):
    rand = int(random.uniform(0, len(testX)))
    rand2 = int(random.uniform(0, 3))
    sampleX = testX[rand:rand + 1]
    sampleY = testY[rand:rand + 1]

    pred = test.predict(sampleX)
    pred = np.reshape(pred, [test.output_dim[0], test.dec_features])
    pred = gen.DeNormalize(gen.prev_dataset, pred)
    pred = np.cumsum(pred, axis=0)

    sampleY = np.squeeze(sampleY)
    sampleY = gen.DeNormalize(gen.prev_dataset, sampleY)
    sampleY = np.cumsum(sampleY, axis=0)

    plt.plot(sampleY[:,rand2])
    plt.plot(pred[:,rand2])
    plt.show()
