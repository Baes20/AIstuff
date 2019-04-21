from WaveNet import WaveNet
from tensorflow.python.keras.callbacks import *
from FXTMdataset import MarketDataGenerator
import os
from DecodedFXTM import FXTMDecoded

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#######################HyperParams#########################
train_ratio = 0.8
seq_length = 1200
output_count = 16
batch_size = 32

n_filter = 32
n_fc = 64
n_layer = 10

epoch = 200
###########################################################

mfile = './SavedModel/WaveNet/WaveNet.h5'
summarydir = "./Summary/Wavenet/"
tensorboard = TensorBoard(log_dir=summarydir, write_graph=True, histogram_freq=1)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
# tensorboard --host 127.0.0.1 --logdir=D:\Projects\AI\Summary\Wavenet\

gen = MarketDataGenerator(train_ratio, seq_length, output_count, batch_size)

trainX = gen.trainX
trainY = gen.trainY
validX = gen.validX
validY = gen.validY
testX = gen.testX
testY = gen.testY

test = WaveNet(n_filter, n_fc, n_layer)
test.compile(gen.input_dim, gen.output_dim, optimizer='adam', mode=0, default_loss='mse')
test.model_train.fit([trainX, trainY], trainY, batch_size=128, epochs=epoch, callbacks=[model_saver],
                     validation_data=([validX, validY], validY))
