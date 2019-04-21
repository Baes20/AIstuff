from FXTMdataset import MarketDataGenerator

from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
import random
import matplotlib.pyplot as plt

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''


def visualize(Xvalid, Yvalid, Transformer):
    for i in range(10):
        rand = int(random.uniform(0, Xvalid.shape[0]))
        sampleX = Xvalid[rand]
        sampleY = Yvalid[rand]
        sample_pred = Transformer.predict(sampleX, output_length)

        plt.plot(np.concatenate((sampleX, sampleY)))
        plt.plot(np.concatenate((sampleX, sample_pred)))
        plt.show()

        rand2 = int(random.uniform(0, Xtrain.shape[0]))
        sampleX2 = Xtrain[rand2]
        sampleY2 = Ytrain[rand2]
        sample_pred2 = Transformer.predict(sampleX2, output_length)
        plt.plot(np.concatenate((sampleX2, sampleY2)))
        plt.plot(np.concatenate((sampleX2, sample_pred2)))
        plt.show()


seq_length = 128
output_length = 64
batch_size = 16
epoch = 1

gen = MarketDataGenerator(train_ratio=0.6, seq_length=seq_length, output_count=output_length, batch_size=batch_size)
# gen = NasdaqGenerator(train_ratio=0.8, seq_length=seq_length, output_count=output_length, batch_size=batch_size)
Xtrain = np.squeeze(gen.trainX)
Ytrain = np.squeeze(gen.trainY)
Xvalid = np.squeeze(gen.testX)
Yvalid = np.squeeze(gen.testY)
input_dim = gen.input_dim
output_dim = gen.output_dim
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

from Transformer2 import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

d_model = gen.output_dim[1]  # == 3
s2s = Transformer(seq_length, output_last_dim=input_dim[1] * input_dim[2], len_limit=output_length, d_model=d_model,
                  d_inner_hid=6,
                  n_head=32, d_k=3, d_v=3, layers=8, dropout=0.)

mfile = './models/Transformer/en2de.model.h5'
summarydir = "./models/summary/Transformer"

lr_scheduler = LRSchedulerPerStep(d_model, 4000)

model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(input_dim, output_dim, Adam(0.001, 0.9, 0.98, epsilon=1e-9))

s2s.model.summary()

tensorboard = TensorBoard(log_dir=summarydir, write_graph=True)

s2s.model.fit([Xtrain, Ytrain], None, batch_size=batch_size, epochs=epoch,
              validation_data=([Xvalid, Yvalid], None),
              callbacks=[lr_scheduler, model_saver, tensorboard])

visualize(Xvalid, Yvalid, s2s)

# if 'test' in sys.argv:
#     print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
#     while True:
#         quest = input('> ')
#         print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
#         rets = s2s.beam_search(quest.split(), delimiter=' ')
#         for x, y in rets: print(x, y)
# else:
