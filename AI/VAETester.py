from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from VAE2 import SeqVAE
from FXTMdataset import MarketDataGenerator
import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5

from tensorflow.python.keras.models import *

mfile = '.\SavedModel\VAE/VAE.h5'
mfile_enc = '.\SavedModel\VAE\VAE_encoder.h5'
mfile_dec = '.\SavedModel\VAE\VAE_decoder.h5'
mfile_arch = '.\SavedModel\VAE\VAE_arch.json'
mfile_enc_arch = '.\SavedModel\VAE\VAE_encoder_arch.json'
mfile_dec_arch = '.\SavedModel\VAE\VAE_decoder_arch.json'
summarydir = ".\Summary\VAE"

input_dim = [16, 3, 1]
encoder_layers = [500]
decoder_layers = [500, input_dim[1] * input_dim[2]]
batch_size = 32
epoch = 10

latent_dim = 25

gen = MarketDataGenerator(0.8, 16, 8, batch_size)
whole = gen.og_dataset

wholeX, _ = gen.createTestData_nparray(whole, 16, 16)

print(wholeX.shape)

vae = SeqVAE(input_dim, encoder_layers, decoder_layers, latent_dim)
vae.compile(input_dim, keras.optimizers.adam(epsilon=0.0001))

vae.model.load_weights(mfile)
encoded = vae.encode(wholeX)
decoded = vae.decode(encoded)
pd.DataFrame(encoded).to_csv("./Datasets/3FXTM1D_moving10_delta_norm_encoded.csv")
decoded = np.reshape(decoded, newshape=[-1, input_dim[1] * input_dim[2]])
testX = np.reshape(whole, newshape=[-1, input_dim[1] * input_dim[2]])


plt.plot(testX)
plt.plot(decoded)
plt.show()
