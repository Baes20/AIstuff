from __future__ import absolute_import, division, print_function

import os
import random
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from dualstageAttn import ResLSTM
from FXTMdataset import MarketDataGenerator
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SeqVAE:
    def __init__(self, input_shape, encoder_layers, decoder_layers, latent_dim, epsilon_std=0.1):
        self.encoder = ResLSTM(encoder_layers, return_entire_outputs=False)
        self.decoder = ResLSTM(decoder_layers, return_entire_outputs=True)
        self.encoder_mean = Dense(latent_dim)
        self.encoder_logvar = Dense(latent_dim)
        self.latent_dim = latent_dim
        self.e_std = epsilon_std
        self.lat_dim = latent_dim
        self.epsilon_std = epsilon_std

    def compile(self, input_dim, optimizer):
        input = Input(shape=(input_dim[0], input_dim[1], input_dim[2]))
        input_flat = Reshape(target_shape=(input_dim[0], input_dim[1] * input_dim[2]))(input)

        x = self.encoder(input_flat)
        self.z_mean = self.encoder_mean(x)
        self.z_logstd = self.encoder_logvar(x)

        latent_dim = self.latent_dim
        epsilon_std = self.epsilon_std

        @tf.function
        def sampling(z_mean, z_log_sigma):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean + z_log_sigma * epsilon

        z = Lambda(lambda x: sampling(x[0], x[1]))([self.z_mean, self.z_logstd])
        z_input = Input(shape=(self.latent_dim,))
        z_decoder = RepeatVector(input_dim[0])(z)
        _z_decoder = RepeatVector(input_dim[0])(z_input)

        decoder_out = self.decoder(z_decoder)
        _decoder_out = self.decoder(_z_decoder)
        x_prime = Reshape(target_shape=(input_dim[0], input_dim[1], input_dim[2]))(decoder_out)
        _x_prime = Reshape(target_shape=(input_dim[0], input_dim[1], input_dim[2]))(_decoder_out)

        @tf.function
        def loss(true, pred):
            kl_loss = - 0.5 * K.mean(
                1 + self.z_logstd - tf.square(self.z_mean) - K.exp(self.z_logstd))

            reconstruction_loss = K.mean(tf.keras.losses.mean_squared_error(true, pred))
            model_loss = kl_loss + reconstruction_loss

            return model_loss

        self.model = Model(input, x_prime)
        self.encoder_model = Model(input, z)
        self.decoder_model = Model(z_input, _x_prime)

        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.summary()

    def predict(self, input):
        return self.model.predict(input)

    def encode(self, input):
        return self.encoder_model.predict(input)

    def decode(self, z):
        return self.decoder_model.predict(z)


if __name__ == "__main__":
    input_dim = [16, 3, 1]
    encoder_layers = [400]
    decoder_layers = [400, input_dim[1] * input_dim[2]]
    batch_size = 32
    epoch = 150

    latent_dim = 30

    gen = MarketDataGenerator(0.8, 16, 8, batch_size)

    trainX = gen.trainX[:-2]
    print(trainX.shape)
    testX = gen.testX

    mfile = '.\SavedModel\VAE/VAE.h5'
    summarydir = ".\Summary\VAE"
    # there is a warning that it is slow, however, it's ok.
    # lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
    model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir=summarydir, write_graph=True)

    vae = SeqVAE(input_dim, encoder_layers, decoder_layers, latent_dim)
    vae.compile(input_dim, optimizer=Adam(lr=0.001, epsilon=1e-9))

    vae.model.fit(trainX, trainX, batch_size=batch_size, epochs=epoch,
                        validation_data=(testX, testX), callbacks=[model_saver, tensorboard])

    # for i in range(10):
    #     rand = int(random.uniform(0, len(testX)))
    #     sample = np.expand_dims(testX[rand], axis=0)
    #     pred = vae.predict(testX[rand:rand + 1])
    #     pred = np.reshape(pred, [16, 3])
    #     plt.plot(testX[rand, :, :, 0])
    #     plt.plot(pred)
    #     plt.show()

# EXTREMELY SUCCESSFUL!!
