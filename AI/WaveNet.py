from tensorflow.python.keras.layers import *
import tensorflow.python.keras as keras
import tensorflow as tf
import os
from FXTMdataset import MarketDataGenerator
from tensorflow.python.keras.callbacks import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WaveNet:
    def __init__(self, n_conv_filters, pp_layers, n_fc, num_layer, filter_width=2):
        self.dilation_rates = [2 ** i for i in range(num_layer)]
        self.preprocess_layers = []
        self.filter_layers = []
        self.gate_layers = []
        self.postprocess_layers = []
        self.fc = Dense(n_fc, activation='relu')
        self.model_train = None
        self.model_infer = None
        self.input_dim = None
        self.output_dim = None

        for dilation_rate in self.dilation_rates:
            self.preprocess_layers.append(Conv1D(pp_layers, 1, padding='same', activation='relu'))

            self.filter_layers.append(Conv1D(filters=n_conv_filters, kernel_size=filter_width,
                                             padding='causal', dilation_rate=dilation_rate))

            self.gate_layers.append(Conv1D(filters=n_conv_filters, kernel_size=filter_width,
                                           padding='causal', dilation_rate=dilation_rate))

            self.postprocess_layers.append(Conv1D(pp_layers, 1, padding='same', activation='relu'))

    def compile(self, input_dim, output_dim, optimizer, mode=0, default_loss='mse'):
        # mode=0: traditional, else: train enc_inp as well
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.enc_features = input_dim[1] * input_dim[2]
        self.dec_features = output_dim[1] * output_dim[2]
        projection = Dense(self.dec_features, use_bias=False)

        # input = concat(encoder_seq + decoder_seq[:,-1,:]
        encoder_input = Input(shape=(None, input_dim[1], input_dim[2]), name="encoder_in")
        encoder_input_flat = Reshape(target_shape=(-1, self.enc_features), name="enc_reshape")(encoder_input)
        encoder_input_first_out = Lambda(lambda x: x[:, 1:, :], name="encoder_take_away_first")(encoder_input_flat)

        decoder_input = Input(shape=(None, output_dim[1], output_dim[2]), name="decoder_in")
        decoder_input_flat = Reshape(target_shape=(-1, self.dec_features), name="dec_reshape")(decoder_input)
        decoder_input_lagged = Lambda(lambda x: x[:, :-1, :], name="decoder_take_away_last")(decoder_input_flat)

        input_train = Concatenate(axis=1)([encoder_input_flat, decoder_input_lagged])
        input_infer = Concatenate(axis=1)([encoder_input_first_out, decoder_input_flat])

        def feedforward(x):

            skips = []

            for i in range(len(self.dilation_rates)):
                x = self.preprocess_layers[i](x)
                x_f = self.filter_layers[i](x)
                x_g = self.gate_layers[i](x)

                z = Multiply()([Activation('tanh')(x_f), Activation('sigmoid')(x_g)])
                z = self.postprocess_layers[i](z)

                x = Add()([x, z])

                skips.append(z)

            out = Activation('relu')(Add()(skips))

            out = self.fc(out)
            out = projection(out)

            return out

        train_x = feedforward(input_train)
        infer_x = feedforward(input_infer)

        if mode == 1:
            train_pred = train_x
            groundtruth = Concatenate(axis=1)([encoder_input_flat, decoder_input_flat])
            groundtruth = Lambda(lambda x: x[:, 1:, :])(groundtruth)

            def loss(true, pred):
                l = keras.losses.mean_squared_error(groundtruth, train_pred)
                return l

            self.model_train = keras.models.Model([encoder_input, decoder_input], train_pred)
            self.model_train.compile(optimizer=optimizer, loss=loss)

        else:
            train_pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(train_x)
            train_pred = Reshape(target_shape=(output_dim[0], output_dim[1], output_dim[2]))(train_pred)

            infer_pred = Lambda(lambda x: x[:, -output_dim[0]:, :], name="train_prediction")(infer_x)
            infer_pred = Reshape(target_shape=(output_dim[0], output_dim[1], output_dim[2]))(infer_pred)

            self.model_train = keras.models.Model([encoder_input, decoder_input], train_pred)
            self.model_infer = keras.models.Model([encoder_input, decoder_input], infer_pred)

            self.model_train.compile(optimizer=optimizer, loss=default_loss)

        self.model_train.summary()
        self.model_infer.summary()

    def predict(self, input):  # assuming input = (1, input_seq, symbols, features)
        history_seq = input

        # output = (1, output_seq, symbols, features)
        pred_seq = np.zeros((1, self.output_dim[0], self.output_dim[1], self.output_dim[2]))

        for i in range(self.output_dim[0]):
            encoder_input = history_seq[:, :-self.output_dim[0], :, :]
            decoder_input = history_seq[:, -self.output_dim[0]:, :, :]
            last_step_pred = self.model_infer.predict([encoder_input, decoder_input])[:, -1, :, :]
            pred_seq[:, i, :, :] = last_step_pred
            last_step_pred = np.expand_dims(last_step_pred, axis=1)

            history_seq = np.concatenate([history_seq, last_step_pred], axis=1)

        return pred_seq
