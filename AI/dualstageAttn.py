from __future__ import absolute_import, division, print_function
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.initializers import *


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention:
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 1]))([attn, v])
        return output, attn


class MultiHeadAttention:
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [-1, s[1], n_head, d_k])  # [batch_size, len_q, n_head, d_k]
                x = tf.transpose(x, [2, 0, 1, 3])  # [n_head, batch_size, len_q, d_k]
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs,
                                        mask=mask)  # why batch*n_head: to apply attention w/o changing attn mechanism

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], d_v])  # [n_head, batch_size, len_v, d_v]
                x = tf.transpose(x, [1, 2, 0, 3])  # [batch_size, len_v, n_head, d_v]
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward:
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class ResLSTMCell(LSTMCell):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(ResLSTMCell, self).__init__(units,
                                          activation,
                                          recurrent_activation,
                                          use_bias,
                                          kernel_initializer,
                                          recurrent_initializer,
                                          bias_initializer,
                                          unit_forget_bias,
                                          kernel_regularizer,
                                          recurrent_regularizer,
                                          bias_regularizer,
                                          kernel_constraint,
                                          recurrent_constraint,
                                          bias_constraint,
                                          dropout,
                                          recurrent_dropout,
                                          implementation,
                                          )

    def call(self, inputs, states, training=None):
        new_input, new_state = super(ResLSTMCell, self).call(inputs, states, training)  # h, [h,c]
        input_added = Add()(input, new_input)
        return input_added, new_state

    def __call__(self, inputs, states):
        new_input, new_state = self.call(inputs, states)
        return new_input, new_state


class ResLSTM:
    def __init__(self, layers, return_entire_outputs=False):
        self.reo = return_entire_outputs
        self.layers = layers
        self.LSTMs = self.make_stacked_LSTMCells(layers, return_entire_outputs)
        self.mask = Masking()

    def make_stacked_LSTMCells(self, layers, return_sequence):
        LSTMs = []
        for i in range(len(layers)):
            if not return_sequence and i == len(layers) - 1:
                LSTMs.append(CuDNNLSTM(layers[i], return_sequences=False))
            else:
                LSTMs.append(CuDNNLSTM(layers[i], return_sequences=True))
        return LSTMs

    def __call__(self, input):  # input = [batch, timestep, feature]
        x = input
        for i in range(len(self.layers)):
            next = self.LSTMs[i].__call__(x)
            if i > 0 and self.layers[i] == self.layers[i-1]:
                if i == len(self.layers) - 1 and not self.reo:
                    x = next
                else:
                    x = add([x, next])
            else:
                x = next
        return x


class Encoder:
    def __init__(self, input_dim, output_dim, layers):
        pass


if __name__ == "__main__":
    inp = Input(shape=(16, 3))
    test = ResLSTM([64, 64, 64, 64, 64, 32, 32, 32, 32])
    out = test(inp)
    model = tf.keras.Model(inputs=inp, outputs=out, )
    model.summary()
