from __future__ import absolute_import, division, print_function
import random, os, sys
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.initializers import *
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


class EncoderLayer:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
        output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn, enc_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

def add_pos_enc(x):
    _, max_len, d_emb = K.int_shape(x)
    pos = GetPosEncodingMatrix(max_len, d_emb)
    x_sum = K.sum(x, axis=2) # [3, 4, 5, 0, 0, 0..]
    mask = K.cast(K.not_equal(x_sum, 0), 'float32') # [1, 1, 1, 0, 0 ,0...]
    mask = K.repeat_elements(K.expand_dims(mask, -1), d_emb, axis=2)
    x = Lambda(lambda x: x[0] + (x[1] * pos))(x, mask)
    return x

def GetPadMask(q, k):
    k = K.sum(k, axis=2)
    q_shape = tf.shape(q)
    k_shape = tf.shape(k)
    # ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    ones = K.expand_dims(K.ones(shape=(q_shape[0], q_shape[1]), dtype='float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')

    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetPadMask_og(q, k):
    k = K.sum(k, axis=2)
    q_shape = tf.shape(q)
    k_shape = tf.shape(k)
    # ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    ones = K.expand_dims(K.ones(shape=(q_shape[0], q_shape[1]), dtype='float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')

    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


# test = tf.random.uniform([8, 16])
# submask = GetSubMask(test)
# padmask = GetPadMask_og(test, test)
# mask = tf.minimum(padmask, submask)
# print(mask)

class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        # self.emb_layer = word_emb
        # self.pos_layer = pos_emb
        # self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        # x = self.emb_layer(src_seq)
        # x = add_pos_enc(src_seq)
        x = src_seq
        if src_pos is not None:
            pos = src_pos
            x = Add()([x, pos])
        # x = self.emb_dropout(x)
        if return_att: atts = []
        mask = Lambda(lambda x: GetPadMask_og(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        # self.emb_layer = word_emb
        # self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
        # x = add_pos_enc(tgt_seq)
        x = tgt_seq
        if tgt_pos is not None:
            pos = self.pos_layer(tgt_pos)
            x = Add()([x, pos])

        self_pad_mask = Lambda(lambda x: GetPadMask_og(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []
        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)
        return (x, self_atts, enc_atts) if return_att else x


class Transformer:
    def __init__(self, input_dim, output_last_dim, len_limit, d_model=256,
                 d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1,
                 share_word_emb=False):
        self.src_loc_info = True
        self.d_model = d_model
        self.len_limit = len_limit

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout)
        self.target_layer = TimeDistributed(Dense(output_last_dim, use_bias=False))

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask



    def compile(self, input_dim, output_dim, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(input_dim[0], input_dim[1] * input_dim[2]), dtype='float32')
        tgt_seq_input = Input(shape=(output_dim[0], output_dim[1] * output_dim[2]), dtype='float32')

        # src_seq_input = tf.random.uniform(shape=(8, input_dim[0], input_dim[1] * input_dim[2])) #for debugging
        # tgt_seq_input = tf.random.uniform(shape=(8, output_dim[0], output_dim[1] * output_dim[2]))

        src_seq = src_seq_input
        self.GO = Lambda(lambda x: x[:, output_dim[0]:output_dim[0] + 1])(src_seq_input)

        def make_dec_inp(tgt, G):
            return K.concatenate([G, tgt[:, :-1]], axis=1)

        dec_inp = Lambda(lambda x: make_dec_inp(x[0], x[1]))([tgt_seq_input, self.GO])
        tgt_true = tgt_seq_input

        if not self.src_loc_info: src_pos = None

        enc_output = self.encoder(src_seq, None, active_layers=active_layers)
        dec_output = self.decoder(dec_inp, None, src_seq, enc_output, active_layers=active_layers)
        final_output = self.target_layer(dec_output)

        def get_loss(args):
            y_pred, y_true = args
            # y_true = tf.cast(y_true, 'int32')
            loss = (y_true - y_pred) ** 2 / 2
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([final_output, tgt_true])
        self.model = Model([src_seq_input, tgt_seq_input], loss)
        self.model.add_loss([loss])
        self.output_model = Model([src_seq_input, tgt_seq_input], final_output)
        self.model.compile(optimizer, loss=None)

    def reshape(self, input_seq):
        if len(input_seq.shape) > 2:  # timestep, symbols, OHLC
            input_seq = np.reshape(input_seq, (1, input_seq.shape[0], input_seq.shape[1] * input_seq.shape[2]))
        else:
            input_seq = np.reshape(input_seq, (1, input_seq.shape[0], input_seq.shape[1]))
        return input_seq

    def predict(self, input_seq, output_count):  # input_seq = [t, symbols, OHLC] or [t, symbols*OHLC]
        src_seq = self.reshape(input_seq)  # [1, timestep, symbols*OHLC]
        decoded = []
        target_seq = np.zeros((1, output_count, src_seq.shape[2]), dtype='float32')  # [1,len_limit,symbols*OHLC]
        for i in range(output_count):
            output = self.output_model.predict_on_batch([src_seq, target_seq])
            current = output[0, i, :]
            decoded.append(current)
            target_seq[0, i] = current
        decoded = np.stack(decoded,axis=0)
        return decoded


    def decoder_sequence(self, input_seq, delimiter=''):
        return input_seq

class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = K.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = Lambda(lambda x: x + pos)(x)
        return x


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])
