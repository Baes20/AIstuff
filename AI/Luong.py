from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def getHparamString(learningRate, Layers, Attention, seq_length, output_count):
    lrStr = "lR=%f" % learningRate
    layerStr = "".join(str(f) + "," for f in Layers)
    return lrStr + ",layer=" + layerStr + "Attn=" + str(Attention) + ",seqLength=" + str(seq_length) + \
           ",outputlength="+ str(output_count)


class Decoder:
    def __init__(self, hidden_dim, output_dim):
        self.fc1 = tf.layers.Dense(hidden_dim, activation=tf.nn.relu, name="DecoderBlock_1")
        self.fc2 = tf.layers.Dense(int(hidden_dim / 2), activation=tf.nn.relu, name="DecoderBlock_2")
        self.fc3 = tf.layers.Dense(output_dim, activation=tf.nn.tanh, name="DecoderBlock_3")

    def __call__(self, input):
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class AttnProjection:
    def __init__(self, hidden_dim, output_dim, encoder_outputs):
        self.encoder_outputs = encoder_outputs
        self.fc1 = tf.layers.Dense(hidden_dim, activation=tf.nn.relu, name="Attn_fc1")
        self.fc2 = tf.layers.Dense(hidden_dim / 2, activation=tf.nn.tanh, name="Attn_fc2")
        self.score = tf.layers.Dense(1, activation=tf.nn.tanh, name="Attn_fc3")
        self.projection = tf.layers.Dense(output_dim, activation=None, name="projection_layer")

    def __call__(self, decoder_output):
        scores = []
        unstacked_outputs = tf.unstack(self.encoder_outputs, axis=1)
        for encoder_output in unstacked_outputs:  # for each output in outputs
            cat = tf.concat([encoder_output, decoder_output], axis=1)  # concatenate with decoder output
            out = self.fc1(cat)  # calculate
            out = self.fc2(out)  # correlation between two
            score = self.score(out)  # and get a score
            scores.append(score)  # put it in a vector

        scores = tf.stack(scores, axis=1)  # turn it into tensor
        scores = tf.reshape(scores, [-1, len(unstacked_outputs)])  # reshape it
        scores = tf.nn.softmax(scores, axis=1)  # softmax it
        scores = tf.reshape(scores, [-1, len(unstacked_outputs), 1])  # get ready to multiply with encoder

        context = tf.reduce_sum(tf.multiply(self.encoder_outputs, scores), axis=1)
        c_d_concat = tf.concat([context, decoder_output], axis=1)
        output = self.projection(c_d_concat)
        return output


class Luong:

    def __init__(self, generator, layers, decoder_layers, isAttending=False):
        self.input_dim = generator.input_dim
        self.output_dim = generator.output_dim
        self.layers = layers
        self.isAttending = isAttending
        self.iter = generator.dataset.make_one_shot_iterator()
        self.iteration = generator.iteration
        self.testX = generator.testX
        self.testY = generator.testY

        X, Y = self.iter.get_next()
        X = tf.dtypes.cast(X, dtype=tf.float32)
        Y = tf.dtypes.cast(Y, dtype=tf.float32)

        with tf.name_scope("X"):
            self.X = X
            self.X_flattened = tf.reshape(self.X, [-1, self.input_dim[0], self.input_dim[1] * self.input_dim[2]],
                                          name="X_flattened")

        with tf.name_scope("rnn_encoder"):
            self.encoder_outputs, self.encoder_final_state = self.build_encoder(layers, self.X_flattened)

        with tf.name_scope("Y"):
            self.Y = Y
            self.Y_flattened = tf.reshape(self.Y, [-1, self.output_dim[0], self.output_dim[1] * self.output_dim[2]],
                                          name="Y_flattened")

        with tf.name_scope("Decoder_input"):
            self.dec_inp, self.GO = self.build_decoder_input(self.Y)

        with tf.name_scope("Attention"):
            self.projection_layer = AttnProjection(int(self.layers[0] / 2), self.output_dim[1] * self.output_dim[2],
                                                   self.encoder_outputs)

        with tf.name_scope("rnn_decoder"):
            self.decoder_cell = self.build_residual_rnn_cell(decoder_layers)

            with tf.name_scope("rnn_decoder_train"):
                self.train_pred = self.build_decoder_train(self.decoder_cell,
                                                           self.encoder_final_state,
                                                           self.dec_inp,
                                                           self.projection_layer)

            with tf.name_scope("rnn_decoder_infer"):
                self.infer_pred = self.build_decoder_infer(self.decoder_cell,
                                                           self.encoder_final_state,
                                                           self.projection_layer,
                                                           GO=self.GO)

        self.saver = tf.train.Saver()

    def build_decoder_input(self, dec_inp_raw):
        shape = [tf.shape(self.X)[0], 1, self.output_dim[1] * self.output_dim[2]]
        GO = tf.zeros(shape, name="GO", dtype=tf.float32)
        dec_inp_flat = tf.reshape(dec_inp_raw, [-1, self.output_dim[0], self.output_dim[1] * self.output_dim[2]])

        dec_inp_unstacked = tf.unstack(dec_inp_flat, axis=1)
        dec_inp_first_removed = dec_inp_unstacked[1:]
        dec_inp_restacked = tf.stack(dec_inp_first_removed, axis=1)

        dec_inp = tf.concat([GO, dec_inp_restacked], axis=1)
        return dec_inp, GO

    @staticmethod
    def normalize_tensor(tensor):
        tensor = tf.div(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor)
            ),
            tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            )
        )
        return tensor

    def count_all_trainable_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    @staticmethod
    def build_residual_rnn_cell(layers):
        firstLayerRnn = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            num_units=layers[0])]

        cells = firstLayerRnn + [tf.nn.rnn_cell.ResidualWrapper(
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=num_units)) for num_units in layers]
        # cells = [tf.nn.rnn_cell.ResidualWrapper(
        #     tf.nn.rnn_cell.GRUCell(num_units=num_units)) for num_units in layers]

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def build_encoder(self, encoder_layers, encoder_input):
        encoder_cell = self.build_residual_rnn_cell(encoder_layers)
        outputs, final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input, dtype=tf.float32)
        return outputs, final_state

    def build_decoder_train(self, decoder_cell, encoder_final_state, decoder_input, projection_layer):
        state = encoder_final_state
        outputs = []
        dec_inp = tf.unstack(decoder_input, axis=1)
        for i in range(self.output_dim[0]):
            prev = dec_inp[i]
            output, state = decoder_cell(prev, state)
            output = projection_layer(output)
            outputs.append(output)

        train_pred_flat = tf.stack(outputs, axis=1)  # [Batch, Timestep, feature] again
        train_pred = tf.reshape(train_pred_flat, shape=[-1, self.output_dim[0], self.output_dim[1], self.output_dim[2]])

        return train_pred

    def build_decoder_infer(self, decoder_cell, encoder_final_state, projection_layer, GO):
        infer_state = encoder_final_state
        GO = tf.reshape(GO, shape=[-1, self.output_dim[1] * self.output_dim[2]])
        prev = GO
        infer_outputs = []
        for i in range(self.output_dim[0]):
            infer_output, infer_state = decoder_cell(prev, infer_state)
            infer_final_output = projection_layer(infer_output)
            infer_outputs.append(infer_final_output)
            prev = infer_final_output

        infer_pred_flat = tf.stack(infer_outputs, axis=1)
        infer_pred = tf.reshape(infer_pred_flat, shape=[-1, self.output_dim[0], self.output_dim[1], self.output_dim[2]])

        return infer_pred

    def train(self, epoch, learning_rate, continue_from_ckpt=False):

        with tf.name_scope("loss"):
            loss = tf.losses.mean_squared_error(self.Y, self.train_pred)
            val_loss = tf.losses.mean_squared_error(self.testY, self.infer_pred)
            # loss = tf.losses.huber_loss(self.Y, self.train_pred)
            # val_loss = tf.losses.huber_loss(self.testY, self.infer_pred)
            tf.summary.scalar('val_loss', val_loss)

        with tf.name_scope("metrics"):
            MAE_train = tf.reduce_mean(tf.metrics.mean_absolute_error(self.Y, self.infer_pred))
            MRE_train = tf.reduce_mean(tf.metrics.mean_relative_error(self.Y, self.infer_pred, self.Y))
            tf.summary.scalar('MAE', MAE_train)
            tf.summary.scalar('MRE', MRE_train)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train = optimizer.minimize(loss, global_step=global_step)

        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            if continue_from_ckpt:
                sess.run(init)
                self.saver.restore(sess, "./LuongModel/Luong.ckpt")
            else:
                sess.run(init)

            HparamString = getHparamString(learningRate=learning_rate,
                                           Layers=self.layers,
                                           Attention=self.isAttending,
                                           seq_length=self.input_dim[0],
                                           output_count=self.output_dim[0])
            print(HparamString)

            writer = tf.summary.FileWriter("./summary/Luong/" + HparamString)
            merged_summary = tf.summary.merge_all()
            writer.add_graph(sess.graph)

            for i in range(epoch):
                iteration_start_time = time.time()
                _, L = sess.run([train, loss])
                if i % 10 == 0:
                    s = sess.run(merged_summary, feed_dict={self.X: self.testX, self.Y: self.testY})
                    writer.add_summary(s, i)
                elapsed_time = time.time() - iteration_start_time
                print("epoch:",i, "loss:", L,"elapsed time:", elapsed_time)

            save_path = self.saver.save(sess, "./LuongModel/Luong.ckpt")
            print("Model saved in path: %s" % save_path)


    def predict(self, test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./LuongModel/Luong.ckpt")
            predict = sess.run(self.infer_pred, feed_dict={self.X: test})
        return predict
