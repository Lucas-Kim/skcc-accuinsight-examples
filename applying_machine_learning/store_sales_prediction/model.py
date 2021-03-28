#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import prep

np.random.seed(42)
tf.random.set_seed(42)

class AutoRegressionLSTM(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell_warmup = tf.keras.layers.LSTMCell(units)
        self.lstm_cell_decoder = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell_warmup, return_state=True)
        self.dense1 = tf.keras.layers.Dense(8, activation='tanh')  # 시간마다 sales 만 뱉어냄
        self.dense2 = tf.keras.layers.Dense(1, activation='linear') # 시간마다 sales 만 뱉어냄

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        '''
        Output : 
        If return_state: a list of tensors. The first tensor is the output. 
        The remaining tensors are the last states, each with shape [batch_size, state_size], 
        where state_size could be a high dimension tensor shape.
        '''
        # print(x)
        # print(x.shape)  => (# of sample, # of units), x와 state[0]은 동일함(h_t), state[1]은 c_t 의미
        # predictions.shape => (batch, features)

        prediction1 = self.dense1(x)
        prediction = self.dense2(prediction1)

        # prediction = self.dense(x) # seq2seq와 다른 부분. encoder RNN 의 결과로 prediction 시작.
                                   # seq2seq는 encoding 모두 완료하고, decoder에서 prediction 시작
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        
        # Insert the first prediction
        predictions.append(prediction)
        
        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            x, state = self.lstm_cell_decoder(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction1 = self.dense1(x)
            prediction = self.dense2(prediction1)
            # Add the prediction to the output
            predictions.append(prediction)
        
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions




####################### Dual-stage Attention based model


class Encoderlstm(Layer):
    def __init__(self, m):
        """
        m : feature dimension (# of units)
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Encoderlstm, self).__init__(name="encoder_lstm")
        self.lstm = LSTM(m, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,m)
        c_s : cell_state (shape = batch,m)
        x : time series encoder inputs (shape = batch,T,n)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, m*2
        query = RepeatVector(x.shape[2])(query)  # batch, n, m*2
        x_perm = Permute((2, 1))(x)  # batch, n, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, n, T
        score = self.v(score)  # batch, n, 1
        score = Permute((2, 1))(score)  # batch,1,n
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 feature 별 중요도
        return attention_weights


class Encoder(Layer):
    def __init__(self, T, m):
        super(Encoder, self).__init__(name="encoder")
        self.T = T
        self.input_att = InputAttention(T)
        self.lstm = Encoderlstm(m)
        self.initial_state = None
        self.alpha_t = None

    def call(self, data, h0, c0, n, training=False):
        """
        data : encoder data (shape = batch, T, n)
        n : data feature num
        """
        self.lstm.reset_state(h0=h0, c0=c0)
        alpha_seq = tf.TensorArray(tf.float32, self.T)
        for t in range(self.T):
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  # (batch,1,n)

            h_s, c_s = self.lstm(x)

            self.alpha_t = self.input_att(h_s, c_s, data)  # batch,1,n

            alpha_seq = alpha_seq.write(t, self.alpha_t)
        alpha_seq = tf.reshape(alpha_seq.stack(), (-1, self.T, n))  # batch, T, n
        output = tf.multiply(data, alpha_seq)  # batch, T, n

        return output


class Decoderlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(Decoderlstm, self).__init__(name="decoder_lstm")
        self.lstm = LSTM(p, return_state=True)
        self.initial_state = None

    def call(self, x, training=False):
        """
        x : t 번째 input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]


class TemporalAttention(Layer):
    def __init__(self, m):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(m)
        self.w2 = Dense(m)
        self.v = Dense(1)

    def call(self, h_s, c_s, enc_h):
        """
        h_s : hidden_state (shape = batch,p)
        c_s : cell_state (shape = batch,p)
        enc_h : time series encoder inputs (shape = batch,T,m)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, p*2
        query = RepeatVector(enc_h.shape[1])(query)
        score = tf.nn.tanh(self.w1(enc_h) + self.w2(query))  # batch, T, m
        score = self.v(score)  # batch, T, 1
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # encoder hidden state h(i) 의 중요성 (0<=i<=T)
        return attention_weights


class Decoder(Layer):
    def __init__(self, T, p, m):
        super(Decoder, self).__init__(name="decoder")
        self.T = T
        self.temp_att = TemporalAttention(m)
        self.dense = Dense(1)
        self.lstm = Decoderlstm(p)
        self.enc_lstm_dim = m
        self.dec_lstm_dim = p
        self.context_v = None
        self.dec_h_s = None
        self.beta_t = None

    def call(self, data, enc_h, h0=None, c0=None, training=False):
        """
        data : decoder data (shape = batch, T-1, 1)
        enc_h : encoder hidden state (shape = batch, T, m)
        """
        h_s = None
        self.lstm.reset_state(h0=h0, c0=c0)
        self.context_v = tf.zeros((enc_h.shape[0], 1, self.enc_lstm_dim))  # batch,1,m
        self.dec_h_s = tf.zeros((enc_h.shape[0], self.dec_lstm_dim))  # batch, p
        for t in range(self.T - 1):  # 0~T-1
            x = Lambda(lambda x: data[:, t, :])(data)
            x = x[:, tf.newaxis, :]  #  (batch,1,1)
            x = tf.concat([x, self.context_v], axis=-1)  # batch, 1, m+1
            x = self.dense(x)  # batch,1,1

            h_s, c_s = self.lstm(x)  # batch,p

            self.beta_t = self.temp_att(h_s, c_s, enc_h)  # batch, T, 1

            self.context_v = tf.matmul(
                self.beta_t, enc_h, transpose_a=True
            )  # batch,1,m
        return tf.concat(
            [h_s[:, tf.newaxis, :], self.context_v], axis=-1
        )  # batch,1,m+p


class DARNN(Model):
    def __init__(self, T, m, p, n):
        super(DARNN, self).__init__(name="DARNN")
        """
        T : 주기 (time series length)
        m : encoder lstm feature length
        p : decoder lstm feature length
        h0 : lstm initial hidden state
        c0 : lstm initial cell state
        """
        self.m = m
        self.n = n
        self.encoder = Encoder(T=T, m=m)
        self.decoder = Decoder(T=T, p=p, m=m)
        self.lstm = LSTM(m, return_sequences=True)
        self.dense1 = Dense(p)
        self.dense2 = Dense(1)

    def call(self, inputs, training=False, mask=None):
        """
        inputs : [enc , dec]
        enc_data : batch,T,n
        dec_data : batch,T-1,1
        """
        enc_data, dec_data = inputs
        batch = enc_data.shape[0]
        h0 = tf.zeros((batch, self.m))
        c0 = tf.zeros((batch, self.m))
        enc_output = self.encoder(
            enc_data, n=self.n, h0=h0, c0=c0, training=training
        )  # batch, T, n
        enc_h = self.lstm(enc_output)  # batch, T, m
        dec_output = self.decoder(
            dec_data, enc_h, h0=h0, c0=c0, training=training
        )  # batch,1,m+p
        output = self.dense2(self.dense1(dec_output))
        output = tf.squeeze(output)
        return output





#################### train DARNN

class model_DARNN():
    def __init__(self, train_data, test_data, interval, m, p, n, batch_size, learning_rate, epochs):
        '''
        :param interval : interval of time series
        :param m: encoder lstm unit length
        :param p: decoder lstm unit length
        :param n: number of features
        '''
        self.pre_darnn = prep.Preprocess_DARNN(train_data, test_data, interval)
        self.pre_darnn.Show_shape(option='train')
        print('---------------------------------------')
        self.pre_darnn.Show_shape(option='test')

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = DARNN(T=interval, m=m, p=p, n=n)
        self.interval = interval
        self.n = n
        self.train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.pre_darnn.encoder_sequence_tr, self.pre_darnn.decoder_sequence_tr, self.pre_darnn.target_tr)
            )
            .batch(self.batch_size)
            .shuffle(buffer_size=self.pre_darnn.encoder_sequence_tr.shape[0])
            .prefetch(tf.data.experimental.AUTOTUNE)
            )

        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.pre_darnn.encoder_sequence_ts, self.pre_darnn.decoder_sequence_ts, self.pre_darnn.target_ts)
            )\
            .batch(self.batch_size)

    def createModel(self):
        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.pre_darnn.encoder_sequence_tr, self.pre_darnn.decoder_sequence_tr, self.pre_darnn.target_tr)
            )
                .batch(self.batch_size)
                .shuffle(buffer_size=self.pre_darnn.encoder_sequence_tr.shape[0])
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            (self.pre_darnn.encoder_sequence_ts, self.pre_darnn.decoder_sequence_ts, self.pre_darnn.target_ts)
        ) \
            .batch(self.batch_size)

        @tf.function
        def train_step(model, inputs, labels, loss_fn, optimizer, train_loss):
            with tf.GradientTape() as tape:
                prediction = model(inputs, training=True)
                loss = loss_fn(labels, prediction)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        @tf.function
        def test_step(model, inputs, labels, loss_fn, test_loss):
            prediction = model(inputs, training=False)
            loss = loss_fn(labels, prediction)
            test_loss(loss)
            return prediction

        loss_fn = tf.keras.losses.MSE

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        test_loss = tf.keras.metrics.Mean(name="test_loss")
        train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
        test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")
        history_loss = []

        for epoch in range(self.epochs):
            for enc_data, dec_data, labels in train_ds:
                inputs = [enc_data, dec_data]
                train_step(self.model, inputs, labels, loss_fn, optimizer, train_loss)

            template = "Epoch {}, Loss: {}"
            print(template.format(epoch + 1, train_loss.result()))
            history_loss.append(train_loss.result().numpy())
            train_loss.reset_states()
            test_loss.reset_states()

        i = 0
        for enc_data, dec_data, label in test_ds:
            inputs = [enc_data, dec_data]
            pred = test_step(self.model, inputs, label, loss_fn, test_loss)
            if i == 0:
                preds = pred.numpy()
                labels = label.numpy()
                i += 1
            else:
                preds = np.concatenate([preds, pred.numpy()], axis=0)
                labels = np.concatenate([labels, label.numpy()], axis=0)
        print(test_loss.result(), test_accuracy.result() * 100)

        return preds, labels, history_loss


    def coeff_InputAttention(self, variable_dict):

        variable_key = list(variable_dict.keys())
        alpha = []
        variables = []
        for i in range(self.n):
            alpha.append(np.mean(self.model.encoder.alpha_t[:, 0, i].numpy()))
            for key in variable_key:
                if f"{i}" in variable_dict[key]:
                    variables.append(f"{key}{i}")
        
        plt.figure(figsize=(6, 4))
        plt.bar(x=variables, height=alpha, color="navy")
        plt.style.use("seaborn-pastel")
        plt.title("alpha")
        plt.xlabel("variables")
        plt.xticks(rotation=90)
        plt.ylabel("prob")
        plt.show()


    def coeff_TemporalAttention(self):
        enc_data, dec_data, label = next(iter(self.test_ds))
        inputs = [enc_data, dec_data]

        pred = self.model(inputs)
        beta = []
        for i in range(self.interval-1):
            beta.append(np.mean(self.model.decoder.beta_t[:, i, 0].numpy()))
        plt.bar(x=range(self.interval-1), height=beta, color="navy")
        plt.style.use("seaborn-pastel")
        plt.title("Beta")
        plt.xlabel("time")
        plt.ylabel("prob")
        plt.show()




def FCNN(x_tr, y_tr, epoch, batch, opt, los, x_val, y_val):

    model = tf.keras.models.Sequential([
        # Shape: (time, features) => (time*features)
        keras.layers.Flatten(input_shape=[30, 16]), 
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(3, activation='linear')
    ])
    
    model.compile(loss=los, optimizer="adam")
    history = model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val))
    return model, history


def LSTM_1_layer(x_tr, y_tr, epoch, batch, los, x_val, y_val):
    model = keras.models.Sequential([
        keras.layers.RNN(keras.layers.LSTMCell(64), input_shape=[None, 16]),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(3, activation='linear')
    ])

    optimizer = keras.optimizers.Adam(lr=0.005)
    model.compile(loss=los, optimizer=optimizer)
    history = model.fit(x_tr, y_tr, batch_size=batch, epochs=epoch, validation_data=(x_val, y_val))

    return model, history


def BidirectionalLSTM(x_tr, y_tr, epoch, batch, opt, los, x_val, y_val):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, ), input_shape=(None, 16)))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(3, activation='linear'))

    model.compile(loss=los, optimizer=opt)
    history = model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val))

    return model, history


def Multilayer_LSTM(x_tr, y_tr, epoch, batch, opt, los, x_val, y_val):
    model = keras.models.Sequential([
        keras.layers.RNN(keras.layers.LSTMCell(64), return_sequences=True, input_shape=[None, 16]),
        keras.layers.RNN(keras.layers.LSTMCell(32), return_sequences=True),
        # keras.layers.RNN(keras.layers.LSTMCell(16)),
        # keras.layers.RNN(keras.layers.LSTMCell(16), return_sequences=True),
        keras.layers.RNN(keras.layers.LSTMCell(12)),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(3, activation='linear')
    ])

    model.compile(loss=los, optimizer=opt)
    history = model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val))

    return model, history


def eval_on_features(index, date, features, target, regressor):
    x_train, x_test = features[:index], features[index:]
    y_train, y_test = target[:index], target[index:]

    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    y_pred_train = regressor.predict(x_train)

    print("Train MAE: {:.2f}".format(mean_absolute_error(y_train, y_pred_train)))
    print("Test MAE: {:.2f}".format(mean_absolute_error(y_test, y_pred)))
    # print("Train RMSE: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    # print("Test RMSE: {:.2f}".format(mean_squared_error(y_test, y_pred))) # squared=True is default

    plt.figure(figsize=(15, 5))
    plt.xticks(date, rotation=90, ha='left')

    plt.plot(range(index), y_train, label="train")
    plt.plot(range(index, len(y_test) + index), y_test, '-', label="test")
    plt.plot(range(index), y_pred_train, '--', label="pred train")
    plt.plot(range(index, len(y_test) + index), y_pred, '--', label="pred test")
    plt.legend(loc=(1.01, 0))

def eval_on_features_returnmodel(index, date, features, target, regressor):
    x_train, x_test = features[:index], features[index:]
    y_train, y_test = target[:index], target[index:]

    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    y_pred_train = regressor.predict(x_train)

    print("Train MAE: {:.2f}".format(mean_absolute_error(y_train, y_pred_train)))
    print("Test MAE: {:.2f}".format(mean_absolute_error(y_test, y_pred)))
    # print("Train RMSE: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    # print("Test RMSE: {:.2f}".format(mean_squared_error(y_test, y_pred))) # squared=True is default

    plt.figure(figsize=(15, 5))
    plt.xticks(date, rotation=90, ha='left')

    plt.plot(range(index), y_train, label="train")
    plt.plot(range(index, len(y_test) + index), y_test, '-', label="test")
    plt.plot(range(index), y_pred_train, '--', label="pred train")
    plt.plot(range(index, len(y_test) + index), y_pred, '--', label="pred test")
    plt.legend(loc=(1.01, 0))

    return regressor


def check_coefficient(rlr, feature_name, threshold):
    ridge_coef = np.array(rlr.coef_).reshape(rlr.coef_.shape[-1])
    ridge_coef_df = pd.DataFrame(list(zip(feature_name, ridge_coef)),
                                 columns=['X_name', 'ridge_coef'])

    high_ridge_coef = ridge_coef_df[ridge_coef_df['ridge_coef'] > threshold]
    high_ridge_coef = high_ridge_coef.sort_values(by=['ridge_coef'], ascending=False)
    print(high_ridge_coef.head())

    plt.figure(figsize=(15, 2))
    plt.bar(range(high_ridge_coef.shape[0]), high_ridge_coef['ridge_coef'], color="r", align="center")
    plt.xticks(range(high_ridge_coef.shape[0]), high_ridge_coef['X_name'], rotation=90)
    plt.show()
