

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


def minmax_scale_y(feature_tr, feature_ts):
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    tr_y_scaled = scaler_y.fit_transform(feature_tr).reshape(-1, 1)
    ts_y_scaled = scaler_y.transform(feature_ts).reshape(-1, 1)

    tr_y_scaled = pd.DataFrame(tr_y_scaled, columns=feature_tr.columns)
    ts_y_scaled = pd.DataFrame(ts_y_scaled, columns=feature_ts.columns)
    return tr_y_scaled, ts_y_scaled, scaler_y

def minmax_scale_x(feature_tr, feature_ts):
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    tr_x_scaled = scaler_x.fit_transform(np.array(feature_tr))
    ts_x_scaled = scaler_x.transform(np.array(feature_ts))

    tr_x_scaled = pd.DataFrame(tr_x_scaled, columns=feature_tr.columns)
    ts_x_scaled = pd.DataFrame(ts_x_scaled, columns=feature_ts.columns)
    return tr_x_scaled, ts_x_scaled

def create_dataset(data, look_back, look_after, y_feature, x_feature):
    x_arr, y_arr = [], []
    for i in range(len(data) - (look_back + look_after) + 1):
        x_arr.append(data.loc[i:(i + look_back - 1), x_feature])
        y_arr.append(data.loc[(i + look_back):(i + look_back + look_after - 1), y_feature])

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    print('X shape : ' + str(x_arr.shape))
    print('Y shape : ' + str(y_arr.shape))

    return x_arr, y_arr

class Preprocess_DARNN():
    def __init__(self, train_data, test_data, interval):
        self.Y_train = train_data.iloc[:, 0]
        self.X_train = train_data.iloc[:, 1:]
        self.Y_test = test_data.iloc[:, 0]
        self.X_test = test_data.iloc[:, 1:]
        self.interval = interval
        self.encoder_sequence_tr, self.decoder_sequence_tr, self.target_tr = \
            self.Create_dataset(self.X_train, self.Y_train)
        self.encoder_sequence_ts, self.decoder_sequence_ts, self.target_ts = \
            self.Create_dataset(self.X_test, self.Y_test)


    def Create_dataset(self, X_df, Y_df):
        encoder_list = []
        decoder_list = []
        target_list = []

        for i in range(1, X_df.shape[0] - self.interval):
            encoder_list.append(np.array(X_df.iloc[i: i + self.interval, :]))
            decoder_list.append(np.array(Y_df.iloc[i: i + self.interval - 1]))
            target_list.append(Y_df.iloc[i + self.interval - 1])

        encoder_sequence = np.array(encoder_list)
        decoder_sequence = np.array(decoder_list)

        decoder_sequence = np.reshape(decoder_sequence, (-1, self.interval - 1, 1))
        target = np.array(target_list)

        return encoder_sequence, decoder_sequence, target

    def Show_shape(self, option):
        if option == 'train':
            print('Shape of encoder input : {}'.format(self.encoder_sequence_tr.shape))
            print('Shape of decoder input : {}'.format(self.decoder_sequence_tr.shape))
            print('Shape of target input : {}'.format(self.target_tr.shape))
        else:
            print('Shape of encoder input : {}'.format(self.encoder_sequence_ts.shape))
            print('Shape of decoder input : {}'.format(self.decoder_sequence_ts.shape))
            print('Shape of target input : {}'.format(self.target_ts.shape))

