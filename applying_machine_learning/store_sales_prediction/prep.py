

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

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


def minmax_scale_y(feature):
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    df_y_scaled = scaler_y.fit_transform(feature).reshape(-1,1)
    df_y_scaled = pd.DataFrame(df_y_scaled, columns=['Sales'])
    return df_y_scaled, scaler_y

def minmax_scale_x(feature):
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    df_x_scaled = scaler_x.fit_transform(np.array(feature))
    df_x_scaled = pd.DataFrame(df_x_scaled, columns=['Customers', 'Open', 'Promo', 'SchoolHoliday',
       'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4',
       'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7', 'StateHoliday_0',
       'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'])
    return df_x_scaled


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


def onehot(data, feature):
    enc = OneHotEncoder()
    X_onehot = enc.fit_transform(data).toarray()
    X_names_onehot = enc.get_feature_names(feature)
    print(X_onehot.shape)
    return X_onehot, X_names_onehot


def poly_transform(data, feature, deg):
    poly_transformer = PolynomialFeatures(degree=deg, interaction_only=True, include_bias=False)
    X_onehot_poly = poly_transformer.fit_transform(data)
    X_names_onehot_poly = poly_transformer.get_feature_names(feature)

    print(X_onehot_poly.shape)
    print(X_names_onehot_poly)

    return X_onehot_poly, X_names_onehot_poly

def add_lag(df, poly, name_poly):
    df['lag1_sales'] = df['Sales'].shift(1)
    df['lag2_sales'] = df['Sales'].shift(2)
    df['lag3_sales'] = df['Sales'].shift(3)
    df['lag4_sales'] = df['Sales'].shift(4)
    df['lag5_sales'] = df['Sales'].shift(5)

    df = df.fillna(0)

    X_lag5 = np.hstack([df['lag1_sales'].values.reshape(-1, 1),
                        df['lag2_sales'].values.reshape(-1, 1),
                        df['lag3_sales'].values.reshape(-1, 1),
                        df['lag4_sales'].values.reshape(-1, 1),
                        df['lag5_sales'].values.reshape(-1, 1)])

    X_onehot_poly_lag5 = np.hstack((X_lag5, poly))
    X_names_onehot_poly_lag5 = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5'] + name_poly

    print(X_onehot_poly_lag5.shape)
    print(X_names_onehot_poly_lag5[:10])

    return X_onehot_poly_lag5, X_names_onehot_poly_lag5





