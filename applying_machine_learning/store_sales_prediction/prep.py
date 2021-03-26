

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

def make_dataset(sales_data):
    maxday = 30   # x 값으로 볼 time series 길이
    maxtarget = 3  # y 값으로 볼 time series 길이
    dataset = []

    for i in range(maxday, len(sales_data) - maxtarget+1):
        print('making dataset progress : {}/{}'.format(i, len(sales_data)), end='\r')

        historySet = sales_data.Sales.loc[i - maxday:i - 1]
        targetSet = sales_data.Sales.loc[i:i + maxtarget - 1]
        
        CustomersSet = sales_data.Customers.loc[i - maxday:i - 1]
        OpenSet = sales_data.Open.loc[i - maxday:i - 1]
        PromoSet = sales_data.Promo.loc[i - maxday:i - 1]
        SchoolHolidaySet = sales_data.SchoolHoliday.loc[i - maxday:i - 1]
        DayOfWeek_1Set = sales_data.DayOfWeek_1.loc[i - maxday:i - 1]
        DayOfWeek_2Set = sales_data.DayOfWeek_2.loc[i - maxday:i - 1]
        DayOfWeek_3Set = sales_data.DayOfWeek_3.loc[i - maxday:i - 1]
        DayOfWeek_4Set = sales_data.DayOfWeek_4.loc[i - maxday:i - 1]
        DayOfWeek_5Set = sales_data.DayOfWeek_5.loc[i - maxday:i - 1]
        DayOfWeek_6Set = sales_data.DayOfWeek_6.loc[i - maxday:i - 1]
        DayOfWeek_7Set = sales_data.DayOfWeek_7.loc[i - maxday:i - 1]     
        StateHoliday_0Set = sales_data.StateHoliday_0.loc[i - maxday:i - 1]
        StateHoliday_aSet = sales_data.StateHoliday_a.loc[i - maxday:i - 1]
        StateHoliday_bSet = sales_data.StateHoliday_b.loc[i - maxday:i - 1]
        StateHoliday_cSet = sales_data.StateHoliday_c.loc[i - maxday:i - 1]
        
        target_history = np.reshape(np.array(historySet), (maxday, 1))
        target_sales = np.reshape(np.array(targetSet), (maxtarget, 1))
        CustomersSet_history = np.reshape(np.array(CustomersSet), (maxday, 1))
        OpenSet_history = np.reshape(np.array(OpenSet), (maxday, 1))
        PromoSet_history = np.reshape(np.array(PromoSet), (maxday, 1))
        SchoolHolidaySet_history = np.reshape(np.array(SchoolHolidaySet), (maxday, 1))
        DayOfWeek_1Set_history = np.reshape(np.array(DayOfWeek_1Set), (maxday, 1))
        DayOfWeek_2Set_history = np.reshape(np.array(DayOfWeek_2Set), (maxday, 1))
        DayOfWeek_3Set_history = np.reshape(np.array(DayOfWeek_3Set), (maxday, 1))
        DayOfWeek_4Set_history = np.reshape(np.array(DayOfWeek_4Set), (maxday, 1))
        DayOfWeek_5Set_history = np.reshape(np.array(DayOfWeek_5Set), (maxday, 1))
        DayOfWeek_6Set_history = np.reshape(np.array(DayOfWeek_6Set), (maxday, 1))
        DayOfWeek_7Set_history = np.reshape(np.array(DayOfWeek_7Set), (maxday, 1))
        StateHoliday_0Set_history = np.reshape(np.array(StateHoliday_0Set), (maxday, 1))
        StateHoliday_aSet_history = np.reshape(np.array(StateHoliday_aSet), (maxday, 1))
        StateHoliday_bSet_history = np.reshape(np.array(StateHoliday_bSet), (maxday, 1))
        StateHoliday_cSet_history = np.reshape(np.array(StateHoliday_cSet), (maxday, 1))
                
        dataset.append(
            {'target_history': target_history,
                 'CustomersSet_history': CustomersSet_history,
                 'OpenSet_history': OpenSet_history,
             'PromoSet_history': PromoSet_history,
            'SchoolHolidaySet_history': SchoolHolidaySet_history,
            'DayOfWeek_1Set_history': DayOfWeek_1Set_history,
            'DayOfWeek_2Set_history': DayOfWeek_2Set_history,
            'DayOfWeek_3Set_history': DayOfWeek_3Set_history,
            'DayOfWeek_4Set_history': DayOfWeek_4Set_history,
            'DayOfWeek_5Set_history': DayOfWeek_5Set_history,
            'DayOfWeek_6Set_history': DayOfWeek_6Set_history,
            'DayOfWeek_7Set_history': DayOfWeek_7Set_history,
            'StateHoliday_0Set_history': StateHoliday_0Set_history,
            'StateHoliday_aSet_history': StateHoliday_aSet_history,
            'StateHoliday_bSet_history': StateHoliday_bSet_history,
            'StateHoliday_cSet_history': StateHoliday_cSet_history,
                 'target_sales': target_sales,
            })
    return dataset



def DataBatch(dataset):
    maxday = 30   # x 값으로 볼 time series 길이
    maxtarget = 3  # y 값으로 볼 time series 길이
    
    y = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    x10 = []
    x11 = []
    x12 = []
    x13 = []
    x14 = []
    x15 = []
    target = []
    
    for d in dataset:
        y.append(d['target_history'])
        x1.append(d['CustomersSet_history'])
        x2.append(d['OpenSet_history'])
        x3.append(d['PromoSet_history'])
        x4.append(d['SchoolHolidaySet_history'])
        x5.append(d['DayOfWeek_1Set_history'])
        x6.append(d['DayOfWeek_2Set_history'])
        x7.append(d['DayOfWeek_3Set_history'])
        x8.append(d['DayOfWeek_4Set_history'])
        x9.append(d['DayOfWeek_5Set_history'])
        x10.append(d['DayOfWeek_6Set_history'])
        x11.append(d['DayOfWeek_7Set_history'])
        x12.append(d['StateHoliday_0Set_history'])
        x13.append(d['StateHoliday_aSet_history'])
        x14.append(d['StateHoliday_bSet_history'])
        x15.append(d['StateHoliday_cSet_history'])
        target.append(d['target_sales'])
    
    y = np.reshape(y, (-1, maxday, 1))  # (# of batch_size(window에 따라 데이터 뽑았을 때 데이터 갯수), time_size, # of feature(=1))
    x1 = np.reshape(x1, (-1, maxday, 1))
    x2 = np.reshape(x2, (-1, maxday, 1))
    x3 = np.reshape(x3, (-1, maxday, 1))
    x4 = np.reshape(x4, (-1, maxday, 1))
    x5 = np.reshape(x5, (-1, maxday, 1))
    x6 = np.reshape(x6, (-1, maxday, 1))
    x7 = np.reshape(x7, (-1, maxday, 1))
    x8 = np.reshape(x8, (-1, maxday, 1))
    x9 = np.reshape(x9, (-1, maxday, 1))
    x10 = np.reshape(x10, (-1, maxday, 1))
    x11 = np.reshape(x11, (-1, maxday, 1))
    x12 = np.reshape(x12, (-1, maxday, 1))
    x13 = np.reshape(x13, (-1, maxday, 1))
    x14 = np.reshape(x14, (-1, maxday, 1))
    x15 = np.reshape(x15, (-1, maxday, 1))
    target = np.reshape(target, (-1, maxtarget, 1))
    
    x_trainFinal = np.reshape(np.stack((y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15), axis=2), (-1, maxday, 16))
    y_trainFinal = target
   
    return x_trainFinal, y_trainFinal