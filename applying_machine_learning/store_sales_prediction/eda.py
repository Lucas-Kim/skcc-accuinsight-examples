
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
import matplotlib.pyplot as plt

def decompose(df, feature, method, fequency):
    components = seasonal_decompose(df[feature], model=method, freq=fequency)
    ts = (df.Sales.to_frame('Original')
          .assign(Trend=components.trend)
          .assign(Seasonality=components.seasonal)
          .assign(Residual=components.resid))
    _ = ts.plot(subplots=True, figsize=(14, 8))



def rolling_mean(df, feature, win):
    rolling_mean = df[feature].rolling(window=win).mean()
    rolling_std = df[feature].rolling(window=win).std()

    plt.figure(figsize=(15, 5))
    plt.plot(df['Sales'], color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.show()

