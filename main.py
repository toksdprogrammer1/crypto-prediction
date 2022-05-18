from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
#from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
# Generate the data
import numpy as np
import os
import pandas as pd
from binance.client import Client
import datetime as dt


def main():
    # client configuration
    api_key = ''
    api_secret = ''

    #symbol = "BTCUSDT"
    symbol = "ETHUSDT"
    interval = '1d'
    Client.KLINE_INTERVAL_1DAY
    data_dir = "./data/"
    #print(os.listdir(data_dir))
    all_data = read_and_clean_data_from_csv(symbol,data_dir)
    #all_data = read_and_clean_data_from_api(api_key,api_secret,interval,symbol,data_dir)

    # change the timestamp
    all_data.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in all_data.close_time]

    # convert data to float and plot
    all_data = all_data.astype(float)
    print(all_data.head())
    print("There are " + str(all_data[:'2021'].shape[0]) + " observations in the training data")
    print("There are " + str(all_data['2022':].shape[0]) + " observations in the test data")
    #plt.plot(all_data['close'])
    #plt.savefig('btcusdt.png')


    fig = go.Figure(data=go.Scatter(x=all_data.index, y=all_data['close'], mode='lines+markers'))
    fig.show()

    X_train, y_train, X_test = ts_train_test(all_data, 5, 2)
    X_train.shape[0], X_train.shape[1]

    # Convert the 3-D shape of X_train to a data frame so we can see:
    X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
    y_train_see = pd.DataFrame(y_train)
    train_market_date = pd.concat([X_train_see, y_train_see], axis=1)
    print("Train market data")
    print(train_market_date.head())

    # Convert the 3-D shape of X_test to a data frame so we can see:
    X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
    test_market_date = pd.DataFrame(X_test_see)
    print("Test market data")
    print(test_market_date.head())

    print("There are " + str(X_train.shape[0]) + " samples in the training data")
    print("There are " + str(X_test.shape[0]) + " samples in the test data")

def read_and_clean_data_from_csv(symbol, data_dir):
    # print(os.listdir(data_dir))
    all_data = pd.read_csv(data_dir + symbol + '.csv')

    # change the timestamp
    all_data.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in all_data.close_time]

    return all_data


def read_and_clean_data_from_api(api_key, api_secret, interval, symbol, data_dir):

    client = Client(api_key, api_secret)

    Client.KLINE_INTERVAL_1DAY
    klines = client.get_historical_klines(symbol, interval, "1 Jan,2018")
    all_data = pd.DataFrame(klines)
    # create colums name
    all_data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                    'taker_base_vol', 'taker_quote_vol', 'ignore']

    all_data.to_csv(data_dir + symbol + '.csv', index=None, header=True)

    return all_data

def ts_train_test(all_data, time_steps, for_periods):
    '''
    input:
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2018/1/1-2021/12/31
      X_test:  data from 2022 -
    time_steps: # of the input time steps
    for_periods: # of the output time steps
    '''
    # create training and test set
    ts_train = all_data[:'2021'].iloc[:, 4:5].values
    ts_test = all_data['2022':].iloc[:, 4:5].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train[i - time_steps:i, 0])
        y_train.append(ts_train[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Preparing to create X_test
    inputs = pd.concat((all_data["close"][:'2021'], all_data["close"]['2022':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)

    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test

if __name__ == "__main__":
    main()