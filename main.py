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
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM
#from keras.optimizers import gradient_descent_v2
from keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from keras.layers import Dropout

from plotly.subplots import make_subplots

from datetime import datetime, timedelta
from numpy import array

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
    print(all_data.tail())
    print("There are " + str(all_data[:'2021'].shape[0]) + " observations in the training data")
    print("There are " + str(all_data['2022':].shape[0]) + " observations in the test data")
    #plt.plot(all_data['close'])
    #plt.savefig('btcusdt.png')


    #fig = go.Figure(data=go.Scatter(x=all_data.index, y=all_data['close'], mode='lines+markers'))
    #fig.show()

    X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 5, 1)
    X_train.shape[0], X_train.shape[1]

    # Convert the 3-D shape of X_train to a data frame so we can see:
    # X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
    # y_train_see = pd.DataFrame(y_train)
    # train_market_date = pd.concat([X_train_see, y_train_see], axis=1)
    # print("Train market data")
    # print(train_market_date.tail())
    #
    # Convert the 3-D shape of X_test to a data frame so we can see:
    # X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
    # test_market_date = pd.DataFrame(X_test_see)
    # print("Test market data")
    # print(test_market_date.tail())
    #
    # print("There are " + str(X_train.shape[0]) + " samples in the training data")
    #print("There are " + str(X_test.shape[0]) + " samples in the test data")



    #print("LSTM Prediction")
    #my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, sc)
    #print(LSTM_prediction[1:10])
    #print("Mean Squared Error")
    #print(actual_pred_plot(all_data, LSTM_prediction))

    #print("GRU Prediction")
    #my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)
    #print(GRU_prediction[1:10])
    #actual_pred_plot(all_data, GRU_prediction)

    print("GRU Prediction Regularization")
    my_GRU_model, GRU_predictions = GRU_model_regularization(X_train, y_train, X_test, sc)
    print(len(GRU_predictions))


    #print(X_test)
    future_preds  = predict_future_days(all_data, 5, 30, my_GRU_model, sc)
    print("Mean Squared Error")
    print(actual_pred_plot(all_data, GRU_predictions, future_preds))


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
    print(X_test)
    return X_train, y_train, X_test


def ts_train_test_normalize(all_data, time_steps, for_periods):
    '''
    input:
      data: dataframe with dates and price data
    output:
      X_train, y_train: data from 2018/1/1-2021/12/31
      X_test:  data from 2022 -
      sc:      insantiated MinMaxScaler object fit to the training data
    '''
    # create training and test set
    ts_train = all_data[:'2021'].iloc[:, 4:5].values
    ts_test = all_data['2022':].iloc[:, 4:5].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data

    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["close"][:'2021'], all_data["close"]['2022':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    #print(inputs)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])


    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc


def actual_pred_plot(all_data, preds, future_preds):
    '''
    Plot the actual vs.prediction
    '''

    actual_pred = pd.DataFrame(columns=['close', 'prediction'])
    actual_pred['close'] = all_data.loc['2022':, 'close'][0:len(preds)]
    actual_pred['prediction'] = preds[:, 0]
    #actual_pred['Next_30_days_prediction'] = future_preds['Predicted Close']

    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['close']), np.array(actual_pred['prediction']))



    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=actual_pred.index, y=actual_pred['close'], mode='lines+markers', name='Actual'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=actual_pred.index, y=actual_pred['prediction'], mode='lines+markers', name='Prediction'), secondary_y=True)
    fig2.add_trace(
        go.Scatter(x=future_preds.index, y=future_preds['Predicted_Close'], mode='lines+markers', name='Next 30_days prediction'),
        secondary_y=True)
    fig2.show()

    # fig = go.Figure(data=go.Scatter(x=all_data.index, y=all_data['close'], mode='lines+markers'))
    # fig.show()

    return m.result().numpy()

def LSTM_model(X_train, y_train, X_test, sc):
    # create a model

    #SGD = gradient_descent_v2.SGD(...)
    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    # my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    # my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    # Compiling
    my_LSTM_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    # Fitting to the training set
    my_LSTM_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)

    return my_LSTM_model, LSTM_prediction


def GRU_model(X_train, y_train, X_test, sc):
    # create a model

    # The GRU architecture
    my_GRU_model = Sequential()
    # First GRU layer with Dropout regularisation
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    # my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    # my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dense(units=2))

    # Compiling the RNN
    my_GRU_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    # Fitting to the training set
    my_GRU_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)

    return my_GRU_model, GRU_prediction


def GRU_model_regularization(X_train, y_train, X_test, sc):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model


    # The GRU architecture
    my_GRU_model = Sequential()
    # First GRU layer with Dropout regularisation
    my_GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Second GRU layer
    my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))

    # Third GRU layer
    my_GRU_model.add(GRU(units=50, return_sequences=True, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # Fourth GRU layer
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    # The output layer
    my_GRU_model.add(Dense(units=1))
    # Compiling the RNN
    my_GRU_model.compile(loss="mean_squared_error",optimizer="adam")
    # Fitting to the training set
    my_GRU_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)


    GRU_predictions = my_GRU_model.predict(X_test)
    GRU_predictions = sc.inverse_transform(GRU_predictions)

    return my_GRU_model, GRU_predictions

def predict_future_days(all_data, time_steps,  pred_days, model, sc):

    test_data = all_data['2022':].iloc[:, 4:5].values
    x_input = test_data[len(test_data) - time_steps:].reshape(1, -1)
    print("Before transformation Input data for prediction")
    print(x_input)
    input = x_input.reshape(-1, 1)
    input = sc.transform(input)
    print("Input data for prediction")
    print(input)
    # Preparing X_test
    X_test = []
    for i in range(0, len(input)):
        X_test.append(input[i, 0])
    print(X_test)
    #temp_input = list(x_input)
    #print("temp input {} ".format(temp_input))
    #temp_input = temp_input[0].tolist()
    temp_input = X_test
    print("input {} ".format(temp_input))
    x_input = input
    #
    lst_output = []
    n_steps = time_steps
    i = 0
    #pred_days = 30
    while (i < pred_days):

        if (len(temp_input) > time_steps):

            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1, -1)
            #x_input = sc.transform(x_input)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            print(temp_input)

            lst_output.extend(yhat.tolist())
            i = i + 1

        else:

            x_input = x_input.reshape((1, n_steps, 1))
            print("{} day input {}".format(i, x_input))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i + 1
    temp_mat = np.empty(pred_days + 1)
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1, -1).tolist()[0]
    next_predicted_days_value = temp_mat
    next_predicted_days_value = sc.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]
    print(next_predicted_days_value)
    print(all_data["close_time"].iloc[-1])
    last_date = all_data["close_time"].iloc[-1]
    dt_object = datetime.fromtimestamp(last_date/1000)
    print(dt_object)
    days = pd.date_range(dt_object, dt_object + timedelta(pred_days - 1), freq='D')
    print(days)
    future_preds = pd.DataFrame({
         'Predicted_Close': next_predicted_days_value
     }, index=days)
    print(future_preds)
    # names = cycle(['Predicted next 60 days close price'])
    # fig = px.line(new_pred_plot, x=new_pred_plot.index, y=new_pred_plot['Predicted Close'],
    #               labels={'value': 'Stock price', 'index': 'Timestamp'})
    # fig.update_layout(title_text='Next 60 days Closing price prediction',
    #                   plot_bgcolor='white', legend_title_text='Close Price')
    # fig.for_each_trace(lambda t: t.update(name=next(names)))
    # fig.update_xaxes(showgrid=False)
    # fig.update_yaxes(showgrid=False)
    # fig.show()
    return future_preds


if __name__ == "__main__":
    main()