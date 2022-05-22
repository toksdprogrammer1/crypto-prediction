
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt

from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
#from statsmodels.tsa.stattools import adfuller
from binance.client import Client
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

# client configuration
api_key = ''
api_secret = ''
interval = '1d'
data_dir = "./data/"

st.title('Decision Support System Cryptocurrency Market')

##################
# Set up sidebar #
##################

# Add in location to select image.

symbol = st.sidebar.selectbox('Select one symbol', ( 'BTCUSDT', 'ETHUSDT'))

def load_and_clean_data_from_api(api_key, api_secret, interval, symbol, data_dir):

    client = Client(api_key, api_secret)

    Client.KLINE_INTERVAL_1DAY
    klines = client.get_historical_klines(symbol, interval, "1 Jan,2018")
    all_data = pd.DataFrame(klines)
    # create colums name
    all_data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                    'taker_base_vol', 'taker_quote_vol', 'ignore']

    all_data.to_csv(data_dir + symbol + '.csv', index=None, header=True)

    return all_data

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

@st.cache(allow_output_mutation=True)
def GRU_model(X_train, y_train, X_test, sc):
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
    my_GRU_model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)


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
         'close': next_predicted_days_value
     }, index=days)
    print(future_preds)

    return future_preds

###################
# Set up main app #
###################

data_load_state = st.text('Loading data and creating model...')

all_data = load_and_clean_data_from_api(api_key,api_secret,interval,symbol,data_dir)

# change the timestamp
all_data.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in all_data.close_time]

# convert data to float and plot
all_data = all_data.astype(float)

print("There are " + str(all_data[:'2021'].shape[0]) + " observations in the training data")
print("There are " + str(all_data['2022':].shape[0]) + " observations in the test data")

# normalize data and diviv]de  into train and test data
X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 10, 1)
X_train.shape[0], X_train.shape[1]

my_GRU_model, GRU_predictions = GRU_model(X_train, y_train, X_test, sc)
print(len(GRU_predictions))

preds  = predict_future_days(all_data, 10, 30, my_GRU_model, sc)

actual_pred = pd.DataFrame(columns=['close', 'prediction'])
actual_pred['close'] = all_data.loc['2022':, 'close'][0:len(GRU_predictions)]
actual_pred['prediction'] = GRU_predictions[:, 0]

m = MeanSquaredError()
m.update_state(np.array(actual_pred['close']), np.array(actual_pred['prediction']))

data_load_state.text("Mean Squared Error: " + str(m.result().numpy()))

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=actual_pred.index, y=actual_pred['close'], mode='lines+markers', name='Actual'), secondary_y=True)
fig.add_trace(go.Scatter(x=actual_pred.index, y=actual_pred['prediction'], mode='lines+markers', name='Prediction'), secondary_y=True)
fig.add_trace(
        go.Scatter(x=preds.index, y=preds['close'], mode='lines+markers', name='Next 30_days prediction'),
        secondary_y=True)
#fig.show()

# Plot!
st.plotly_chart(fig, use_container_width=True)

st.write('Next 30_days prediction ')
st.dataframe(preds)


merged = pd.concat([actual_pred, preds])


indicator_bb = BollingerBands(merged['close'])

bb = merged
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['close','bb_h','bb_l']]

macd = MACD(merged['close']).macd()

rsi = RSIIndicator(merged['close']).rsi()


st.write('Stock Bollinger Bands')

st.line_chart(bb)

progress_bar = st.progress(0)

# https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py

st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)

st.write('Stock RSI ')
st.line_chart(rsi)


st.write('Recent data ')
st.dataframe(all_data.tail(10))


################
# Download csv #
################

import base64
from io import BytesIO

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' # decode b'abc' => abc

st.markdown(get_table_download_link(all_data), unsafe_allow_html=True)


