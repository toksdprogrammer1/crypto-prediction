import pickle
import pandas as pd # Additional functions for analysing and manipulating data
import websocket
import requests
import rel
import json
import math # Mathematical functions
import numpy as np # Fundamental package for scientific computing with Python
import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, MinMaxScaler 

models = pickle.load(open(f'model0.pkl','rb'))

anotherdf   = pd.DataFrame()
real_prices = []
predictions = []
errors      = [0]
correction  = 0
corrected_errors = [0]
average_corrected_errors = [0]
average_N_errors = [0]
n_average = 5
sequence_length = 60
timeranges = ['4.0S']
FEATURES = ['av_ask', 'av_bid', 'p_zoomed', 'da_zoomed', 'db_zoomed']


def get_data(datasource, timerange):
    datasource = datasource.drop(columns=['time_coinapi', 'time_exchange', 'id', 'symbol', 'sequence'])
    datasource['pressure_ask'] = datasource['ask_price'] * datasource ['ask_size']
    datasource['pressure_bid'] = datasource['bid_price'] * datasource ['bid_size']
    datasource['ts'] = pd.to_datetime(datasource['ts'])
    datasource = datasource.set_index('ts')
    datasource.sort_values(by=['ts'])
    datasource = datasource.resample(timerange).sum()
    datasource['av_ask'] = datasource['pressure_ask'] / datasource['ask_size']
    datasource['av_bid'] = datasource['pressure_bid'] / datasource['bid_size']
    datasource['p_zoomed'] = (datasource['pressure_ask'] + datasource['pressure_bid'] ) / (datasource['ask_size'] + datasource['bid_size'])
    datasource['da_zoomed'] = abs(datasource['p_zoomed'] - datasource['av_ask'])
    datasource['db_zoomed'] = abs(datasource['p_zoomed'] - datasource['av_bid'])
    datasource.fillna(method="ffill", inplace=True)
    datasource.fillna(method="bfill", inplace=True)
    datasource.fillna(0, inplace=True)

    filtered = datasource[FEATURES]

    # Convert the data to numpy values
    np_data_unscaled = np.array(filtered)

    # Transform the data by scaling each feature to a range between 0 and 1
    scaler      = MinMaxScaler()
    scaler_pred = MinMaxScaler()
    scaler_da   = MinMaxScaler()
    scaler_db   = MinMaxScaler()

    p = pd.DataFrame(filtered['p_zoomed'])
    da = pd.DataFrame(filtered['da_zoomed'])
    db = pd.DataFrame(filtered['db_zoomed'])

    np_data_scaled = scaler.fit_transform(np_data_unscaled)
    scaler_pred.fit_transform(p)
    scaler_da.fit_transform(da)
    scaler_db.fit_transform(db)

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data
    # train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

    # Create the training and test data
    # tr_data = np_data_scaled[0:train_data_len, :]
    # ts_data = np_data_scaled[train_data_len - sequence_len:, :]

    scalers = [scaler, scaler_pred, scaler_da, scaler_db]

    return datasource, scalers


def on_message(ws, message):
    if  ("Successfully subscribed!" in message) \
            or ("Received Prediction!" in message) \
            or ("QUOTE_COINBASE_SPOT_BTC_USD" not in message):
        return

    global anotherdf, FEATURES, scalers, predictions, real_prices, timeranges, errors, n_average, average_N_errors, average_corrected_errors, corrected_errors

    index   = timeranges[0]
    temp    = json.loads(message)
    data    = {
            'ask_price': temp.get('askPrice'),
            'ask_size': temp.get('askSize'),
            'bid_price': temp.get('bidPrice'),
            'bid_size': temp.get('bidSize'),
            'time_coinapi': temp.get('timeCoinApi'),
            'time_exchange': temp.get('timeExchange'),
            'sequence': temp.get('sequence'),
            'symbol': temp.get('symbolId'),
            'id': temp.get('id'),
            'ts': temp.get('timeCoinApi')
        }

    try:

        anotherdf = pd.concat([anotherdf, pd.DataFrame.from_records([data])])

        prediction_base, scalers = get_data(anotherdf, index)
        prediction_base = prediction_base.tail(sequence_length)
        prediction_base = prediction_base[FEATURES]
        prediction_base_scaled = scalers[0].transform(prediction_base.values)

        if prediction_base.shape[0] >= sequence_length:
            prediction = models.predict(np.array(prediction_base_scaled.reshape(-1, sequence_length, len(FEATURES))),verbose = 0)
            pred_p = scalers[1].inverse_transform(prediction[:,0].reshape(-1,1))[0][0]

            p = pred_p
            errors.append(p - data.get('ask_price'))

            average_N_errors.append(np.average(errors[-n_average:]))
            if len(errors) > 2:
                p = p - average_N_errors[-1]

            print(f"point: {anotherdf.shape[0]}::  \task_price: {data.get('ask_price')} ::  \tpredicted price: {p} (${p - data.get('ask_price')} or {np.round(100 - (data.get('ask_price') * 100)/p, 6)}%)")

            predictions.append(float(p))
            real_prices.append(data.get('ask_price'))
            corrected_errors.append(abs(np.round(100 - (data.get('ask_price') * 100)/p, 6)))
            average_corrected_errors.append(np.average(corrected_errors))

            if data.get('symbol') == "QUOTE_COINBASE_SPOT_BTC_USD":
                response_payload = {
                    "action": "publish_predictions",
                    "payload": [{
                        "symbol_id": str(data.get('id')),
                        "price": float(p),
                        "side": "TEST"
                    }]
                }
                # print(response_payload)
                response = json.dumps(response_payload)
                ws.send(f'{response}')
            else:
                return
        else:
            print(f"{anotherdf.shape[0]} point ask_price:  \t{data.get('ask_price')}   \t:: {prediction_base.shape[0]}/{sequence_length}... accumulating data")
    except Exception as e:
        raise e

def on_error(ws, error):
    print("Error: ", error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")
    ws.send('{"action":"subscribe_symbol", "key":"QUOTE_COINBASE_SPOT_BTC_USD"}')


if __name__ == "__main__":
    server = 'wss://atz3h3bzji.execute-api.eu-west-2.amazonaws.com/development?token='
    authUrl = 'https://maz5ef1wy1.execute-api.eu-west-2.amazonaws.com/development/auth'
    headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = json.dumps({'username': 'ai.websocket', 'password': 'Nintendo321*'})
    res = requests.post(authUrl, headers=headers, data=data)
    tokenResponse = res.json()
    token = tokenResponse['AuthenticationResult']['AccessToken']

    websocket.enableTrace(False)
    #ws = websocket.WebSocketApp("wss://api.gemini.com/v1/marketdata/BTCUSD",
    ws = websocket.WebSocketApp(server + token,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()