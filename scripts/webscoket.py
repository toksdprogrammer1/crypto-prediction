from typing import Dict
import websocket
import _thread
import time
import rel
import json
import pickle 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import RobustScaler, MinMaxScaler 


model = pickle.load(open('model.pkl','rb'))

sequence_length = 50

FEATURES = ['ask_price', 'ask_size', 'bid_price', 'bid_size', 'p_asterisk', 'p_ask_delta', 'p_bid_delta']

def get_data():
    df = pd.read_csv('20220812.csv')
    df = df[df['symbol'].str.contains("BTC_USD")]
    df.index = df.time_coinapi

    df = df.drop(columns=['ts', 'time_exchange', 'time_coinapi', 'id', 'symbol', 'sequence'])

    train_df = df.sort_values(by=['time_coinapi']).copy()
    train_df = df.head(5000)
    data = pd.DataFrame(train_df)
    data['p_asterisk'] = ((data['ask_price'] * data['ask_size']) + (data['bid_price'] * data['bid_size']))/ (data['ask_size'] + data['bid_size'])
    #data['p_asterisk_delta'] = (data.p_asterisk - data.p_asterisk.shift(1)).abs()
    #data['p_asterisk_delta'] = data['p_asterisk_delta'].fillna(0)
    #data['p_asterisk_delta2'] = data.p_asterisk_delta / data.p_asterisk
    data['p_ask_delta'] = (data.p_asterisk - data.ask_price).abs()
    data['p_bid_delta'] = (data.p_asterisk - data.bid_price).abs()
    data['Prediction'] = data['p_asterisk']

    data_filtered = data[FEATURES]

    # We add a prediction column and set dummy values to prepare the data for scaling
    return data_filtered.copy()


def get_predicted_price():
    jsonStr = '{"symbol": "COINBASE_SPOT_BTC_USD", "askPrice": 21984.88, "bidSize": 9.101E-5, "bidPrice": 21984.06, "askSize": 0.001}'
    dict = json.loads(jsonStr)
    p = ((dict['askPrice'] * dict['askSize']) + (dict['bidPrice'] * dict['bidSize']))/ (dict['askSize'] + dict['bidSize'])
    p_ask_delta = abs(p - dict['askPrice'])
    p_bid_delta = abs(p - dict['bidPrice'])
    latest_quote = np.array([[dict['askPrice'], dict['askSize'], dict['bidPrice'], dict['bidSize'], p, p_ask_delta, p_bid_delta]])

    data_filtered_ext = get_data()
    df_temp = data_filtered_ext[-49:]

    N = sequence_length

    # Get the last N day closing price values and scale the data to be values between 0 and 1
    last_N_days = df_temp[-sequence_length:].values
    last_N_days = np.concatenate((last_N_days, latest_quote), axis=0)

    scaler = MinMaxScaler()
    last_N_days_scaled = scaler.transform(last_N_days)

    # Create an empty list and Append past N days
    X_test_new = []
    X_test_new.append(last_N_days_scaled)

    # Convert the X_test data set to a numpy array and reshape the data
    pred_price_scaled = model.predict(np.array(X_test_new))
    scaler_pred = MinMaxScaler()
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

    # Print last price and predicted price for the next day
    #price_today = np.round(new_df['ask_price'][-1], 2)
    predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)

    print(f'The predicted  price is {predicted_price}')


def on_message(ws, message):
    print('--'*20)
    print(message)
    print('--'*20)
    get_predicted_price()
    # data = json.loads(message)

    # result = model.predict(data)




    # if data.get('symbol') == "COINBASE_SPOT_BTC_USD":
    #     print(data)
    #     response_payload = {
    #         "action": "publish_prediction",
    #         "payload": {
    #             "symbol_id": data.get('id'),
    #             "price": 22123.44,
    #             "side": "ASK"
    #         }
    #     }
    #     # print(response_payload)
    #     ws.send(json.dumps(response_payload))

def on_error(ws, error):
    print("Error: ", error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    ws.send('{"action":"subscribe_symbol", "key":"COINBASE_SPOT_BTC_USD"}')

if __name__ == "__main__":
    server = 'wss://atz3h3bzji.execute-api.eu-west-2.amazonaws.com/development?token='
    token = 'eyJraWQiOiIwYWZWTXhMMmJkVVd1eXJsbUZxVFBwS2FGVms3K0p1aVNINTBDNDJPVVo0PSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI4NGZhYWU3MC0xNDFhLTQ2MmQtOTQ3Zi1lMGIwYTRhNGIzOTgiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0yLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMl9XdTF3c3RHdTciLCJjbGllbnRfaWQiOiI2aXAzNTJqcWFzam5tbnY4MGYzdmJxaTd1dCIsIm9yaWdpbl9qdGkiOiI3ZWY5NWZmZC04MDEyLTQxOWEtODM3OC0zY2QwODdiMjhjZTMiLCJldmVudF9pZCI6IjMyNzc1MTYzLWM4MTAtNDU1ZS1hNGNjLTQ3NGNhMjNhNDA5MyIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE2NjM2MTM0NDksImV4cCI6MTY2MzY5OTg0OSwiaWF0IjoxNjYzNjEzNDQ5LCJqdGkiOiI0YTI4MTEzYS00MjE2LTQ5MDYtYmY1ZC1jYWUwNTRkMTkyM2UiLCJ1c2VybmFtZSI6InNjb3R0LmRvdWdsYXMifQ.kACa8flrndy23ZoKfnxs_nVLkqo_H7jbtshjbjJSQYzIO82uGqigVzxzEs9dv36ybLFH12tC4tsa9-2PShxIamH1ZfJAWNJm757Z2PeW_T38UWAyB0mj8n8tGfQyqx7eYZrFZE21CC6W9kKIc_vZFh5hf3Wk8GaVWUrXrnIIivUFtXhZh6lIPcSsOmYT0R3cQbBi3r0W70V-6XbHDFsQrHHzqtzZQTcL5rdtLxGhbdK9HU4x-vr1XTo3wnWoaiS7akq7RR6yZ0DGaFYtGbx7SBciB_PGhOI22aKXNypGX6DTzWBBMGjpTUneXq8yh382C6BR1_I0kg9X5nbD1niCRA'

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(server + token,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()


# ws.send("{
#   "action": "publish_prediction",
#   "payload": {
#     "symbol_id": "FUNKYGUID3838383",
#     "price": 22123.44,
#     "side": "ASK"
#   }
# }")



# {
#     "symbol":"COINBASE_SPOT_BTC_USD",
#     "askPrice":20280.46,
#     "bidSize":0.001,
#     "id":"ef419f28-73b4-4f3d-a84c-42957e6f7203",
#     "bidPrice":20276.94,
#     "askSize":0.00101245
# }