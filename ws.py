import json
import rel
import websocket
import requests


def on_message(ws, message):
    
    if "Successfully subscribed!" in message:
        return

    # print("=" * 80)
    global anotherdf, FEATURES, scalers, predictions, real_prices, timeranges, errors

    # index   = timeranges[0]
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
    print("======"*30)
    print(message)
    

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