import websocket
import _thread
import time
import rel
import json



def on_message(ws, message):
    print('--'*20)
    print(message)
    print('--'*20)
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