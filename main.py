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
    token = 'eyJraWQiOiIwYWZWTXhMMmJkVVd1eXJsbUZxVFBwS2FGVms3K0p1aVNINTBDNDJPVVo0PSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI4NGZhYWU3MC0xNDFhLTQ2MmQtOTQ3Zi1lMGIwYTRhNGIzOTgiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0yLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMl9XdTF3c3RHdTciLCJjbGllbnRfaWQiOiI2aXAzNTJqcWFzam5tbnY4MGYzdmJxaTd1dCIsIm9yaWdpbl9qdGkiOiI0ZWMwZDlhOC0xMGMxLTQ5N2MtOTM5Yy1mYTVmNzYwOGE4MTUiLCJldmVudF9pZCI6ImI1NzUzMDYxLWYzYTctNDdkMS1iMGYyLTkwYjZhMDQ3ZTZlOSIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE2NjM0MjUyMDMsImV4cCI6MTY2MzUxMTYwMywiaWF0IjoxNjYzNDI1MjAzLCJqdGkiOiIxYTdlNjU3Ny05YzZkLTQ1NWUtYmM5OC1jOWNkMTQ0ODE3ZjMiLCJ1c2VybmFtZSI6InNjb3R0LmRvdWdsYXMifQ.Aa_UzXqqR_rLK8h05JeU0NyhvJAJ8JbtNKB-bw8Ixnx3gcUJVVFAEegVDo7hYiJ3l6xN05_kL0Wysb6J9xaNBqefgV9HHikH6U7zMJC6ngqgMeqdnJ4tNKnKEDYMNFuCUzCIIBJQV_o_zm2qz_wud3ruwqlK_LLuSUmIaCwICBU_WLwL4faMn-ANVm_00-LTAh79NRoGnI6R98FgBo9nK7kNO9OM_-ZETihVmjJwNZLDw44Lpvb8--ZOnyhk1Yke0JW87LEIXWJ6ZWRYhWkcw3T3KTPmaDvQOzrPuu4oxO-4yn1sOCN-I_poNs-ALGHKUghNrob6P65DB5fWmW-bug'

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
#   "action": "publish_prediction",
#   "payload": {
#     "symbol_id": "FUNKYGUID3838383",
#     "price": 22123.44,
#     "side": "ASK"
#   }
# }")



# {
#     "symbol":"COINBASE_SPOT_BTC_USD",
#     "askPrice":20280.46,
#     "bidSize":0.001,
#     "id":"ef419f28-73b4-4f3d-a84c-42957e6f7203",
#     "bidPrice":20276.94,
#     "askSize":0.00101245
# }