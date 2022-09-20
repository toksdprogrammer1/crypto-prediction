import websocket
import _thread
import time
import rel


def on_message(ws, message):
    print(message)
    print('')

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    ws.send('{"action":"subscribe_symbol", "key":"COINBASE_SPOT_BTC_USD"}')

if __name__ == "__main__":
    server = 'wss://atz3h3bzji.execute-api.eu-west-2.amazonaws.com/development?token='
    token = 'eyJraWQiOiIwYWZWTXhMMmJkVVd1eXJsbUZxVFBwS2FGVms3K0p1aVNINTBDNDJPVVo0PSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI4NGZhYWU3MC0xNDFhLTQ2MmQtOTQ3Zi1lMGIwYTRhNGIzOTgiLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0yLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMl9XdTF3c3RHdTciLCJjbGllbnRfaWQiOiI2aXAzNTJqcWFzam5tbnY4MGYzdmJxaTd1dCIsIm9yaWdpbl9qdGkiOiI3ZWY5NWZmZC04MDEyLTQxOWEtODM3OC0zY2QwODdiMjhjZTMiLCJldmVudF9pZCI6IjMyNzc1MTYzLWM4MTAtNDU1ZS1hNGNjLTQ3NGNhMjNhNDA5MyIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE2NjM2MTM0NDksImV4cCI6MTY2MzY5OTg0OSwiaWF0IjoxNjYzNjEzNDQ5LCJqdGkiOiI0YTI4MTEzYS00MjE2LTQ5MDYtYmY1ZC1jYWUwNTRkMTkyM2UiLCJ1c2VybmFtZSI6InNjb3R0LmRvdWdsYXMifQ.kACa8flrndy23ZoKfnxs_nVLkqo_H7jbtshjbjJSQYzIO82uGqigVzxzEs9dv36ybLFH12tC4tsa9-2PShxIamH1ZfJAWNJm757Z2PeW_T38UWAyB0mj8n8tGfQyqx7eYZrFZE21CC6W9kKIc_vZFh5hf3Wk8GaVWUrXrnIIivUFtXhZh6lIPcSsOmYT0R3cQbBi3r0W70V-6XbHDFsQrHHzqtzZQTcL5rdtLxGhbdK9HU4x-vr1XTo3wnWoaiS7akq7RR6yZ0DGaFYtGbx7SBciB_PGhOI22aKXNypGX6DTzWBBMGjpTUneXq8yh382C6BR1_I0kg9X5nbD1niCRA'

    websocket.enableTrace(True)
    #ws = websocket.WebSocketApp("wss://api.gemini.com/v1/marketdata/BTCUSD",
    ws = websocket.WebSocketApp(server + token,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)


    ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()




# sec-websocket-accept: LM2XPPKbd5WXM523ovwH1fKXbxc=
# -----------------------
# Opened connection
# ++Sent raw: b'\x81\xbcF\xf1\x12\xcf=\xd3s\xac2\x98}\xa1d\xcb0\xbc3\x93a\xac4\x98p\xaa\x19\x82k\xa2$\x9e~\xedj\xd10\xa4#\x880\xf5d\xb2]\x86\x08\xb3S\x9c\x03\xaeA\x9f\t\xa5M\x8d\x12\xb2M\x9a\x15\xb50\xb2'
# ++Sent decoded: fin=1 opcode=1 data=b'{"action":"subscribe_symbol", "key":"COINBASE_SPOT_BTC_USD"}'
# websocket connected
# ++Rcv raw: b'\x81\x18Successfully subscribed!'
# ++Rcv decoded: fin=1 opcode=1 data=b'Successfully subscribed!'
# Successfully subscribed!

# ++Rcv raw: b'\x88\x0c\x03\xe9Going away'
# ++Rcv decoded: fin=1 opcode=8 data=b'\x03\xe9Going away'
# ++Sent raw: b'\x88\x82~>\x8c\x12}\xd6'
# ++Sent decoded: fin=1 opcode=8 data=b'\x03\xe8'
# ### closed ###
# ^Cjk@jk-ThinkPa