
#For connection to the websocket stream
import json
jsonStr = '{"symbol": "COINBASE_SPOT_BTC_USD", "askPrice": 20984.88, "bidSize": 9.101E-5, "bidPrice": 20984.06, "askSize": 0.001}'
dict = json.loads(jsonStr)
p = ((dict['askPrice'] * dict['askSize']) + (dict['bidPrice'] * dict['bidSize']))/ (dict['askSize'] + dict['bidSize'])
p_ask_delta = abs(p - dict['askPrice'])
p_bid_delta = abs(p - dict['bidPrice'])
latest_quote = np.array([[dict['askPrice'], dict['askSize'], dict['bidPrice'], dict['bidSize'], p, p_ask_delta, p_bid_delta]])


df_temp = data_filtered_ext[-49:]

N = sequence_length

# Get the last N day closing price values and scale the data to be values between 0 and 1
last_N_days = df_temp[-sequence_length:].values
last_N_days = np.concatenate((last_N_days, latest_quote), axis=0)

last_N_days_scaled = scaler.transform(last_N_days)

# Create an empty list and Append past N days
X_test_new = []
X_test_new.append(last_N_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

# Print last price and predicted price for the next day
#price_today = np.round(new_df['ask_price'][-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)

print(f'The predicted  price is {predicted_price}')


