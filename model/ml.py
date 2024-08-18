import numpy as np
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Data preparation function (as described in your example)
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

data = """
T6, 02/08/2024 21 21 21 21 21 21
T4, 31/07/2024 20 20 20 20 20 20
CN, 28/07/2024 19 19 19 19 19 19
T6, 26/07/2024 18 18 18 18 18 18
T4, 24/07/2024 17 17 17 17 17 17
CN, 21/07/2024 16 16 16 16 16 16
T6, 19/07/2024 15 15 15 15 15 15
T4, 17/07/2024 14 14 14 14 14 14
CN, 14/07/2024 13 13 13 13 13 13
T6, 12/07/2024 12 12 12 12 12 12
T4, 10/07/2024 11 11 11 11 11 11
CN, 07/07/2024 10 10 10 10 10 10
T6, 05/07/2024 09 09 09 09 09 09
T4, 03/07/2024 08 08 08 08 08 08
CN, 30/06/2024 07 07 07 07 07 07
T6, 28/06/2024 06 06 06 06 06 06
T4, 26/06/2024 05 05 05 05 05 05
CN, 23/06/2024 04 04 04 04 04 04
T6, 21/06/2024 03 03 03 03 03 03
T4, 19/06/2024 02 02 02 02 02 02
CN, 16/06/2024 01 01 01 01 01 01
"""

data = """
T6, 02/08/2024 03 03 03 03 03 03
T4, 31/07/2024 04 04 04 04 04 04
CN, 28/07/2024 05 05 05 05 05 05
T6, 26/07/2024 06 06 06 06 06 06
T4, 24/07/2024 07 07 07 07 07 07
CN, 21/07/2024 08 08 08 08 08 08
T6, 19/07/2024 09 09 09 09 09 09
T4, 17/07/2024 10 10 10 10 10 10
CN, 14/07/2024 11 11 11 11 11 11
T6, 12/07/2024 12 12 12 12 12 12
T4, 10/07/2024 12 12 12 12 12 12
CN, 07/07/2024 13 13 13 13 13 13
T6, 05/07/2024 14 14 14 14 14 14
T4, 03/07/2024 15 15 15 15 15 15
CN, 30/06/2024 16 16 16 16 16 16
T6, 28/06/2024 17 17 17 17 17 17
T4, 26/06/2024 18 18 18 18 18 18
CN, 23/06/2024 19 19 19 19 19 19
T6, 21/06/2024 20 20 20 20 20 20
T4, 19/06/2024 21 21 21 21 21 21
CN, 16/06/2024 22 22 22 22 22 22
"""
# Read and preprocess data
with open('data/lott645.txt', 'r') as file:
    data = file.read()
data = [line.strip().split(', ') for line in data.strip().split('\n')]
df = pd.DataFrame(data, columns=['Day', 'DateCoordinates'])
df[['Date', 'Coordinates']] = df['DateCoordinates'].str.extract(r'(\d{2}/\d{2}/\d{4})\s+(.+)')
df.drop('DateCoordinates', axis=1, inplace=True)
df['Coordinates'] = df['Coordinates'].apply(lambda x: list(map(int, x.split())))
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.sort_values('Date', inplace=True)
coordinates = np.array(df['Coordinates'].tolist())

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_coordinates = scaler.fit_transform(coordinates)

# Convert to dataset with multiple time steps input and output
n_steps_in, n_steps_out = 3, 1
X, y = split_sequences(scaled_coordinates, n_steps_in, n_steps_out)
print(f"===X and y===")
print(X)
print(y)
# Define model
n_features = X.shape[2]
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X, y, epochs=300, verbose=1)

# Make a prediction
x_input = array([coordinates[-n_steps_in:]])  
x_input = scaler.transform(x_input.reshape(n_steps_in, n_features)).reshape(1, n_steps_in, n_features)
yhat = model.predict(x_input, verbose=0)
yhat = scaler.inverse_transform(yhat[0])
print(yhat)
