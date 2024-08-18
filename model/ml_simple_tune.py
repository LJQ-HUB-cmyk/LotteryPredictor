# import numpy as np
# from numpy import array
# from numpy import hstack
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# import keras_tuner as kt

# # Data preparation function (as described in your example)
# def split_sequences(sequences, n_steps_in, n_steps_out):
#     X, y = [], []
#     for i in range(len(sequences)):
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out
#         if out_end_ix > len(sequences):
#             break
#         seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

# data = """
# T6, 02/08/2024 21 21 21 21 21 21
# T4, 31/07/2024 20 20 20 20 20 20
# CN, 28/07/2024 19 19 19 19 19 19
# T6, 26/07/2024 18 18 18 18 18 18
# T4, 24/07/2024 17 17 17 17 17 17
# CN, 21/07/2024 16 16 16 16 16 16
# T6, 19/07/2024 15 15 15 15 15 15
# T4, 17/07/2024 14 14 14 14 14 14
# CN, 14/07/2024 13 13 13 13 13 13
# T6, 12/07/2024 12 12 12 12 12 12
# T4, 10/07/2024 11 11 11 11 11 11
# CN, 07/07/2024 10 10 10 10 10 10
# T6, 05/07/2024 09 09 09 09 09 09
# T4, 03/07/2024 08 08 08 08 08 08
# CN, 30/06/2024 07 07 07 07 07 07
# T6, 28/06/2024 06 06 06 06 06 06
# T4, 26/06/2024 05 05 05 05 05 05
# CN, 23/06/2024 04 04 04 04 04 04
# T6, 21/06/2024 03 03 03 03 03 03
# T4, 19/06/2024 02 02 02 02 02 02
# CN, 16/06/2024 01 01 01 01 01 01
# """

# data = """
# T6, 02/08/2024 03 03 03 03 03 03
# T4, 31/07/2024 04 04 04 04 04 04
# CN, 28/07/2024 05 05 05 05 05 05
# T6, 26/07/2024 06 06 06 06 06 06
# T4, 24/07/2024 07 07 07 07 07 07
# CN, 21/07/2024 08 08 08 08 08 08
# T6, 19/07/2024 09 09 09 09 09 09
# T4, 17/07/2024 10 10 10 10 10 10
# CN, 14/07/2024 11 11 11 11 11 11
# T6, 12/07/2024 12 12 12 12 12 12
# T4, 10/07/2024 12 12 12 12 12 12
# CN, 07/07/2024 13 13 13 13 13 13
# T6, 05/07/2024 14 14 14 14 14 14
# T4, 03/07/2024 15 15 15 15 15 15
# CN, 30/06/2024 16 16 16 16 16 16
# T6, 28/06/2024 17 17 17 17 17 17
# T4, 26/06/2024 18 18 18 18 18 18
# CN, 23/06/2024 19 19 19 19 19 19
# T6, 21/06/2024 20 20 20 20 20 20
# T4, 19/06/2024 21 21 21 21 21 21
# CN, 16/06/2024 22 22 22 22 22 22
# """
# # # Read and preprocess data
# # with open('data/lott645.txt', 'r') as file:
# #     data = file.read()
# data = [line.strip().split(', ') for line in data.strip().split('\n')]
# df = pd.DataFrame(data, columns=['Day', 'DateCoordinates'])
# df[['Date', 'Coordinates']] = df['DateCoordinates'].str.extract(r'(\d{2}/\d{2}/\d{4})\s+(.+)')
# df.drop('DateCoordinates', axis=1, inplace=True)
# df['Coordinates'] = df['Coordinates'].apply(lambda x: list(map(int, x.split())))
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df.sort_values('Date', inplace=True)
# coordinates = np.array(df['Coordinates'].tolist())

# # Scaling the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_coordinates = scaler.fit_transform(coordinates)

# class MyTuner(kt.RandomSearch):
#     def run_trial(self, trial, *args, **kwargs):
#         # Get the hp from trial
#         hp = trial.hyperparameters

#         n_steps_in = hp.Int('n_steps_in', min_value=1, max_value=30, step=1)
#         X, y = split_sequences(scaled_coordinates, n_steps_in, n_steps_out=1)
#         n_features = X.shape[2]
#         model = Sequential()
#         model.add(LSTM(units=hp.Int('units', min_value=2, max_value=256, step=2), activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), input_shape=(n_steps_in, n_features)))
#         model.add(RepeatVector(1))
#         model.add(LSTM(units=hp.Int('units', min_value=2, max_value=256, step=2), activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), return_sequences=True))
#         model.add(TimeDistributed(Dense(n_features)))
#         model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']), loss='mse')
#         model.fit(X, y, epochs=hp.Int('epochs', min_value=50, max_value=300, step=10), verbose=0)
#         val_loss = model.evaluate(X, y)
#         return val_loss

# tuner = MyTuner(
#     hypermodel=MyTuner,
#     objective='val_loss',
#     max_trials=10,
#     directory='my_dir',
#     project_name='lstm_tuning'
# )
# # Run the tuner
# tuner.search()

# # Get the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"Best hyperparameters: {best_hps.values}")

# # Convert to dataset with multiple time steps input and output
# n_steps_in, n_steps_out = best_hps.get('n_steps_in'), 1
# X, y = split_sequences(scaled_coordinates, n_steps_in, n_steps_out)
# epochs = best_hps.get('epochs')

# # Define model
# n_features = X.shape[2]
# best_model = Sequential()
# best_model.add(LSTM(units=best_hps.get('units'), activation=best_hps.get('activation'), input_shape=(n_steps_in, n_features)))
# best_model.add(RepeatVector(n_steps_out))
# best_model.add(LSTM(units=best_hps.get('units'), activation=best_hps.get('activation'), return_sequences=True))
# best_model.add(TimeDistributed(Dense(n_features)))
# best_model.compile(optimizer=best_hps.get('optimizer'), loss='mse')

# # Fit the model
# best_model.fit(X, y, epochs=best_hps.get('epochs'), verbose=1)

# # Make a prediction
# x_input = array([coordinates[-n_steps_in:]])  
# x_input = scaler.transform(x_input.reshape(n_steps_in, n_features)).reshape(1, n_steps_in, n_features)
# yhat = best_model.predict(x_input, verbose=0)
# yhat = scaler.inverse_transform(yhat[0])
# print(yhat)

import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import keras_tuner as kt

# Data preparation function
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

# Sample Data
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

# Custom Tuner class
class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        n_steps_in = hp.Int('n_steps_in', min_value=1, max_value=30, step=1)
        n_steps_out = 1
        X, y = split_sequences(scaled_coordinates, n_steps_in, n_steps_out)
        n_features = X.shape[2]
        print(f"X.shape: {X.shape}")

        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=2, max_value=256, step=2), 
                       activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), 
                       input_shape=(n_steps_in, n_features)))
        model.add(RepeatVector(1))
        model.add(LSTM(units=hp.Int('units', min_value=2, max_value=256, step=2), 
                       activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), 
                       return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']), loss='mse')

        model.fit(X, y, epochs=hp.Int('epochs', min_value=50, max_value=300, step=10), verbose=0)
        val_loss = model.evaluate(X, y)
        return val_loss

# Instantiate the custom tuner
tuner = MyTuner(
    max_trials=10,
    overwrite=True,
    directory='my_dir',
    project_name='lstm_tuning'
)

# Run the tuner
tuner.search()

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# Convert to dataset with multiple time steps input and output
n_steps_in, n_steps_out = best_hps.get('n_steps_in'), 1
X, y = split_sequences(scaled_coordinates, n_steps_in, n_steps_out)

# Define model
n_features = X.shape[2]
best_model = Sequential()
best_model.add(LSTM(units=best_hps.get('units'), activation=best_hps.get('activation'), input_shape=(n_steps_in, n_features)))
best_model.add(RepeatVector(n_steps_out))
best_model.add(LSTM(units=best_hps.get('units'), activation=best_hps.get('activation'), return_sequences=True))
best_model.add(TimeDistributed(Dense(n_features)))
best_model.compile(optimizer=best_hps.get('optimizer'), loss='mse')

# Fit the model
best_model.fit(X, y, epochs=best_hps.get('epochs'), verbose=1)

# Make a prediction
x_input = array([coordinates[-n_steps_in:]])  
x_input = scaler.transform(x_input.reshape(n_steps_in, n_features)).reshape(1, n_steps_in, n_features)
yhat = best_model.predict(x_input, verbose=0)
yhat = scaler.inverse_transform(yhat[0])
print(yhat)
