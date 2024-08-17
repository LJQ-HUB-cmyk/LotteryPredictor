# import keras_tuner as kt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Read and preprocess data
# with open('data/lott645.txt', 'r') as file:
#     data = file.read()
# # data = """
# # T6, 02/08/2024 09 14 19 34 37 41
# # T4, 31/07/2024 02 04 12 16 18 42
# # CN, 28/07/2024 01 13 21 25 31 39
# # T6, 26/07/2024 18 20 28 31 39 45
# # T4, 24/07/2024 07 08 20 27 32 34
# # CN, 21/07/2024 09 10 23 25 28 38
# # T6, 19/07/2024 01 10 20 25 34 35
# # T4, 17/07/2024 09 10 11 16 21 30
# # CN, 14/07/2024 09 18 33 37 38 43
# # T6, 12/07/2024 11 17 25 26 28 29
# # T4, 10/07/2024 23 24 36 37 40 45
# # CN, 07/07/2024 04 08 22 23 26 45
# # T6, 05/07/2024 04 23 33 38 40 44
# # T4, 03/07/2024 11 18 24 34 38 43
# # CN, 30/06/2024 05 23 25 28 30 43
# # T6, 28/06/2024 04 06 16 32 41 44
# # T4, 26/06/2024 08 10 29 30 33 40
# # CN, 23/06/2024 09 11 19 29 31 44
# # T6, 21/06/2024 03 07 11 16 19 35
# # T4, 19/06/2024 08 12 17 23 26 27
# # CN, 16/06/2024 03 16 17 18 25 37
# # """
# data = [line.strip().split(', ') for line in data.strip().split('\n')]
# df = pd.DataFrame(data, columns=['Day', 'DateCoordinates'])
# df[['Date', 'Coordinates']] = df['DateCoordinates'].str.extract(r'(\d{2}/\d{2}/\d{4})\s+(.+)')
# df.drop('DateCoordinates', axis=1, inplace=True)
# df['Coordinates'] = df['Coordinates'].apply(lambda x: list(map(int, x.split())))
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df.sort_values('Date', inplace=True)
# coordinates = np.array(df['Coordinates'].tolist())

# # Scaling
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_coordinates = scaler.fit_transform(coordinates)

# # Create dataset
# def create_dataset(data, time_step=1):
#     X, y = [], []
#     for i in range(len(data) - time_step):
#         X.append(data[i:(i + time_step)])
#         y.append(data[i + time_step])
#     return np.array(X), np.array(y)

# time_step = 10
# X, y = create_dataset(scaled_coordinates, time_step)
# X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# # Train-test split
# train_size = int(len(X) * 0.7)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Build and tune the model
# def build_model(hp):
#     model = Sequential()
#     model.add(LSTM(units=hp.Int('units_1', 2, 256, step=2), return_sequences=True, input_shape=(time_step, X.shape[2])))
#     model.add(LSTM(units=hp.Int('units_3', 2, 256, step=2)))
#     model.add(Dense(X.shape[2], activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid'])))
#     model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
#     return model

# tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=2, directory='my_dir', project_name='lstm_tuning')
# tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16)

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"Best hyperparameters: {best_hps.values}")
# best_model = tuner.hypermodel.build(best_hps)
# best_model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# predictions = best_model.predict(X_test)
# predictions = scaler.inverse_transform(predictions)
# print("Next predicted numbers:", predictions[-1])
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read and preprocess data
with open('data/lott645.txt', 'r') as file:
    data = file.read()
# data = """
# T6, 02/08/2024 09 14 19 34 37 41
# T4, 31/07/2024 02 04 12 16 18 42
# CN, 28/07/2024 01 13 21 25 31 39
# T6, 26/07/2024 18 20 28 31 39 45
# T4, 24/07/2024 07 08 20 27 32 34
# CN, 21/07/2024 09 10 23 25 28 38
# T6, 19/07/2024 01 10 20 25 34 35
# T4, 17/07/2024 09 10 11 16 21 30
# CN, 14/07/2024 09 18 33 37 38 43
# T6, 12/07/2024 11 17 25 26 28 29
# T4, 10/07/2024 23 24 36 37 40 45
# CN, 07/07/2024 04 08 22 23 26 45
# T6, 05/07/2024 04 23 33 38 40 44
# T4, 03/07/2024 11 18 24 34 38 43
# CN, 30/06/2024 05 23 25 28 30 43
# T6, 28/06/2024 04 06 16 32 41 44
# T4, 26/06/2024 08 10 29 30 33 40
# CN, 23/06/2024 09 11 19 29 31 44
# T6, 21/06/2024 03 07 11 16 19 35
# T4, 19/06/2024 08 12 17 23 26 27
# CN, 16/06/2024 03 16 17 18 25 37
# """
data = [line.strip().split(', ') for line in data.strip().split('\n')]
df = pd.DataFrame(data, columns=['Day', 'DateCoordinates'])
df[['Date', 'Coordinates']] = df['DateCoordinates'].str.extract(r'(\d{2}/\d{2}/\d{4})\s+(.+)')
df.drop('DateCoordinates', axis=1, inplace=True)
df['Coordinates'] = df['Coordinates'].apply(lambda x: list(map(int, x.split())))
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.sort_values('Date', inplace=True)
coordinates = np.array(df['Coordinates'].tolist())

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_coordinates = scaler.fit_transform(coordinates)

# Create dataset function
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

# Custom Tuner class
class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial
        hp = trial.hyperparameters

        # Define "time_step" as a hyperparameter
        time_step = hp.Int('time_step', min_value=2, max_value=20, step=2)
        
        # Create dataset with the tunable time_step
        X, y = create_dataset(scaled_coordinates, time_step)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
        train_factor = hp.Float('train_factor', min_value=0.5, max_value=0.9, step=0.1)

        # Train-test split
        train_size = int(len(X) * train_factor)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build the model
        model = Sequential()
        model.add(LSTM(units=hp.Int('units_1', 2, 256, step=2), return_sequences=True, input_shape=(time_step, X.shape[2])))
        model.add(LSTM(units=hp.Int('units_2', 2, 256, step=2)))
        model.add(Dense(X.shape[2], activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid', 'softmax'])))
        model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])), loss='mean_squared_error')

        # Train the model
        model.fit(
            X_train, y_train,
            epochs=hp.Int('epochs', min_value=50, max_value=200, step=50),
            batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
            validation_data=(X_test, y_test),
            verbose=0  # Set to 1 to see training progress
        )
        
        # Evaluate the model
        val_loss = model.evaluate(X_test, y_test)
        
        # Return the objective value to minimize (validation loss)
        return val_loss

# Instantiate the custom tuner
tuner = MyTuner(
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="lstm_tuning"
)

# Run the tuner
tuner.search()

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")
time_step = best_hps.get('time_step')
X, y = create_dataset(scaled_coordinates, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Train-test split
train_factor = best_hps.get('train_factor')
train_size = int(len(X) * train_factor)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the best model using the best hyperparameters
best_model = Sequential()
best_model.add(LSTM(units=best_hps.get('units_1'), return_sequences=True, input_shape=(time_step, X.shape[2])))
best_model.add(LSTM(units=best_hps.get('units_2')))
best_model.add(Dense(X.shape[2], activation=best_hps.get('activation')))
best_model.compile(optimizer=Adam(best_hps.get('learning_rate')), loss='mean_squared_error')

# Fit the best model
best_model.fit(
    X_train, y_train,
    epochs=best_hps.get('epochs'),
    batch_size=best_hps.get('batch_size'),
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
print("Next predicted numbers:", predictions[-1])
