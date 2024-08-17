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

# Create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 5
X, y = create_dataset(scaled_coordinates, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Train-test split
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and tune the model
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', 32, 256, step=32), return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(units=hp.Int('units_2', 32, 256, step=32), return_sequences=True))
    model.add(LSTM(units=hp.Int('units_3', 32, 256, step=32)))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model

tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=2, directory='my_dir', project_name='lstm_tuning')
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), verbose=1)

predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
print("Next predicted numbers:", predictions[-1])
