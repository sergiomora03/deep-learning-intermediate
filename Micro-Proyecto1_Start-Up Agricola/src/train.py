# src/train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU


Error = pd.DataFrame(columns = ['Producto', 'RMSE'])


def train():
    data = pd.read_csv("input/DatosAgro.txt", sep = "\t").drop('Unnamed: 6', axis= 1)
    for Producto in data.Producto.unique():
        # Filtrando por producto
        data.query("".join(["'", Producto,"'", "== Producto"]))
        temp = data.groupby('Fecha').agg(np.sum).Pedido

        # Creando train y test
        size = int(len(temp)*0.9)
        train_Y, test_Y = temp[0:size], temp[size:len(temp)]

        # Realizando transformación MaxMinScaler
        scaler = MinMaxScaler()
        scaler.fit(train_Y, y=None)
        scaled_train_data = scaler.transform(train_Y)
        scaled_test_data = scaler.transform(test_Y)

        # Construyendo el generador
        n_input = 12; n_features= 1
        # TODO revisar el TimeseriesGenerator ingresamos: scaled_test_data
        generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

        # Creando modelo
        lstm_model = Sequential()
        lstm_model.add(GRU(200, input_shape=(n_input, n_features)))
        #lstm_model.add(LSTM(units=50, return_sequences = True))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.summary()

        # Entrenando modelo
        lstm_model.fit_generator(generator,epochs=20)

        # Viendo perdida en el modelo
        losses_lstm = lstm_model.history.history['loss']
        plt.figure(figsize=(20,5))
        plt.xticks(np.arange(0,21,1))
        plt.plot(range(len(losses_lstm)),losses_lstm); 

        # realizando roling forecast
        lstm_predictions_scaled = list()
        batch = scaled_train_data[-n_input:]
        current_batch = batch.reshape((1, n_input, n_features))

        for i in range(len(test_Y)):
            lstm_pred = lstm_model.predict(current_batch)[0]
            lstm_predictions_scaled.append(lstm_pred)
            current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

        # Transformando del scaler
        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
        error_LSTM = mean_squared_error(test_Y, lstm_predictions) ** 0.5
        print(f'Test RMSE: {error_LSTM}')

        # Grafincando el rollback
        RollBack=pd.concat([pd.DataFrame({'TEST':test_Y}),pd.DataFrame({'LSTM':np.concatenate(lstm_predictions, axis=0)})],axis=1)
        RollBack.head()
        RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)
        plt.xlabel('time', fontsize=15); 

        # agregando fechas al rollback
        RollBack = pd.concat([RollBack,pd.DataFrame({'Time':temp.index[size:]})],axis=1)
        RollBack.head()

        # Colocando en index
        RollBack.set_index('Time', inplace=True)
        RollBack.head()
        RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)
        plt.xlabel('time', fontsize=15); 

        # Agregando valores al Error de todos los productos
        Error = Error.append({'Producto': Producto, 'RMSE': error_LSTM}, ignore_index = True)
        print(Error)
        print(f'Last test RMSE LSTM: {mean_squared_error(RollBack.TEST, RollBack.LSTM)}')
        break


if __name__ == "__main__":
    train()
