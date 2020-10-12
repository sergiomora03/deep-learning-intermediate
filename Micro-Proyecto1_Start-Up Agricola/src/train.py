# src/train.py

import os

from numpy import testing
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import config
import argparse
import create_folds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras.backend as K
from keras.preprocessing import timeseries_dataset_from_array
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(1)


def train(Error, test = 1):
    if test == 1:
        data, Fechas = create_folds.datos_agro()
        for Producto in data.Producto.unique(): #data.Producto.unique()[0:test]
            print(f'Corriendo el producto: {Producto}')
            # Filtrando por producto
            temp = data.query("".join(["'", Producto,"'", "== Producto"])).groupby('Fecha').agg(np.sum).Pedido
            temp = temp.reindex(Fechas, fill_value = 0)

            # Creando train y test
            size = int(len(temp)*0.9)
            train_Y, test_Y = temp.values.reshape(-1, 1)[0:size], temp.values.reshape(-1, 1)[size:len(temp)]
            #print(train_Y, test_Y)

            # Realizando transformación MaxMinScaler
            scaler = MinMaxScaler()
            scaler.fit(train_Y, y=None)
            scaled_train_data = scaler.transform(train_Y)
            scaled_test_data = scaler.transform(test_Y)

            # Construyendo el generador
            n_input = train_Y.shape[0]; n_features= 1
            generator = timeseries_dataset_from_array(scaled_train_data, scaled_train_data, sequence_length=n_input, sampling_rate = n_features, batch_size=1, shuffle=False)

            # Creando modelo
            lstm_model = Sequential()
            lstm_model.add(GRU(200, input_shape=(n_input, n_features)))
            #lstm_model.add(LSTM(units=50, return_sequences = True))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            #lstm_model.summary()

            # Entrenando modelo
            lstm_model.fit(generator, epochs=20, verbose = 0)

            # Viendo perdida en el modelo
            losses_lstm = lstm_model.history.history['loss']
            plt.figure(figsize=(20,5))
            plt.xticks(np.arange(0,21,1))
            plt.title(f'Loss Product {Producto}')
            plt.plot(range(len(losses_lstm)),losses_lstm)
            plt.savefig(os.path.join(config.PLOT_LOSS_OUTPUT, f'loss_product_{Producto}.png')); 

            # realizando roling forecast
            lstm_predictions_scaled = list()
            batch = scaled_train_data[-n_input:]
            current_batch = batch.reshape((1, n_input, n_features))

            for i in range(len(test_Y)):
                lstm_pred = K.round(lstm_model.predict(current_batch)[0])
                lstm_predictions_scaled.append(lstm_pred)
                current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

            # Transformando del scaler
            lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
            rolling_error_LSTM = mean_squared_error(test_Y, lstm_predictions) ** 0.5
            print(f'    SI rolling forecast -> Test RMSE: {rolling_error_LSTM}')

            # Grafincando el rollback
            RollBack=pd.concat([pd.DataFrame({'TEST':np.concatenate(test_Y)}), pd.DataFrame({'LSTM':np.concatenate(lstm_predictions, axis=0)})], axis=1)
            # agregando fechas al rollback
            RollBack = pd.concat([RollBack,pd.DataFrame({'Time':temp.index[size:]})],axis=1)

            # Colocando en index
            RollBack.set_index('Time', inplace=True)
            RollBack.head()
            RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)
            plt.title(f'Forecast Product {Producto}')
            plt.xlabel('time', fontsize=15)
            plt.savefig(os.path.join(config.PLOT_FORECAST_OUTPUT, f'forecast_product_{Producto}.png')); 

            # Agregando valores al Error de todos los productos
            error_LSTM = mean_squared_error(RollBack.TEST, RollBack.LSTM)
            print(f'    NO rolling forecast -> Test RMSE: {error_LSTM}')
            Error = Error.append({'Producto': Producto, 'Rolling Forecast RMSE': rolling_error_LSTM, 'RMSE': error_LSTM}, ignore_index=True)
            Error.to_csv(os.path.join(config.ERROR_OUTPUT, 'Errores.csv'))

            # Guardando modelo
            lstm_model.save(os.path.join(config.MODEL_OUTPUT, f'lstm__model_{Producto}.h5'))
            break
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--test",
    type=int
    )
    args = parser.parse_args()

    Error = pd.DataFrame(columns = ['Producto', 'Rolling Forecast RMSE', 'RMSE'])
    train(Error, test = args.test)
