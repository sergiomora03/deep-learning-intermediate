# src/models.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import config

import optuna
from prettytable import PrettyTable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras as K
import tensorflow as tf
from keras import layers
from keras.preprocessing import timeseries_dataset_from_array
from keras.models import Sequential
from keras import layers

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(1)


initnorm = K.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)

def create_model(trial):
    """
    create_model create RNN with some LSTM layers.

    :param trial: Search space to optim with optuna
    :type trial: optuna.trial object
    :return: RNN model
    :rtype: tensorflow model
    """
    K.backend.clear_session()
    model = Sequential(name = 'Optuna_BRNN')
    n_layers = trial.suggest_int('n_layers', 1, 5)
    for i in range(n_layers):
        num_hidden_one = trial.suggest_int(f"layer_Bidirectional_{i}_n_units", 4, 128, log=True)
        model.add(layers.Bidirectional(layers.LSTM(num_hidden_one, return_sequences=True)))
    num_hidden_two = trial.suggest_int("layer_Bidirectional_n_units", 4, 128, log=True)
    model.add(layers.Bidirectional(layers.LSTM(num_hidden_two)))
    for k in range(n_layers):
        num_hidden = trial.suggest_int(f'layer_Dense_{k}_n_units', 4, 128, log=True)
        activation_selected = trial.suggest_categorical(f"layer_Dense_{k}_activation", ["selu", "sigmoid", "tanh"])
        model.add(layers.Dense(num_hidden, activation=activation_selected, kernel_initializer=initnorm, bias_initializer='zeros'))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def create_optimizer(trial):
    """
    create_optimizer build and choose an optimazer

    :param trial: Search space to optim with optuna
    :type trial: optuna.trial object
    :return: optimizador de tensorflow
    :rtype: tf.optimizers
    """
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["beta_1"] = trial.suggest_loguniform('beta_1', 0.0001, 0.9)
        kwargs["beta_2"] = trial.suggest_loguniform('beta_2', 0.0001, 0.9)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def trainer(trial):

    # Inicializamos la tabla donde guardamos los resultados
    x = PrettyTable(["Exac_E", "Exac_V", "Exac_P", "Optimizer"])

    # Inicializamos el error 
    err_p = 999

    for i in range(0,3,1):
        r = i^3
        size = int(len(temp)*0.9)
        train_Y, test_Y = temp.values.reshape(-1, 1)[0:size], temp.values.reshape(-1, 1)[size:len(temp)]
        epocas = trial.suggest_categorical('epocas', [20, 40])

        scaler = MinMaxScaler()
        scaler.fit(train_Y, y=None)
        scaled_train_data = scaler.transform(train_Y)
        scaled_test_data = scaler.transform(test_Y)
        n_input = train_Y.shape[0]; n_features= 1
        generator = timeseries_dataset_from_array(scaled_train_data,
                                                scaled_train_data,
                                                sequence_length=n_input,
                                                sampling_rate = n_features,
                                                batch_size=1, shuffle=False)


        model = create_model(trial)
        optimizer = create_optimizer(trial)
        model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # Ajustamos el modelo
        model.fit(generator, epochs=epocas, verbose=0, shuffle=False)

        # Rolling forecast
        lstm_predictions_scaled = list()
        batch = scaled_train_data[-n_input:]
        current_batch = batch.reshape((1, n_input, n_features))
        for i in range(len(test_Y)):
            lstm_pred = K.round(model.predict(current_batch)[0])
            lstm_predictions_scaled.append(lstm_pred)
            current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
        lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

        # Gráfica de perdida
        losses_lstm = model.history.history['loss']
        plt.figure(figsize=(20,5))
        plt.xticks(np.arange(0,21,1))
        plt.title(f'Loss Product {Producto}')
        plt.plot(range(len(losses_lstm)),losses_lstm)
        plt.savefig(os.path.join(config.PLOT_LOSS_OUTPUT, f'loss_product_{Producto}.png')); 

        # Gráfica de validación
        RollBack=pd.concat([pd.DataFrame({'TEST':np.concatenate(test_Y)}), pd.DataFrame({'LSTM':np.concatenate(lstm_predictions, axis=0)})], axis=1)
        RollBack = pd.concat([RollBack,pd.DataFrame({'Time':temp.index[size:]})],axis=1)
        RollBack.set_index('Time', inplace=True)
        RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)
        plt.title(f'Forecast Product {Producto}')
        plt.xlabel('time', fontsize=15)
        plt.savefig(os.path.join(config.PLOT_FORECAST_OUTPUT, f'forecast_product_{Producto}.png')); 


        rolling_error_LSTM = mean_squared_error(test_Y, lstm_predictions) ** 0.5
        error_LSTM = mean_squared_error(RollBack.TEST, RollBack.LSTM)
        print(f'    SI rolling forecast -> Test RMSE: {rolling_error_LSTM}')
        print(f'    NO rolling forecast -> Test RMSE: {error_LSTM}')
        print(f'    El modelo evaluado por si mismo da = {model.evaluate(RollBack.TEST, RollBack.LSTM)}')

        Error = Error.append({'Producto': Producto, 'Rolling Forecast RMSE': rolling_error_LSTM, 'RMSE': error_LSTM}, ignore_index=True)
        Error.to_csv(os.path.join(config.ERROR_OUTPUT, 'Errores.csv'))

        # Guardando modelo
        model.save(os.path.join(config.MODEL_OUTPUT, f'lstm__model_{Producto}.h5'))


        #print('Desempeño (exactitud): accu_v1='+str(accu_v) +' , accu_v2='+str(accu_p)  + ' , Optimizer=' + str(optimizer.get_config()["name"]))

        #x.add_row([np.round(accu_e,4), np.round(accu_v,4), np.round(accu_p,4), optimizer.get_config()["name"]])

    #print(x)
    return model, rolling_error_LSTM, error_LSTM

def objective(trial):
    """
    objective define objective function with optuna

    :param trial: Search space to optim with optuna
    :type trial: optuna.trial object
    :return: metric to eval the model
    :rtype: float
    """

    model, rolling_error_LSTM, error_LSTM = trainer(trial)

    evaluate = (rolling_error_LSTM + error_LSTM) / 2

    return evaluate


def run_study(name):
    """
    run_study create several trials to study diferent space to tune hiperparameters

    :param name: name of study
    :type name: str
    """
    study = optuna.create_study(study_name = f'Optuna_BRNN_{name}',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                    n_warmup_steps=30,
                                                                    interval_steps=10))
    study.optimize(objective, n_trials=10, n_jobs = -1, show_progress_bar = False)#, callbacks=[tensorboard_callback])
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    #return trial


def easy_model():
    lstm_model = Sequential()
    lstm_model.add(GRU(200, input_shape=(n_input, n_features)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.summary()