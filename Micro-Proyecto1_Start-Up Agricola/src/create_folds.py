# src/create_folds.py

import config
import pandas as pd

def datos_agro():
    data = pd.read_csv(config.TRAINING_FILE, sep = "\t").drop('Unnamed: 6', axis= 1)
    data.Fecha = pd.to_datetime(data.Fecha)
    Fechas = pd.date_range(min(data.Fecha), max(data.Fecha))
    return data, Fechas
