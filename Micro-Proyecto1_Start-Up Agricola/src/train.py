# src/train.py

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

def run(fold):
    data = pd.read_csv("../input/DatosAgro.txt")
    
    