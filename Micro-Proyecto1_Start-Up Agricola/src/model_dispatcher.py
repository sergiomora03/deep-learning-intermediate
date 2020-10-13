# src/model_dispatcher.py

import models
import argparse
import create_folds


def reader_trial(mod = 'all', study = False):
    if study:
        if mod == 'all':
            data, _ = create_folds.datos_agro()
            for Producto in data.Producto.unique():
                print(f'Leyendo arquitectura del Producto: {Producto}')
                models.read_study(Producto)
        else:
            models.read_study(mod)
    else:
        if mod == 'all':
            data, _ = create_folds.datos_agro()
            for Producto in data.Producto.unique():
                print(f'Leyendo arquitectura del Producto: {Producto}')
                models.read_trial(Producto)
        else:
            models.read_trial(mod)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--producto", type=str)
    parser.add_argument("--study", type=bool)
    args = parser.parse_args()
    reader_trial(mod = args.producto, study = args.study)