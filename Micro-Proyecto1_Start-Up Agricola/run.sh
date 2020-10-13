#!/bin/sh
python src/train.py --test 1
python src/model_dispatcher.py --producto all --study True
