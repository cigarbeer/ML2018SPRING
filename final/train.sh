#!/bin/bash

python3 src/data.py $1 $2
python3 src/train.py 0
python3 src/train.py 1
python3 src/train.py 2
python3 src/train.py 3
python3 src/train.py 4
python3 src/train.py 5
