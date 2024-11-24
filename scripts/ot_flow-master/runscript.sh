#! /bin/bash

# 2D student-t
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 1.0 --niters 2 --T 10.0
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 3.0 --niters 2000 --T 10.0
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 1.0 --niters 2000 --T 10.0 --exclude_OT True
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 3.0 --niters 2000 --T 10.0 --exclude_OT True

# Keystrokes
python3 trainToyOTflow.py --data Keystrokes --batch_size 7160 --niters 1000 --T 8.0
python3 trainToyOTflow.py --data Keystrokes --batch_size 7160 --niters 1000 --T 8.0 --exclude_OT True

# Heavytail submanifold
python3 trainToyOTflow.py --data Heavytail_submanifold --batch_size 200 --niters 1000 --T 5.0
python3 trainToyOTflow.py --data Heavytail_submanifold --batch_size 200 --niters 1000 --T 5.0 --exclude_OT True

# Lorenz63
python3 trainToyOTflow.py --data Lorenz63 --batch_size 5000 --niters 1000 --T 5.0 --m 64
python3 trainToyOTflow.py --data Lorenz63 --batch_size 5000 --niters 1000 --T 5.0 --exclude_OT True --m 64
