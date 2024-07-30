#! /bin/bash

# 2D student-t
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 1.0 --niters 2 --T 10.0
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 3.0 --niters 2000 --T 10.0
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 1.0 --niters 2000 --T 10.0 --exclude_OT True
python3 trainToyOTflow.py --data student-t --batch_size 10000 -df 3.0 --niters 2000 --T 10.0 --exclude_OT True

# Keystrokes
python3 trainToyOTflow.py --data Keystrokes --batch_size 7160 --niters 1000 --T 8.0
