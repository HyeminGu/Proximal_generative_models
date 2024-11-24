#! /bin/bash

# 2D student-t
python3 GPA_NN.py --dataset Learning_student_t -nu 1.0 --N_dim 2 --f alpha -alpha 2 --formulation LT --epochs 10000 --lr_P 5.0 --N_samples_P 10000 --N_samples_Q 10000 -L 1 --epochs_phi 3 --exp_no 1
python3 GPA_NN.py --dataset Learning_student_t -nu 1.0 --N_dim 2 --f alpha -alpha 2 --formulation LT --epochs 10000 --lr_P 5.0 --N_samples_P 10000 --N_samples_Q 10000 --epochs_phi 3

# Keystrokes
python3 GPA_NN.py --dataset Keystrokes --f alpha -alpha 2.0 --formulation LT -L 1.0 --lr_P 1000.0 --exp_no 3 --epochs 15000
python3 GPA_NN.py --dataset Keystrokes --f alpha -alpha 2.0 --formulation LT --lr_P 1000.0 --exp_no 3 --epochs 15000

# Heavytail submanifold
python3 GPA_NN.py --dataset Heavytail_submanifold alpha -alpha 2.0 --formulation LT -L 1.0 --lr_P 100.0 --exp_no 4 --epochs 20000
python3 GPA_NN.py --dataset Heavytail_submanifold alpha -alpha 2.0 --formulation LT --lr_P 100.0 --exp_no 4 --epochs 20000

# Lorenz63
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 5000 --N_samples_P 10000 --f alpha -alpha 2.0 --formulation LT -L 1.0 --lr_P 1.0 --exp_no 3 --epochs 4000 --save_iter 50
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 5000 --N_samples_P 10000 --f alpha -alpha 2.0 --formulation LT  --lr_P 1.0 --exp_no 3 --epochs 4000 --save_iter 50
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 5000 --N_samples_P 10000 --f KL --formulation DV -L 1.0 --lr_P 1.0 --exp_no 3 --epochs 4000 --save_iter 50
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 5000 --N_samples_P 10000 --f KL --formulation DV  --lr_P 1.0 --exp_no 3 --epochs 350 --save_iter 10

python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 1000 --N_samples_P 2000 --f alpha -alpha 2.0 --formulation LT -L 1.0 --lr_P 0.5 --exp_no 3 --epochs 4000 --save_iter 50
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 1000 --N_samples_P 2000 --f alpha -alpha 2.0 --formulation LT  --lr_P 0.5 --exp_no 3 --epochs 400 --save_iter 10
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 1000 --N_samples_P 2000 --f KL --formulation DV -L 1.0 --lr_P 0.5 --exp_no 3 --epochs 4000 --save_iter 50
python3 GPA_NN.py --dataset Lorenz63 --N_samples_Q 1000 --N_samples_P 2000 --f KL --formulation DV  --lr_P 0.5 --exp_no 3 --epochs 400 --save_iter 10
