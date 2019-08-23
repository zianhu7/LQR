#!/usr/bin/env bash

# This file is used to run the Fiechter experiments so I don't have to retype it every time
#python rollout_fiechter.py --out test --opnorm_error 1 --eval_matrix 1
#python rollout_fiechter.py --out full_ls --full_ls 1 --opnorm_error 1 --eigv_gen 1 --append 0
#python rollout_fiechter.py --out partial --full_ls 0 --opnorm_error 1 --eigv_gen 1 --append 0
python rollout_fiechter.py --out full_ls --full_ls 1 --opnorm_error 1 --eval_matrix 1 --append 0 --steps 100000
#python rollout_fiechter.py --out partial --full_ls 0 --opnorm_error 1 --eval_matrix 1 --append 0