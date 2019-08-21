#!/usr/bin/env bash

python run_scripts/rllib/GenLQR/run_script.py dim3_full_ls --full_ls 1 --dim 3 \
    --rand_num_exp 1 --num_cpus 1 --use_s3 0 --checkpoint_freq 100 --num_iters 1 --multi_node 0 --grid_search 0 \
    --num_samples 2 --eigv_high 2.0