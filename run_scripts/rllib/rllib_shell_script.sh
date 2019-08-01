#!/usr/bin/env bash

# This script is used to spin up and run the rllib experiments
ray exec ../../ray_autoscale.yaml "python run_scripts/rllib/run_exp_rllib.py dim3_full_ls --full_ls 1 --dim 3 \
    --rand_num_exp 1 --num_cpus 36 --use_s3 1 --checkpoint_freq 1 --num_iters 2500 --grid_search 0" \
    --start --stop --cluster-name 3_fls #--tmux