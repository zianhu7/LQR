#!/usr/bin/env bash

# This script is used to spin up and run the rllib experiments
ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/RegretLQR/run_exp_rllib.py dim3_full_ls_regret --dim 3 \
    --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 2500 --multi_node 1 --grid_search 1" \
    --start --stop --cluster-name 3_reg --tmux
ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/RegretLQR/run_exp_rllib.py dim5_full_ls_regret --dim 5 \
    --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 3000 --multi_node 1 --grid_search 1" \
    --start --stop --cluster-name 5_reg --tmux
