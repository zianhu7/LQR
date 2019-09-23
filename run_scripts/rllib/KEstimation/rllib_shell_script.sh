#!/usr/bin/env bash

# This script is used to spin up and run the rllib experiments

ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/KEstimation/k_scripty.py dim1_stability_additive --dim 1 \
    --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 2500 --multi_node 1 --grid_search 1 --num_samples 2\
    --eigv_high 2.0 --eigv_low 0.5 --stability_scaling 30 --elem_sample 1 --horizon 120 --exp_length 6" \
    --start --stop --cluster-name 100_fls --tmux
#ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/GenLQR/run_exp_rllib.py dim3_full_ls_500cond --full_ls 1 --dim 3 \
#--rand_num_exp 1 --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 2500 --multi_node 1 --grid_search 1 --num_samples 2\
#--done_norm_cond 500.0" \
#--start --stop --cluster-name 500_fls
#ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/GenLQR/run_exp_rllib.py dim3_partial_ls --full_ls 0 --dim 3 \
#    --rand_num_exp 1 --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 2500 --multi_node 1 --grid_search 1 --num_samples 2" \
#    --start --stop --cluster-name 3_pls --tmux
#ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/GenLQR/run_exp_rllib.py dim5_full_ls --full_ls 1 --dim 5 \
#    --rand_num_exp 1 --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 3000 --multi_node 1 --grid_search 1" \
#    --start --stop --cluster-name 5_fls --tmux
#ray exec ../../../ray_autoscale.yaml "python LQR/run_scripts/rllib/GenLQR/run_exp_rllib.py dim5_partial_ls --full_ls 0 --dim 5 \
#    --rand_num_exp 1 --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 3000 --multi_node 1 --grid_search 1 --num_samples 2" \
#    --start --stop --cluster-name 5_pls --tmux
