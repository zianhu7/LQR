#!/usr/bin/env bash

ray exec ../ray_autoscale.yaml "python LQR/run_scripts/run_script.py dim3_full_ls_lowcontrol --full_ls 1 --dim 3 \
    --gen_num_exp 0 --num_cpus 36 --use_s3 1 --checkpoint_freq 100 --num_iters 2500 --multi_node 1 --grid_search 0 \
    --num_samples 2 --eigv_high 2.0 --analytic_optimal_cost 0" \
--start --stop --cluster-name fls