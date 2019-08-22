#!/usr/bin/env bash

python graph_rollout.py --out test --checkpoint /Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-21-2019/dim3_full_ls_lowcontrol/dim3_full_ls_lowcontrol/PPO_GenLQREnv-v0_1_2019-08-21_05-33-54bmrot4c6/checkpoint_2400/checkpoint-2400 \
--eval_matrix 0 --eigv_gen 1 --opnorm_error 1 --high 13 --gaussian_actions 0 --write_mode w --steps 10000

#python graph_rollout.py --out test --checkpoint /Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-21-2019/dim3_full_ls_fullcontrol/dim3_full_ls_fullcontrol/PPO_GenLQREnv-v0_0_2019-08-21_05-40-398r5kk1ug/checkpoint_2400/checkpoint-2400 \
#--eval_matrix 0 --eigv_gen 1 --opnorm_error 1 --high 20 --gaussian_actions 1 --write_mode w

#python graph_rollout.py --out test --checkpoint ../trained_policies/full_constrained_R3/checkpoint-2400 \
#--eval_matrix 0 --eigv_gen 1 --opnorm_error 1 --high 13 --gaussian_actions 0 --write_mode w --steps 10000
