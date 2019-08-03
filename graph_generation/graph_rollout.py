'''Used to replay a saved checkpoint'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle

import gym
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class

from envs import GenLQREnv
import graph_generation.plot_suboptimality as ps
from utils.rllib_utils import merge_dicts

# Example Usage via RLlib CLI:
"""
Generates plots from a checkpoint given the GenLQREnv:
    1. '--eval_matrix' System benchmark as a function of rollout length 
       (mutually exclusive from other graphs, need to separately run)
    2. '--eigv_gen' Eigenvalue generalization: Relative LQR Cost Suboptimality wrt 
       top eigenvalue of A, Stability of policy wrt top eigenvalue of A; requires --high for
       eigenvalue bound sampling
    3. '--opnorm_error' Generates plot of ||A - A_est||_2 as a function of rollout 
       length OR top eigenvalue depending on which of 1. or 2. flags it is run with

REQUIRED:
    1. '--full_ls' Based on whether trained policy has learned full or partial ls sampling
    2. If '--eigv_gen' is True, must specify '--rand_num_exp' to fix rollout length for replay
"""


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
                    "given a checkpoint.")

    # TODO(@evinitsky) move this over to
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint directory from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        default='PPO',
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.")
    required_named.add_argument("--env", type=str, default="GenLQREnv", help="The gym environment to use.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--high", type=float, nargs='+', default=1,
                        help="upper bound for eigenvalue initialization")
    parser.add_argument("--eval_matrix", type=int, default=0,
                        help="Whether to benchmark on `Sample complexity of quadratic "
                             "regulator` system for R3")
    parser.add_argument("--eigv_gen", type=int, default=0,
                        help="Eigenvalue generalization tests for eigenvalues of A. Sample matrices randomly and "
                             "see how we perform relative to the top eigenvalue")
    parser.add_argument("--opnorm_error", type=int, default=0,
                        help="Operator norm error of (A-A_est)")
    parser.add_argument("--full_ls", type=int, default=1,
                        help="Sampling type")
    parser.add_argument("--es", type=int, default=1, help="Element sampling")
    parser.add_argument("--rand_num_exp", action='store_true', help="If passed, the total number of experiments is "
                                                                    "sampled uniformly from 2 * dim to "
                                                                    "(horizon / exp_length). Otherwise the number of "
                                                                    "exps is (horizon / exp_length)")

    parser.add_argument("--gaussian_actions", action="store_true", help="Run env with "
                                                                        "standard normal actions")
    parser.add_argument("--create_all_graphs", action="store_true", help="Create all the"
                                                                         " graphs for the paper")
    parser.add_argument("--clear_graphs", action="store_true", help="Remove graphs from output folder")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
             "Supresses loading of configuration from checkpoint.")
    return parser


def create_env_params(args):
    env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
                  "eigv_high": args.high, "elem_sample": args.es, "eval_matrix": args.eval_matrix,
                  "full_ls": args.full_ls, "rand_num_exp": args.rand_num_exp,
                  "gaussian_actions": args.gaussian_actions, "dim": 3}
    return env_params


def run(args, parser, env_params):

    config = args.config
    if not config:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            if not args.config:
                raise ValueError(
                    "Could not find params.pkl in either the checkpoint dir or "
                    "its parent directory.")
        else:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])

    # convert to max cpus available on system
    config['num_workers'] = 1
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    cls = get_agent_class(args.run)

    # pull in the params from training time and overwrite. If statement for backwards compatibility
    if len(config["env_config"]) > 0:
        base_params = config["env_config"]["env_params"]
        env_params = merge_dicts(base_params, env_params)

    config["env_config"] = env_params
    agent = cls(env=GenLQREnv, config=config)
    agent.restore(os.path.join(args.checkpoint, os.path.basename(args.checkpoint).replace('_', '-')))
    num_steps = int(args.steps)

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = GenLQREnv(env_params)
    if args.out is not None:
        rollouts = []
    steps = 0
    action_norm_list = []
    while steps < (num_steps or steps + 1):
        if args.out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            action = agent.compute_action(state)
            action_norm_list.append(np.linalg.norm(action))
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        env_obj = env.unwrapped
        if args.eval_matrix:
            write_val = str(env_obj.num_exp) + ' ' \
                        + str(env_obj.rel_reward) + ' ' + str(env_obj.stable_res) + '\n'
            if args.out is not None:
                with open('output_files/{}_eval_matrix_benchmark.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open("output_files/eval_matrix_benchmark.txt", 'a') as f:
                        f.write(write_val)
                else:
                    with open("output_files/eval_matrix_benchmark_gaussian.txt", 'a') as f:
                        f.write(write_val)
        if args.eigv_gen:
            write_val = str(env_obj.max_EA) + ' ' + str(env_obj.rel_reward) + ' ' \
                        + str(env_obj.stable_res) + '\n'
            if args.out is not None:
                with open('output_files/{}_eigv_generalization.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                if args.gaussian_actions:
                    with open('output_files/eigv_generalization_gaussian.txt', 'a') as f:
                        f.write(write_val)
                else:
                    with open('output_files/eigv_generalization.txt', 'a') as f:
                        f.write(write_val)
        if args.opnorm_error:
            info_string = ''
            # note that args.eigv_gen and args.eval_matrix cannot be simultaneously true
            if args.eigv_gen:
                write_val = str(env_obj.max_EA) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
                info_string = "eig_gen"
            elif args.eval_matrix:
                write_val = str(env_obj.num_exp) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
                info_string = "eval_mat"
            else:
                print("One of args.eigv_gen or args.eval_matrix must be true")
                exit()

            if args.out is not None:
                with open('output_files/{}_opnorm_error_{}.txt'.format(args.out, info_string), 'a') as f:
                    f.write(write_val)
            else:
                if args.gaussian_actions:
                    with open('output_files/opnorm_error_gaussian_{}.txt'.format(info_string), 'a') as f:
                        f.write(write_val)
                else:
                    with open('output_files/opnorm_error_{}.txt'.format(info_string), 'a') as f:
                        f.write(write_val)
        if args.out is not None:
            rollouts.append(rollout)

    # save the norm of the action
    if args.out is not None:
        with open('output_files/{}_action_norm.txt'.format(args.out), 'a') as f:
            f.write(write_val)
    else:
        with open('output_files/action_norm.txt'.format(args.out), 'a') as f:
            f.write(write_val)

    graph(args, env_params)


def graph(args, env_params):
    if args.eval_matrix:
        if args.out:
            fname = "output_files/{}_eval_matrix_benchmark.txt".format(str(args.out))
            fname_2 = 'output_files/{}_opnorm_error.txt'.format(args.out)
        if not args.out:
            fname = "output_files/eval_matrix_benchmark.txt"
            fname_2 = 'output_files/opnorm_error.txt'
        ps.plot_subopt(fname)
        ps.plot_stability(fname)
        if args.opnorm_error:
            ps.plot_opnorms(fname_2, args.eval_matrix)
    if args.eigv_gen:
        if args.out:
            fname = "output_files/{}_eigv_generalization.txt".format(str(args.out))
            fname_2 = 'output_files/{}_opnorm_error.txt'.format(args.out)
        if not args.out:
            fname = "output_files/eigv_generalization.txt"
            fname_2 = 'output_files/opnorm_error.txt'
        # ps.plot_stability(fname)
        ps.plot_generalization_rewards(fname, int(args.high), int(env_params['dim']))
        if args.opnorm_error:
            ps.plot_opnorms(fname_2, False, env_params)


if __name__ == "__main__":
    global env_params
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if args.eigv_gen and args.eval_matrix:
        print("You can't test eigenvalue generalization and simultaneously have a fixed evaluation matrix")
        exit()
    if not args.eigv_gen and not args.eval_matrix:
        print("You have to test at least one of args.eval_matrix or args.eigv_gen")
        exit()

    if args.clear_graphs:
        for file in os.listdir('output_files/'):
            os.remove(os.path.join('output_files/', file))
    env_params = create_env_params(args)
    # TODO(@evinitsky) convert this into a shell script dude
    if args.create_all_graphs:
        # FIGURE SET 1
        # Dimension 3, constrained actions
        # Compare against the evaluation matrix, plot the operator norm error on that matrix
        # args.eigv_gen = False
        # args.eval_matrix = True
        # args.opnorm_error = True
        # gaussian_actions = False
        # args.out = 'dim3_full_constrained_eval'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3,
        #               "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        # run(args, parser, env_params)

        # FIGURE SET 2
        # Dimension 3, Compare against the evaluation matrix, gaussian actions
        # args.eigv_gen = False
        # args.eval_matrix = True
        # args.opnorm_error = True
        # gaussian_actions = True
        # args.out = 'dim3_full_constrained_eval_gauss'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3,
        #               "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 3
        # Dimension 3, constrained actions, only partial LS
        # Compare against the evaluation matrix
        # args.eigv_gen = False
        # args.eval_matrix = True
        # args.opnorm_error = True
        # gaussian_actions = False
        # full_ls = False
        # args.out = 'dim3_partial_constrained_eval'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": full_ls, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
        #               "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/partial_constrained_R3/checkpoint-2430'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 4
        # Dimension 3, Compare against the evaluation matrix, gaussian actions
        # args.eigv_gen = False
        # args.eval_matrix = True
        # args.opnorm_error = True
        # gaussian_actions = True
        # full_ls = False
        # args.out = 'dim3_partial_constrained_eval_gauss'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": full_ls, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/partial_constrained_R3/checkpoint-2430'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 5
        # Dimension 3, constrained actions
        # Don't compare against the evaluation matrix
        # args.eigv_gen = True
        # args.eval_matrix = False
        # args.opnorm_error = True
        # gaussian_actions = False
        # full_ls = True
        # args.out = 'dim3_full_constrained_gen'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": full_ls, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
        #               "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 6
        # Dimension 3, constrained actions
        # Don't compare against the evaluation matrix, gaussian actions
        # args.eigv_gen = True
        # args.eval_matrix = False
        # args.opnorm_error = True
        # gaussian_actions = True
        # args.out = 'dim3_full_constrained_gen_gauss'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
        #               "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 7
        # Dimension 3, constrained actions, partial least squares
        # Don't compare against the evaluation matrix
        args.eigv_gen = True
        args.eval_matrix = False
        args.opnorm_error = True
        gaussian_actions = False
        full_ls = False
        args.out = 'dim3_partial_constrained_gen'
        env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
                      "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
                      "full_ls": full_ls, "rand_num_exp": args.rand_num_exp,
                      "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
                      "eval_mode": True}
        args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                       '../trained_policies/full_constrained_R3/checkpoint-2400'))
        args.checkpoint = '/Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-02-2019/dim3_full_ls/dim3_full_ls/PPO_GenLQREnv-v0_1_lr=0.001_2019-08-02_01-06-570jak_wrm/checkpoint_2500'
        args.high = env_params['eigv_high']

        run(args, parser, env_params)

        # FIGURE SET 8
        # args.eigv_gen = True
        # args.eval_matrix = False
        # args.opnorm_error = True
        # gaussian_actions = False
        # full_ls = True
        # args.out = 'dim5_full_constrained_gen'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": full_ls, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": False, "dim": 5, "analytic_optimal_cost": True,
        #               "eval_mode": True}
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R5/checkpoint-2450'))
        # # args.config = os.path.abspath(os.path.join(os.path.dirname(__file__),
        # #                                            '../policies/full_constrained_R5/'))
        # register_env(env_name, lambda env_config: create_env(env_config))
        # run(args, parser, env_params)


        # FIGURE SET 4
        # args.eigv_gen = False
        # args.eval_matrix = True
        # args.opnorm_error = True
        # gaussian_actions = True
        # args.out = 'dim3_full_constrained_no_eval'
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)

        # FIGURE SET 3
        # args.eigv_gen = True
        # args.eval_matrix = False
        # eval_matrix = False
        # args.opnorm_error = True
        # gaussian_actions = False
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": args.eval_matrix,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        # args.high = env_params['eigv_high']
        #
        # run(args, parser, env_params)


        # FIGURE SET 2
        # del gym.envs.registry.env_specs["GenLQREnv-v0"] # hack to deregister envs
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/partial_constrained_R3/checkpoint-2430'))
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": True,
        #               "full_ls": False, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": False, "dim": 3, "analytic_optimal_cost": True, "eval_mode": True}
        # register_env(env_name, lambda env_config: create_env(env_config))
        # run(args, parser, env_params)

        # args.eigv_gen = True
        # args.eval_matrix = False
        # eval_matrix = False
        # args.opnorm_error = True
        # gaussian_actions = False
        # del gym.envs.registry.env_specs["GenLQREnv-v0"] # hack to deregister envs
        # env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
        #               "eigv_high": 20, "elem_sample": True, "eval_matrix": False,
        #               "full_ls": True, "rand_num_exp": args.rand_num_exp,
        #               "gaussian_actions": False, "dim": 5, "analytic_optimal_cost": True, "eval_mode": True}
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R5/checkpoint-2450'))
        # # args.config = os.path.abspath(os.path.join(os.path.dirname(__file__),
        # #                                            '../policies/full_constrained_R5/'))
        # register_env(env_name, lambda env_config: create_env(env_config))
        # run(args, parser, env_params)
    else:
        run(args, parser, env_params)
