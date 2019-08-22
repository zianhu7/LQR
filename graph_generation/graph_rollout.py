'''Used to replay a saved checkpoint'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.tune.registry import register_env
import argparse
import json
import os
import pickle

import gym
from gym.envs.registration import register
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
import graph_generation.plot_suboptimality as ps

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
    2. If '--eigv_gen' is True, must specify '--gen_num_exp' to fix rollout length for replay
"""


env_name = "GenLQREnv"
env_version_num = 100
env_name = env_name + '-v' + str(env_version_num)


def pass_params_to_gym(_):
    global env_params
    # WARNING: env_params is being passed as a global variable
    register(
        id=env_name,
        entry_point=("envs.GenLQREnv:GenLQREnv"), kwargs={"env_params": env_params},
    )


def create_env(env_config):
    global env_name
    pass_params_to_gym(env_name)
    env = gym.envs.make(env_name)
    return env


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
                    "given a checkpoint.")

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
    required_named.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--high", type=int, default=2,
                        help="upper bound for eigenvalue initialization")
    parser.add_argument("--eval_matrix", type=int, default=0,
                        help="Whether to benchmark on `Sample complexity of quadratic "
                             "regulator` system for R3")
    parser.add_argument("--eigv_gen", type=int, default=0,
                        help="Eigenvalue generalization tests for eigenvalues of A")
    parser.add_argument("--opnorm_error", type=int, default=0,
                        help="Operator norm error of (A-A_est)")
    parser.add_argument("--full_ls", type=int, default=1,
                        help="Sampling type")
    parser.add_argument("--write_mode", type=str, default="a", help="use w to overwrite, a to append")
    parser.add_argument("--es", type=int, default=1, help="Element sampling")
    parser.add_argument("--gen_num_exp", type=int, default=0, help="If 0, the number of experiments is varied")
    parser.add_argument("--gaussian_actions", type=int, default=0, help="Run env with "
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
                  "full_ls": args.full_ls, "gen_num_exp": args.gen_num_exp,
                  "gaussian_actions": args.gaussian_actions, "dim": 3, "eval_mode": True,
                  "analytic_optimal_cost": True}
    return env_params


def run(args, parser, env_params):
    assert (args.eigv_gen or args.eval_matrix) and not (args.eigv_gen and args.eval_matrix), \
        print('either eval matrix or eigv_gen must be true, but both cant simultaneously be true')
    config = args.config
    if not config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path): raise ValueError(
            "Could not find params.json in either the checkpoint dir or "
            "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])

    # convert to max cpus available on system
    config['num_workers'] = 1
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = ModelCatalog.get_preprocessor_as_wrapper(gym.make(args.env))
    # Just check that the env actually has the settings you want.
    for k, v in env_params.items():
        assert getattr(env, k) == v, print('the values of {} do no match. The env has value {} and'
                                           ' you passed {}'.format(k, getattr(env, k), v))
    # env = agent.workers.local_worker().env
    # env.__init__(env_params)
    print(env.eigv_high)
    print(env.gaussian_actions)
    if args.out is not None:
        rollouts = []
    steps = 0
    total_stable = 0
    episode_reward = 0
    rewards = []
    eval_str_list = []
    eigv_str_list = []
    op_norm_str_list = []
    while steps < (num_steps or steps + 1):
        if args.out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        rel_reward = 0
        while not done and steps < (num_steps or steps + 1):
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        rewards.append(reward)
        env_obj = env.unwrapped
        if args.eval_matrix:
            write_val = str(env_obj.num_exp) + ' ' \
                        + str(env_obj.rel_reward) + ' ' + str(env_obj.stable_res) + '\n'
            eval_str_list.append(write_val)

        if args.eigv_gen:
            write_val = str(env_obj.max_EA) + ' ' + str(env_obj.rel_reward) + ' ' \
                        + str(env_obj.stable_res) + '\n'
            eigv_str_list.append(write_val)
        if args.opnorm_error:
            if args.eigv_gen:
                write_val = str(env_obj.max_EA) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
            if args.eval_matrix:
                write_val = str(env_obj.num_exp) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
            op_norm_str_list.append(write_val)

        print('the average reward was {}'.format(np.mean(rewards)))

    if args.eval_matrix:
        if args.out is not None:
            f_name = "output_files/{}_eval_matrix_benchmark.txt".format(args.out)
        else:
            if not args.gaussian_actions:
                f_name = "output_files/eval_matrix_benchmark.txt"
            else:
                f_name = "output_files/eval_matrix_benchmark_gaussian.txt"

        with open(f_name, args.write_mode) as filehandle:
            for list_item in eval_str_list:
                filehandle.write(list_item)

    if args.eigv_gen:
        if args.out is not None:
            f_name = "output_files/{}_eigv_generalization.txt".format(args.out)
        else:
            if not args.gaussian_actions:
                f_name = "output_files/eigv_generalization.txt"
            else:
                f_name = "output_files/eigv_generalization_gaussian.txt"
        with open(f_name, args.write_mode) as filehandle:
            for list_item in eigv_str_list:
                filehandle.write(list_item)

    if args.opnorm_error:
        if args.eigv_gen:
            write_val = str(env_obj.max_EA) + ' ' + str(env_obj.epsilon_A) + ' ' \
                        + str(env_obj.epsilon_B) + '\n'
        if args.eval_matrix:
            write_val = str(env_obj.num_exp) + ' ' + str(env_obj.epsilon_A) + ' ' \
                        + str(env_obj.epsilon_B) + '\n'
        op_norm_str_list.append(write_val)
        if args.out is not None:
            f_name = "output_files/{}_opnorm_error.txt".format(args.out)
        else:
            if not args.gaussian_actions:
                f_name = "output_files/opnorm_error.txt"
            else:
                f_name = "output_files/opnorm_error_gaussian.txt"
        with open(f_name, args.write_mode) as filehandle:
            for list_item in op_norm_str_list:
                filehandle.write(list_item)

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
    env_name = "GenLQREnv"
    env_version_num = 0
    env_name = env_name + '-v' + str(env_version_num)
    ray.init()
    parser = create_parser()
    args = parser.parse_args()
    if args.clear_graphs:
        for file in os.listdir('output_files/'):
            os.remove(os.path.join('output_files/', file))
    env_params = create_env_params(args)
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
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
        full_ls = True
        # args.out = 'dim3_partial_constrained_gen'
        args.out = "temp"
        env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
                      "eigv_high": 13, "elem_sample": True, "eval_matrix": args.eval_matrix,
                      "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
                      "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
                      "eval_mode": True}
        register_env(env_name, lambda env_config: create_env(env_config))
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R3/checkpoint-2400'))
        args.checkpoint = "/Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-21-2019/dim3_full_ls_lowcontrol/dim3_full_ls_lowcontrol/PPO_GenLQREnv-v0_1_2019-08-21_05-33-54bmrot4c6/checkpoint_2400/checkpoint-2400"
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
        #               "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": False, "gen_num_exp": args.gen_num_exp,
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
        #               "full_ls": True, "gen_num_exp": args.gen_num_exp,
        #               "gaussian_actions": False, "dim": 5, "analytic_optimal_cost": True, "eval_mode": True}
        # args.checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                '../trained_policies/full_constrained_R5/checkpoint-2450'))
        # # args.config = os.path.abspath(os.path.join(os.path.dirname(__file__),
        # #                                            '../policies/full_constrained_R5/'))
        # register_env(env_name, lambda env_config: create_env(env_config))
        # run(args, parser, env_params)
    else:
        register_env(env_name, lambda env_config: create_env(env_config))
        run(args, parser, env_params)
