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
    parser.add_argument("--high", type=float, nargs='+', default=1,
                        help="upper bound for eigenvalue initialization")
    parser.add_argument("--eval_matrix", type=bool, default=False,
                        help="Whether to benchmark on `Sample complexity of quadratic "
                             "regulator` system for R3")
    parser.add_argument("--eigv_gen", type=bool, default=False,
                        help="Eigenvalue generalization tests for eigenvalues of A")
    parser.add_argument("--opnorm_error", type=bool, default=False,
                        help="Operator norm error of (A-A_est)")
    parser.add_argument("--full_ls", type=bool, default=True,
                        help="Sampling type")
    parser.add_argument("--es", type=bool, default=True, help="Element sampling")
    parser.add_argument("--gen_num_exp", type=int, default=0, help="If 0, the number of experiments is varied")
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
                  "eigv_high": 1.0, "elem_sample": args.es, "eval_matrix": args.eval_matrix,
                  "full_ls": args.full_ls, "gen_num_exp": args.gen_num_exp,
                  "gaussian_actions": True, "dim": 3, "eval_mode": False,
                  "analytic_optimal_cost": True}
    return env_params


def run(args, parser, env_params):
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

    # if hasattr(agent, "local_evaluator"):
    #     env = agent.local_evaluator.env
    # else:
    #     env = ModelCatalog.get_preprocessor_as_wrapper(gym.make(args.env))
    env = agent.workers.local_worker().env
    env.__init__(env_params)
    import ipdb; ipdb.set_trace()
    print(env.eigv_high)
    print(env.gaussian_actions)
    if args.out is not None:
        rollouts = []
    steps = 0
    total_stable = 0
    episode_reward = 0
    rewards = []
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
                if not args.gaussian_actions:
                    with open('output_files/eigv_generalization.txt', 'a') as f:
                        f.write(write_val)
                else:
                    with open('output_files/eigv_generalization_gaussian.txt', 'a') as f:
                        f.write(write_val)
        if args.opnorm_error:
            if args.eigv_gen:
                write_val = str(env_obj.max_EA) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
            if args.eval_matrix:
                write_val = str(env_obj.num_exp) + ' ' + str(env_obj.epsilon_A) + ' ' \
                            + str(env_obj.epsilon_B) + '\n'
            if args.out is not None:
                with open('output_files/{}_opnorm_error.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open('output_files/opnorm_error.txt', 'a') as f:
                        f.write(write_val)
                else:
                    with open('output_files/opnorm_error_gaussian.txt', 'a') as f:
                        f.write(write_val)
        if args.out is not None:
            rollouts.append(rollout)

        print('the average reward was {}'.format(np.mean(rewards)))
    # graph(args, env_params)


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
        full_ls = False
        args.out = 'dim3_partial_constrained_gen'
        env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0,
                      "eigv_high": 2, "elem_sample": True, "eval_matrix": args.eval_matrix,
                      "full_ls": full_ls, "gen_num_exp": args.gen_num_exp,
                      "gaussian_actions": gaussian_actions, "dim": 3, "analytic_optimal_cost": True,
                      "eval_mode": False}
        register_env(env_name, lambda env_config: create_env(env_config))
        args.checkpoint = "/Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-20-2019/dim3_full_ls_1000000cond_fullrank/dim3_full_ls_1000000cond_fullrank/PPO_GenLQREnv-v0_1_2019-08-20_00-34-047js2qpm3/checkpoint_600/checkpoint-600"
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
