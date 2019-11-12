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
from ray.tune.registry import register_env

from envs import KEstimationEnv
import graph_generation.plot_suboptimality as ps
from utils.rllib_utils import merge_dicts

env_name = "KEstimationEnv"
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)
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
    required_named.add_argument("--env", type=str, default="KEstimationEnv", help="The gym environment to use.")
    parser.add_argument(
        "--steps", default=1000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--high", type=float, nargs='+', default=2,
                        help="upper bound for eigenvalue initialization")
    parser.add_argument("--elem_sample", type=int, default=1, help="Element sampling")
    parser.add_argument("--stability_scaling", type=int, default=10, help="Stability scaling")
    return parser


def create_env_params(args):
    env_params = {"horizon": 120, "reward_threshold": -10, "exp_length": 6,
                  "eigv_low": 0.5, "eigv_high": 2,
                  "elem_sample": True, "stability_scaling": 10,
                  "dim": 1}
    return env_params

def create_env(env_config):
    pass_params_to_gym(env_name)
    env = gym.envs.make(env_name)
    return env


def run(args, parser, env_params):
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
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
    # if len(config["env_config"]) > 0:
    #     base_params = config["env_config"]["env_params"]
    #     env_params = merge_dicts(json.loads(base_params), env_params)

    config["env_config"] = env_params
    agent = cls(env=KEstimationEnv, config=config)
    agent.restore(config_path)
    num_steps = int(args.steps)

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = KEstimationEnv(env_params)
    if args.out is not None:
        rollouts = []
    steps = 0
    action_norm_list = []
    post_stability = 0
    end_stability = 0
    total_exps = 0
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
        total_exps += 1
        s_history = env_obj.stability_history
        first_stable = s_history.index(1)
        post_stability += 1 if all([s_history[i] == 1 for i in range(first_stable, len(s_history))]) else 0
        end_stability += 1 if reward > 0 else 0
    print(f"total post stable: {post_stability/total_exps}")
    print(f"total end stable: {end_stability/total_exps}")




if __name__ == "__main__":
    global env_params
    ray.init()
    parser = create_parser()
    args = parser.parse_args()

    env_params = create_env_params(args)
    # TODO(@evinitsky) convert this into a shell script dude
    run(args, parser, env_params)
