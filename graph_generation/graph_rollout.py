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
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
import plot_suboptimality as ps

# Example Usage via RLlib CLI:
"""
Generates plots from a checkpoint given the GenLQREnv:
    1. '--recht' Recht System benchmark as a function of rollout length (mutually exclusive from other graphs, need to separately run)
    2. '--eigv_gen' Eigenvalue generalization: Relative LQR Cost Suboptimality wrt top eigenvalue of A, Stability of policy wrt top eigenvalue of A; requires --high for eigenvalue bound sampling
    3. '--opnorm_error' Generates plot of ||A - A_est||_2 as a function of rollout length OR top eigenvalue depending on which of 1. or 2. flags it is run with

REQUIRED:
    1. '--full_ls' Based on whether trained policy has learned full or partial ls sampling
    2. If '--eigv_gen' is True, must specify '--gen_num_exp' to fix rollout length for replay
"""

env_name = "GenLQREnv"
env_version_num = 0
env_name = env_name + '-v' + str(env_version_num)


def pass_params_to_gym(env_name):
    register(
        id=env_name,
        entry_point=("envs.GenLQREnv:GenLQREnv"), kwargs={"env_params": env_params}
    )


def create_env(env_config):
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
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.")
    required_named.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--high", type=float, nargs='+', default=1,
                        help="Low bound for eigenvalue initialization")
    parser.add_argument("--recht", type=bool, default=False, 
                        help="Whether to benchmark on Recht system for R3")
    parser.add_argument("--eigv_gen", type=bool, default=False,
                        help="Eigenvalue generalization tests for eigenvalues of A")
    parser.add_argument("--opnorm_error", type=bool, default=False,
                        help="Operator norm error of (A-A_est)")
    parser.add_argument("--full_ls", type=bool, default=True,
                        help="Sampling type")
    parser.add_argument("--es", type=bool, default=True, help="Element sampling")
    parser.add_argument("--gen_num_exp", type=int, default=0, help="Number of experiments per rollout fixed for eigenvalue generalization replay")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
             "Supresses loading of configuration from checkpoint.")
    return parser

def create_env_params(args):
    env_params = {"horizon": 120, "exp_length": 6, "reward_threshold": -10, "eigv_low": 0, 
            "eigv_high": args.high, "elem_sample": args.es, "recht_sys": args.recht, "full_ls":args.full_ls, "gen_num_exp": args.gen_num_exp}
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

    #convert to max cpus available on system
    config['num_workers'] = 1
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = ModelCatalog.get_preprocessor_as_wrapper(gym.make(args.env))
    if args.out is not None:
        rollouts = []
    steps = 0
    total_stable = 0
    episode_reward = 0
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
            if not args.no_render:
                env.render()
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        env_obj = env.unwrapped
        if args.recht:
            write_val = str(env_obj.num_exp) + ' ' \
                        + str(env_obj.rel_reward) + ' ' + str(env_obj.stable_res)+'\n'
            if args.out is not None:
                with open('{}.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                with open("recht_benchmark.txt", 'a') as f:
                    f.write(write_val)
        if args.eigv_gen:
            write_val = str(env_obj.max_EA) + ' ' + str(env_obj.rel_reward) + ' ' + str(env_obj.stable_res) + '\n'
            if args.out is not None:
                with open('{}.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                with open('eigv_generalization.txt', 'a') as f:
                    f.write(write_val)
        if args.opnorm_error:
            if args.eigv_gen:
                write_val = str(env_obj.max_EA) + ' ' + str(env_obj.epsilon_A) + ' ' + str(env_obj.epsilon_B) + '\n'
            if args.recht:
                write_val = str(env_obj.num_exp) + ' ' + str(env_obj.epsilon_A) + ' ' + str(env_obj.epsilon_B) + '\n'
            if args.out is not None:
                with open('{}_opnorm_error.txt'.format(args.out), 'a') as f:
                    f.write(write_val)
            else:
                with open('opnorm_error.txt', 'a') as f:
                    f.write(write_val)
        if args.out is not None:
            rollouts.append(rollout)
    if args.recht:
        if args.out:
            fname = str(args.out)
            fname_2 = '{}_opnorm_error.txt'.format(args.out)
        if not args.out:
            fname = "recht_benchmark.txt"
            fname_2 = 'opnorm_error.txt'
        ps.plot_subopt(fname)
        ps.plot_stability(fname)
        if args.opnorm_error:
            ps.plot_opnorms(fname_2, args.recht)
    if args.eigv_gen:
        if args.out:
            fname = str(args.out)
            fname_2 = '{}_opnorm_error.txt'.format(args.out)
        if not args.out:
            fname = "eigv_generalization.txt"
            fname_2 = 'opnorm_error.txt'
        ps.plot_generalization_rewards(fname, args.high)
        if args.opnorm_error:
            ps.plot_opnorms(fname_2, False)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    env_params = create_env_params(args)
    register_env(env_name, lambda env_config: create_env(env_config))
    # build the runs against the Recht example
    run(args, parser, env_params)

    # build the runs against top eigenvalue generation
