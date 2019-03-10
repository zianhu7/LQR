
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

#Example Usage via RLlib CLI:
    #rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    #--env CartPole-v0 --steps 1000000 --out rollouts.pkl
#Example Usage via executable:
    #./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    #--env CartPole-v0 --steps 1000000 --out rollouts.pkl

env_name = "GenLQREnv"
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)

eigv_low, eigv_high = 1e-6, 1


def pass_params_to_gym(env_name):
    register(
      id=env_name,
      entry_point=("GenLQREnv:GenLQREnv"), kwargs={"env_params":env_params}
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
    required_named.add_argument( "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--low", type=float, nargs='+', default=1e-6, help="Low bound for eigenvalue initialization")
    parser.add_argument("--high", type=float, nargs='+', default=1, help="Low bound for eigenvalue initialization")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    return parser


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
        if args.out is not None:
            rollouts.append(rollout)
        """
        with open('gen_es_recht.txt', 'a') as f:
            #write_val = str(env.unwrapped.eigv_bound)+' '+str(env.unwrapped.rel_reward) + ' '+ str(env.unwrapped.stable_res)
            write_v
            print(write_val)
            f.write(write_val)
            f.write('\n')
        """
        total_stable += bool(env.unwrapped.stable_res)
        episode_reward += reward_total
        rel_reward += env.unwrapped.rel_reward
        print(env.unwrapped.rel_reward)
    num_episodes = num_steps/env_params["horizon"]
    print(rel_reward/num_episodes, total_stable/num_episodes)
    if args.out is not None:
        pickle.dump(rollouts, open(args.out, "wb"))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    env_params = {"horizon":120, "exp_length":6, "reward_threshold":-10, "eigv_low":0.5, "eigv_high":2, "elem_sample":True, "recht_sys":True}
    register_env(env_name, lambda env_config: create_env(env_config))
    run(args, parser, env_params)
