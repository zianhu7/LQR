import argparse
import datetime
import json
import os

import gym
from gym.envs.registration import register
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from utils.parsers import GenLQRParserBaseline


def create_env(env_name, env_config, register_env=True):
    if register_env:
        register(id=env_name,
                 entry_point=("envs.GenLQREnv:GenLQREnv"),
                 max_episode_steps=env_config["horizon"],
                 kwargs={"env_params": env_params}
                 )
    env = gym.envs.make(env_name)
    return env


if __name__ == '__main__':
    parser = GenLQRParserBaseline()
    args = parser.parse_args()

    env_params = {"horizon": args.horizon, "exp_length": args.exp_length,
                  "reward_threshold":-abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "elem_sample": args.elem_sample, "eval_matrix": args.eval_matrix, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": args.eval_mode, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "rand_num_exp": args.rand_num_exp}
    env_name = "GenLQREnv"
    if args.num_cpus == 1:
        env_constructor = create_env(env_name + '-v0', env_params)
        env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    else:
        env = SubprocVecEnv([create_env(env_name + '-v{}'.format(i), env_params) for i in range(args.num_cpus)])

    # TODO(control learning rate, model size)
    # TODO(add s3 uploading)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    model = PPO2('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, n_steps=args.rollout_size)
    model.learn(total_timesteps=args.num_steps)

    if not os.path.exists(os.path.realpath(os.path.expanduser('~/baseline_results'))):
        os.makedirs(os.path.realpath(os.path.expanduser('~/baseline_results')))
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    save_path = os.path.join(path, args.exp_title)

    print('Saving the trained model!')
    model.save(save_path)
    # dump the flow params
    with open(os.path.join(path, args.exp_title) + '.json', 'w') as outfile:
        json.dump(env_params, outfile, sort_keys=True, indent=4)
    del model
    del env_params

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    with open(os.path.join(path, args.exp_title) + '.json', 'r') as outfile:
        env_params = json.load(outfile)
    env_constructor = create_env(env_name + '-v0', env_params, register_env=False)
    env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    obs = env.reset()
    reward = 0
    for i in range(env_params["horizon"]):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))
