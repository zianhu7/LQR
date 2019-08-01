from datetime import datetime

import gym
from gym.envs.registration import register
import numpy as np
import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run, grid_search

from utils.parsers import GenLQRParserRLlib

env_name = "GenLQREnv" 
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)


def pass_params_to_gym(env_name):
    register(id=env_name,
      entry_point="envs.GenLQREnv:GenLQREnv",
      max_episode_steps=env_params["horizon"],
      kwargs={"env_params": env_params}
    )


def create_env(env_config):
    pass_params_to_gym(env_name) 
    env = gym.envs.make(env_name)
    return env

if __name__ == '__main__':
    parser = GenLQRParserRLlib()
    args = parser.parse_args()

    env_params = {"horizon": args.horizon, "exp_length": args.exp_length,
                  "reward_threshold":-np.abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "elem_sample": args.elem_sample, "eval_matrix": args.eval_matrix, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": args.eval_mode, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "rand_num_exp": args.rand_num_exp}
    register_env(env_name, lambda env_config: create_env(env_config))
    num_cpus = args.num_cpus
    if args.multi_node:
        ray.init(redis_address="localhost:6379")
    else:
        ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["train_batch_size"] = args.train_batch_size
    config["num_sgd_iter"]=10
    config["num_workers"] = args.num_cpus
    config["gamma"] = 0.95
    config["horizon"] = args.horizon
    config["use_gae"] = True
    config["lambda"] = 0.1
    if args.grid_search:
        config["lr"] = grid_search([5e-4, 8e-4, 5e-3, 5e-5, 1e-3, 1e-4, 8e-5, 1e-4, 9e-4, 2e-3])
    else:
        config["lr"] = 5e-4
    config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256, 256]}) # number of hidden layers in NN

    s3_string = "s3://eugene.experiments/final_cdc_lqr/" \
                + datetime.now().strftime("%m-%d-%Y") + '/' + args.exp_title
    config['env'] = env_name
    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': 'PPO',
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    run(**exp_dict, queue_trials=True)
