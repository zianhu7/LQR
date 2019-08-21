from datetime import datetime
import json

import gym
from gym.envs.registration import register
import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments, grid_search

from utils.parsers import GenLQRParserRLlib

env_name = "GenLQREnv" 
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)


def pass_params_to_gym(env_name):
    register( id=env_name,
      entry_point=("envs.GenLQREnv:GenLQREnv"),
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
    #horizon, exp_length upper bounds
    env_params = {"horizon": args.horizon, "exp_length": args.exp_length, "reward_threshold": -abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "elem_sample": True, "eval_matrix": False, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": False, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "gen_num_exp": 0}

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
        config["lr"] = grid_search([1e-4, 1e-3])
    else:
        config["lr"] = 1e-3
    config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256, 256]}) # number of hidden layers in NN

    # save the env params for later replay
    flow_json = json.dumps(env_params, sort_keys=True, indent=4)
    config['env_config']['env_params'] = flow_json

    s3_string = "s3://eugene.experiments/final_cdc_lqr/" \
                + datetime.now().strftime("%m-%d-%Y") + '/' + args.exp_title
    config['env'] = env_name

    exp_dict = {args.exp_title: {
            "run": "PPO",  # name of algorithm
            "env": "GenLQREnv-v0",  # name of env
            "config": config,
            "checkpoint_freq": args.checkpoint_freq,  # how often to save model params
            "stop": {"training_iteration": args.num_iters},
            'num_samples': args.num_samples
        }}
    if args.use_s3:
        exp_dict[args.exp_title]['upload_dir'] = s3_string

    trials = run_experiments(exp_dict)