from datetime import datetime
import json

import gym
from gym.envs.registration import register
import pytz

import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run, grid_search

from utils.parsers import KEstimationParserRLlib

env_name = "KEstimationEnv"
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)


def pass_params_to_gym(env_name):
    register( id=env_name,
      entry_point=("envs.KEstimationEnv:KEstimationEnv"),
      max_episode_steps=env_params["horizon"],
      kwargs={"env_params":env_params}
    )


def create_env(env_config):
    pass_params_to_gym(env_name)
    env = gym.envs.make(env_name)
    return env

def on_episode_end(info):
    env = info['env'].get_unwrapped()[0].env
    episode = info["episode"]
    episode.custom_metrics["end_stable"] = env.end_stable
    episode.custom_metrics["avg_action_magnitude"] = env.avg_action_magnitude

if __name__ == '__main__':
    parser = KEstimationParserRLlib()
    args = parser.parse_args()

    env_params = {"horizon": args.horizon, "reward_threshold": -abs(args.reward_threshold),
                  "exp_length": args.exp_length, "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "elem_sample": args.elem_sample, "stability_scaling": args.stability_scaling, "dim": args.dim, "end_scaling": args.end_scaling,
                  "use_lstm": args.use_lstm}
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
        config["lr"] = grid_search([1e-4, 5e-4, 1e-3, 5e-3, 1e-5, 5e-5])
    else:
        config["lr"] = 1e-3
        config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256], "use_lstm": env_params["use_lstm"]}) # number of hidden layers in NN
    config["callbacks"] = {"on_episode_end": ray.tune.function(on_episode_end)}

    # save the env params for later replay
    flow_json = json.dumps(env_params, sort_keys=True, indent=4)
    config['env_config']['env_params'] = flow_json
    date=datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    s3_string = "s3://ethan.experiments/k_estimation/" \
                + date + '/' + args.exp_title
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

    run(**exp_dict, queue_trials=False)
