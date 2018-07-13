import gym
from gym.envs.registration import register

import ray
import ray.rllib.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments

env_name = "LQR_Env"

def register_gym(env_name):
    register(id=env_name, entry_point='LQR.LQR_gym.envs:LQR',
            max_episode_steps=config['horizon'], kwargs={})

def create_env(env_config):
    register_gym(env_name)
    env = gym.envs.make(env_name)
    return env

def create_env(*_):
    register(
        id=env_name,
        entry_point='flow.envs:' + params["env_name"],
        max_episode_steps=env_params.horizon,
        kwargs={
            "env_params": env_params,
            "sumo_params": sumo_params,
            "scenario": scenario
        }
    )
    return gym.envs.make(env_name)

if __name__ == '__main__':
    num_cpus = 3
    ray.init(redirect_output=False)
    config = ppo.DEFAULT_CONFIG.copy()
    config["timesteps_per_batch"] = 64
    config["num_sgd_iter"]=10
    config["gamma"] = 0.95
    config["horizon"] = 200
    config["use_gae"] = False
    config["lambda"] = 0.1
    config["sgd_stepsize"] = .0003
    config["sgd_batchsize"] = 64
    config["min_steps_per_task"] = 100
    config["model"].update({"fcnet_hiddens": [256, 256]}) # number of hidden layers in NN
    register_env(env_name, lambda env_config: create_env(env_config))
    #trials = run_experiments({
    #        "LQR_tests": {
    #            "run": "PPO", # name of algorithm
    #            "env": "LQR_env", # name of env
    #            "config": {
    #               config.items()
    #            },
    #            "checkpoint_freq": 20, # how often to save model params
    #            "max_failures": 999, # Not worth changing
    #            "stop": {"training_iteration": 2}, 
    #            "trial_resources": {"cpu": 1, "gpu": 0, "extra_cpu": 0}
    #        },
    #    })
    agent = ppo.PPOAgent(config=config, env=env_name)
    for i in range(1000):
        result = agent.train()
        print result

