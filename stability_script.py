import gym
from gym.envs.registration import register

import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments

env_name = "StabilityEnv"
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)

def pass_params_to_gym(env_name):
    register(
      id=env_name,
      entry_point=("StabilityEnv:StabilityEnv"),
      max_episode_steps=env_params["horizon"],
      kwargs={"env_params":env_params}
    )


def create_env(env_config):
    pass_params_to_gym(env_name) 
    env = gym.envs.make(env_name)
    return env

if __name__ == '__main__':
    #Ensure that horizon is divisible by exp_length
    env_params = {"horizon":60, "exp_length":6, "reward_threshold":-10}
    register_env(env_name, lambda env_config: create_env(env_config))
    num_cpus = 1
    ray.init(redirect_output=False)
    config = ppo.DEFAULT_CONFIG.copy()
    config["train_batch_size"] = 1000
    config["num_sgd_iter"]=10
    config["num_workers"]=num_cpus
    config["gamma"] = 0.95
    config["horizon"] = env_params["horizon"]
    config["use_gae"] = False
    config["lambda"] = 0.1
    config["lr"] = .0003
    config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256, 256]}) # number of hidden layers in NN


    #trials = run_experiments({
            #"LQR_tests": {
                #"run": "PPO", # name of algorithm
                #"env": "LQREnv-v0", # name of env
                #"config": config,
                #"checkpoint_freq": 20, # how often to save model params
                #"max_failures": 999, # Not worth changing
                #"stop": {"training_iteration": 1},
                #"upload_dir": "s3://ethan.experiments/lqr"
            #},
        #})
    agent = ppo.PPOAgent(config=config, env=env_name)

    for i in range(400):
        result = agent.train()
        print(result)
        #with open('run_reward_means_eigv5+.txt', 'a') as f:
            #f.write(str(result.episode_reward_mean)+'\n')
