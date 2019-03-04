import gym
from gym.envs.registration import register

import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments, grid_search

env_name = "GenLQREnv" 
env_version_num=0
env_name = env_name + '-v' + str(env_version_num)

def pass_params_to_gym(env_name):
    register(
      id=env_name,
      entry_point=("GenLQREnv:GenLQREnv"),
      max_episode_steps=env_params["horizon"],
      kwargs={"env_params":env_params}
    )


def create_env(env_config):
    pass_params_to_gym(env_name) 
    env = gym.envs.make(env_name)
    return env

if __name__ == '__main__':
    #horizon, exp_length upper bounds
    env_params = {"horizon": 120, "exp_length":6, "reward_threshold":-10,
                  "eigv_low": 0.5, "eigv_high": 2, "q_scaling":[0,2], "r_scaling":[0,2],
                  "elem_sample":True}
    register_env(env_name, lambda env_config: create_env(env_config))
    num_cpus = 15
    ray.init(redirect_output=False)
    config = ppo.DEFAULT_CONFIG.copy()
    config["train_batch_size"] = 30000
    config["num_sgd_iter"]=10
    config["num_workers"]=num_cpus
    config["gamma"] = 0.95
    config["horizon"] = env_params["horizon"]
    config["use_gae"] = True
    config["lambda"] = 0.1
    config["lr"] = grid_search([5e-6])
    config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256, 256]}) # number of hidden layers in NN


    trials = run_experiments({
            "GenLQR_tests": {
                "run": "PPO", # name of algorithm
                "env": "GenLQREnv-v0", # name of env
                "config": config,
                "checkpoint_freq": 20, # how often to save model params
                #"max_failures": 999 # Not worth changing
                "stop": {"training_iteration": 2000},
                'upload_dir': "s3://eugene.experiments/cdc_lqr_paper/3-03-2019/LQR_test",
            }
        })
    #agent = ppo.PPOAgent(config=config, env=env_name)
    #filename = "reward_means_{}_{}.txt".format(env_params["horizon"], str(env_params["eigv_high"]).replace('.','-'))

    #for i in range(1000):
        #result = agent.train()
        #print('-'*60)
        #print("Epoch:" + str(i))
        #print(result["episode_reward_mean"])
        #print('-'*60)
        #with open(filename, 'a') as f:
            #f.write(str(result["episode_reward_mean"])+'\n')
