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
    register( id=env_name,
      entry_point=("envs.GenLQREnv:GenLQREnv"),
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
                  "eigv_low": 0.5, "eigv_high": 2,
                  "elem_sample": True, "eval_matrix": False, "full_ls": True,
                  "dim": 5, "eval_mode": False, "analytic_optimal_cost": False, "gaussian_actions": False, "gen_num_exp": 0}
>>>>>>> 5c7b4fcbd8644748306c992546aec5bec3b9a1f4
    register_env(env_name, lambda env_config: create_env(env_config))
    num_cpus = 38
    ray.init(redis_address="localhost:6379")
    # ray.init(redirect_output=False)
    config = ppo.DEFAULT_CONFIG.copy()
    config["train_batch_size"] = 30000
    config["num_sgd_iter"]=10
    config["num_workers"]=num_cpus
    config["gamma"] = 0.95
    config["horizon"] = env_params["horizon"]
    config["use_gae"] = True
    config["lambda"] = 0.1
    config["lr"] = grid_search([5e-4, 8e-4, 5e-3, 5e-5, 1e-3, 1e-4, 8e-5, 1e-4, 9e-4, 2e-3])
    config["sgd_minibatch_size"] = 64
    config["model"].update({"fcnet_hiddens": [256, 256, 256]}) # number of hidden layers in NN


    trials = run_experiments({
            "GenLQR_R5_full_constrained": {
                "run": "PPO", # name of algorithm
                "env": "GenLQREnv-v0", # name of env
                "config": config,
                "checkpoint_freq": 30, # how often to save model params
                #"max_failures": 999 # Not worth changing
                "stop": {"training_iteration": 3000},
                'upload_dir': "s3://ethan.experiments/lqr/3-15-19/full_const_R5",
                'num_samples': 1
            }
        })
    #agent = ppo.PPOAgent(config=config, env=env_name)
    #filename = "reward_means_{}_{}.txt".format(env_params["horizon"],
                           # str(env_params["eigv_high"]).replace('.','-'))

    #for i in range(1000):
        #result = agent.train()
        #print('-'*60)
        #print("Epoch:" + str(i))
        #print(result["episode_reward_mean"])
        #print('-'*60)
        #with open(filename, 'a') as f:
            #f.write(str(result["episode_reward_mean"])+'\n')
