from datetime import datetime
import json
import os

import boto3
import gym
from gym.envs.registration import register
import numpy as np
import tensorflow as tf

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2

from utils.parsers import GenLQRParserBaseline


def create_env(env_name, env_config, register_env=True):
    if register_env:
        register(id=env_name,
                 entry_point=("envs.GenLQREnv:GenLQREnv"),
                 max_episode_steps=env_config["horizon"],
                 kwargs={"env_params": env_params}
                 )
    def _init():
        env = gym.envs.make(env_name)
        return env
    return _init

def create_monitor_env(env_name, env_config, log_dir, register_env=True):
    if register_env:
        register(id=env_name,
                 entry_point=("envs.GenLQREnv:GenLQREnv"),
                 max_episode_steps=env_config["horizon"],
                 kwargs={"env_params": env_params}
                 )
    def _init():
        env = gym.envs.make(env_name)
        return Monitor(env, log_dir, allow_early_resets=True)
    return _init

if __name__ == '__main__':
    # TODO(hyperparam sweeping)
    parser = GenLQRParserBaseline()
    args = parser.parse_args()

    # create the saving directory
    save_path = os.path.realpath(os.path.expanduser('~/baseline_results/{}'.format(args.exp_title)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ###############################################################################################################
    #                              CREATE THE CALLBACK
    ###############################################################################################################
    best_mean_reward, n_steps = -np.inf, 0
    def single_cpu_callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 1000 calls
        if (n_steps + 2) % args.checkpoint_freq == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(save_path), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward,
                                                                                               mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(save_path + 'best_model.pkl')
        n_steps += 1
        return True

    ###############################################################################################################
    #                              CONSTRUCT THE ENV
    ###############################################################################################################
    env_params = {"horizon": args.horizon, "exp_length": args.exp_length,
                  "reward_threshold":-abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "eval_matrix": args.eval_matrix, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": args.eval_mode, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "rand_num_exp": args.rand_num_exp}
    env_name = "GenLQREnv"
    if args.num_cpus == 1:
        env_constructor = Monitor(create_env(env_name + '-v0', env_params)(), save_path, allow_early_resets=True)
        env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    else:
        env = SubprocVecEnv([create_env(env_name + '-v{}'.format(i), env_params) for i in range(args.num_cpus)])

    ###############################################################################################################
    #                              TRAIN
    ###############################################################################################################
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    model = PPO2('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                 n_steps=args.rollout_size, tensorboard_log=save_path, learning_rate=0.00025)
    if args.callback and args.num_cpus==1:
        model.learn(total_timesteps=args.num_steps, callback=single_cpu_callback)
    else:
        model.learn(total_timesteps=args.num_steps)

    print('Saving the trained model!')
    model.save(os.path.join(save_path, args.exp_title))
    with open(os.path.join(save_path, args.exp_title) + '.json', 'w') as outfile:
        json.dump(env_params, outfile, sort_keys=True, indent=4)
    if args.use_s3:
        s3_resource = boto3.resource('s3')
        s3_folder = "final_cdc_lqr/" + datetime.now().strftime("%m-%d-%Y") + '/' + args.exp_title
        for dirName, subdirList, fileList in os.walk(save_path):
            for file in fileList:
                s3_resource.Object("eugene.experiments", s3_folder).upload_file(os.path.join(dirName, file))

    del model
    del env_params

    ###############################################################################################################
    #                              REPLAY
    ###############################################################################################################

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(os.path.join(save_path, args.exp_title))
    with open(os.path.join(save_path, args.exp_title) + '.json', 'r') as outfile:
        env_params = json.load(outfile)
    env = create_env(env_name + '-v0', env_params, register_env=False)()
    reward = 0
    num_trials = 2
    for i in range(num_trials):
        obs = env.reset()
        j = 0
        done = False
        while j < env_params["horizon"] and not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            reward += rewards
            j += 1
        print('trial {}, mean final reward is {}'.format(i, reward / (i + 1)))
