"""Compare the state space coverage of a trained model with a Gaussian"""
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class

from envs import GenLQREnv
from utils.parsers import ReplayParser
from utils.plot_utils import lighten_color
from utils.rllib_utils import merge_dicts

if __name__=='__main__':

    ray.init()
    # Get the arguments
    parser = ReplayParser()
    args = parser.parse_args()

    env_params = {"horizon": args.horizon, "exp_length": args.exp_length,
                  "reward_threshold": -np.abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                   "eval_matrix": 1, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": 1, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": 0, "rand_num_exp": 0}
    args = parser.parse_args()

    # Instantiate the env
    config = args.config
    if not config:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            if not args.config:
                raise ValueError(
                    "Could not find params.pkl in either the checkpoint dir or "
                    "its parent directory.")
        else:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])

    # convert to max cpus available on system
    config['num_workers'] = 1

    # pull in the params from training time and overwrite. If statement for backwards compatibility
    if len(config["env_config"]) > 0:
        base_params = config["env_config"]["env_params"]
        env_params = merge_dicts(base_params, env_params)

    cls = get_agent_class(args.run)
    config["env_config"] = env_params
    agent = cls(env=GenLQREnv, config=config)
    agent.restore(os.path.join(args.checkpoint, os.path.basename(args.checkpoint).replace('_', '-')))

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
    else:
        env = GenLQREnv(env_params)

    # Run the trained model in the env and plot its trajectory
    exp_length = env.exp_length
    control_states = np.zeros((env_params["horizon"], env.dim))
    action_norm = []
    angle = []
    state = env.reset()
    done = False
    index = 0
    while not done:
        control_states[index] = state[index * env.dim : (index + 1) * env.dim]
        # save the state for replay
        action = agent.compute_action(state)
        print(action)
        action_norm.append(np.linalg.norm(action))
        angle = np.arctan(action[1]/action[0])
        state, reward, done, _ = env.step(action)
        index += 1

    # Plot the Gaussian trajectory in the
    env_params["gaussian_actions"] = 1
    env = GenLQREnv(env_params)
    exp_length = env.exp_length
    gauss_states = np.zeros((env_params["horizon"], env.dim))
    state = env.reset()
    done = False
    index = 0
    while not done:
        gauss_states[index] = state[index * env.dim : (index + 1) * env.dim]
        # save the state for replay
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        index += 1

    # now plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(env_params["horizon"] // exp_length):
        curr_state = control_states[i * exp_length: (i + 1) * exp_length]
        ax.plot(curr_state[:, 0], curr_state[:, 1], curr_state[:, 2], color=lighten_color('b', 0.91 ** i))
    for i in range(env_params["horizon"] // exp_length):
        curr_state = gauss_states[i * exp_length : (i + 1) * exp_length]
        ax.plot(curr_state[:, 0], curr_state[:, 1], curr_state[:, 2], color=lighten_color('r', 0.91 ** i))

    # now plot the trajectory in 2d
    plt.figure()
    for i in range(env_params["horizon"] // exp_length):
        curr_state = control_states[i * exp_length: (i + 1) * exp_length]
        plt.plot(curr_state[:, 0], curr_state[:, 1], color=lighten_color('b', 0.91 ** i))
    for i in range(env_params["horizon"] // exp_length):
        curr_state = gauss_states[i * exp_length : (i + 1) * exp_length]
        plt.plot(curr_state[:, 0], curr_state[:, 1], color=lighten_color('r', 0.91 ** i))

    # now plot a scatterplot
    plt.figure()
    plt.scatter(control_states[exp_length-1::exp_length, 0], control_states[exp_length-1::exp_length, 1], color='b')
    plt.scatter(gauss_states[exp_length-1::exp_length, 0], gauss_states[exp_length-1::exp_length, 1], color='r')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)

    # Now we plot the norm of the final point as a function of time
    control_norm = np.linalg.norm(control_states[exp_length - 1::exp_length], axis=1)
    gauss_norm = np.linalg.norm(gauss_states[exp_length - 1::exp_length], axis=1)
    index = list(range(control_norm.shape[0]))

    plt.figure()
    plt.plot(index, control_norm, 'b')
    plt.plot(index, gauss_norm, 'r')

    plt.figure()
    plt.plot(list(range(len(action_norm))), action_norm)

    plt.figure()
    plt.hist(angle)

    plt.show()