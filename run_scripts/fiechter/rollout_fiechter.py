"""Runs the active exploration algorithm from `PAC adaptive control of linear systems`"""
import os

import numpy as np

from envs.GenLQREnv import GenLQREnv
from utils.parsers import RolloutParser


class FiechterExplorer(object):
    """This explorer picks a unit direction for each trial and sets all of its actions in that direction for
    that trial. It cycles through the unit basis for its directions."""
    def __init__(self, params):
        """
        Parameters
        ==========
        params: (dict)
            dict storing all the experiment keys
        dim: (int)
            dimension of the A and B matrices. Assumed to be square.
        magnitude: (float)
            scaling of the magnitude of the action. The action is e_i * magnitude
        """
        self.params = params
        self.dim = params["dim"]
        self.magnitude = np.sqrt(2 / np.pi)
        self.curr_direction = 0

    def update_direction(self):
        self.curr_direction += 1
        self.curr_direction %= self.dim

    def reset(self):
        self.curr_direction = np.random.randint(low=0, high=self.dim)

    def compute_action(self):
        arr = np.zeros(self.dim, 1)
        arr[self.curr_direction] = self.magnitude
        return arr


if __name__ == '__main__':
    parser = RolloutParser()
    args = parser.parse_args()
    FiechterExplorer = FiechterExplorer(args)
    steps = 0
    total_stable = 0
    exp_length = args.exp_length
    env = GenLQREnv(args)
    filepath = os.path.join(os.path.dirname(__file__), '../../graph_generation/output_files')
    while steps < args.steps:
        obs = env.reset()
        env_steps = 0
        done = False
        reward_total = 0.0
        rel_reward = 0
        trial_counter = 0
        FiechterExplorer.reset()
        while env_steps < args.horizon and not done:
            action = FiechterExplorer.compute_action()
            obs, reward, done, _ = env.step(action)
            reward_total += reward

            # we switch unit vectors every exp_length
            trial_counter += 1
            trial_counter %= exp_length
            if trial_counter == 0:
                FiechterExplorer.update_direction()
            steps += 1

        if args.out is not None:
            with open(os.path.join(filepath, '{}.txt'.format(args.out)), 'w') as f:
                write_val = str(env.unwrapped.eigv_bound) + ' ' \
                            + str(env.unwrapped.rel_reward) + ' ' + str(env.unwrapped.stable_res)
                print(write_val)
                f.write(write_val)
                f.write('\n')

        total_stable += bool(env.unwrapped.stable_res)
        rel_reward += env.unwrapped.rel_reward
