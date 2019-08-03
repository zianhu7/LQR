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
        self.magnitude = np.sqrt(self.dim * 2 / np.pi)
        self.curr_direction = 0

    def update_direction(self):
        self.curr_direction += 1
        self.curr_direction %= self.dim

    def reset(self):
        self.curr_direction = np.random.randint(low=0, high=self.dim)

    def compute_action(self):
        arr = np.zeros(self.dim)
        arr[self.curr_direction] = self.magnitude
        return arr


if __name__ == '__main__':
    parser = RolloutParser()
    args = parser.parse_args()
    if args.eigv_gen and args.eval_matrix:
        print("You can't test eigenvalue generalization and simultaneously have a fixed evaluation matrix")
        exit()
    if not args.eigv_gen and not args.eval_matrix:
        print("You have to test at least one of args.eval_matrix or args.eigv_gen")
        exit()

    exp_length = args.exp_length
    env_params = {"horizon": args.horizon, "exp_length": args.exp_length,
                  "reward_threshold":-np.abs(args.reward_threshold),
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "elem_sample": args.elem_sample, "eval_matrix": args.eval_matrix, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": args.eval_mode, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "rand_num_exp": args.rand_num_exp}
    FiechterExplorer = FiechterExplorer(env_params)
    env = GenLQREnv(env_params)

    # Set up the writing
    filepath = os.path.join(os.path.dirname(__file__), '../../graph_generation/output_files')
    if args.append:
        write_mode = 'a'
    else:
        write_mode = 'w'

    # initialize the rollout variables
    steps = 0
    total_stable = 0
    num_episodes = 0
    first_write = True  # used to control whether the files are overwritten
    while steps < args.steps:

        # If we haven't indicated that we should append, the first write will overwrite the contents of the files
        if first_write and not args.append:
            write_mode = 'w'
        else:
            write_mode = 'a'

        obs = env.reset()
        env_steps = 0
        done = False
        reward_total = 0.0
        rel_reward = 0
        trial_counter = 0
        FiechterExplorer.reset()
        while not done:
            action = FiechterExplorer.compute_action()
            obs, reward, done, _ = env.step(action)
            reward_total += reward

            # we switch unit vectors every exp_length
            trial_counter += 1
            trial_counter %= exp_length
            if trial_counter == 0:
                FiechterExplorer.update_direction()
            steps += 1
            env_steps += 1

        if args.out:
            with open(os.path.join(filepath, '{}.txt'.format(args.out)), write_mode) as f:
                write_val = str(env.max_EA) + ' ' \
                            + str(env.rel_reward) + ' ' + str(env.stable_res) + '\n'
                f.write(write_val)

        if args.eval_matrix:
            write_val = str(env.num_exp) + ' ' \
                        + str(env.rel_reward) + ' ' + str(env.stable_res) + '\n'
            if args.out is not None:
                with open(os.path.join(filepath, "{}_fiechter_eval_matrix_benchmark.txt".format(args.out)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "fiechter_eval_matrix_benchmark.txt"), write_mode) as f:
                        f.write(write_val)

        if args.eigv_gen:
            write_val = str(env.max_EA) + ' ' + str(env.rel_reward) + ' ' \
                        + str(env.stable_res) + '\n'
            if args.out is not None:
                with open(os.path.join(filepath, "{}_fiechter_eigv_generalization.txt".format(args.out)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "fiechter_eigv_generalization.txt"), write_mode) as f:
                        f.write(write_val)

        if args.opnorm_error:
            info_string = ''
            if args.eigv_gen:
                write_val = str(env.max_EA) + ' ' + str(env.epsilon_A) + ' ' \
                            + str(env.epsilon_B) + '\n'
                info_string = "eig_gen"
            if args.eval_matrix:
                write_val = str(env.num_exp) + ' ' + str(env.epsilon_A) + ' ' \
                            + str(env.epsilon_B) + '\n'
                info_string = "eval_mat"
            if args.out is not None:
                with open(os.path.join(filepath, "{}_fiechter_opnorm_error_{}.txt".format(args.out, info_string)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "fiechter_opnorm_error_{}.txt".format(info_string)),
                              write_mode) as f:
                        f.write(write_val)

        total_stable += bool(env.stable_res)
        rel_reward += env.rel_reward
        num_episodes += 1
        if first_write:
            first_write = False
    print("Mean rel reward is: {}, Fraction stable is {}".format(rel_reward / num_episodes, total_stable / num_episodes))
