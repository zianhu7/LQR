"""Persistently apply the [1,1,1] vector with bounded norm`"""
import os

import numpy as np

from envs.GenLQREnv import GenLQREnv
from utils.parsers import RolloutParser


class NormExplorer(object):
    """This explorer applies the scaled vector of all 1s with norm np.sqrt(self.dim * 2 / np.pi)"""
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

    def compute_action(self):
        arr = np.ones(self.dim)
        arr = self.magnitude * arr / np.linalg.norm(arr)
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
                  "eval_matrix": args.eval_matrix, "full_ls": args.full_ls,
                  "dim": args.dim, "eval_mode": args.eval_mode, "analytic_optimal_cost": args.analytic_optimal_cost,
                  "gaussian_actions": args.gaussian_actions, "rand_num_exp": args.rand_num_exp}
    NormExplorer = NormExplorer(env_params)
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
        while not done:
            action = NormExplorer.compute_action()
            obs, reward, done, _ = env.step(action)
            reward_total += reward
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
                with open(os.path.join(filepath, "{}_norm_eval_matrix_benchmark.txt".format(args.out)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "norm_eval_matrix_benchmark.txt"), write_mode) as f:
                        f.write(write_val)

        if args.eigv_gen:
            write_val = str(env.max_EA) + ' ' + str(env.rel_reward) + ' ' \
                        + str(env.stable_res) + '\n'
            if args.out is not None:
                with open(os.path.join(filepath, "{}_norm_eigv_generalization.txt".format(args.out)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "norm_eigv_generalization.txt"), write_mode) as f:
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
                with open(os.path.join(filepath, "{}_norm_opnorm_error_{}.txt".format(args.out, info_string)),
                          write_mode) as f:
                    f.write(write_val)
            else:
                if not args.gaussian_actions:
                    with open(os.path.join(filepath, "norm_opnorm_error_{}.txt".format(info_string)),
                              write_mode) as f:
                        f.write(write_val)

        total_stable += bool(env.stable_res)
        rel_reward += env.rel_reward
        num_episodes += 1
        if first_write:
            first_write = False
    print("Mean rel reward is: {}, Fraction stable is {}".format(rel_reward / num_episodes, total_stable / num_episodes))
