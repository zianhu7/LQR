import math

import gym
import numpy as np
import scipy
from gym import spaces
from numpy.linalg import inv

from utils.lqr_utils import check_controllability, check_observability, check_stability, LQR_cost, sample_matrix, sda_estimate


class GenLQREnv(gym.Env):
    def __init__(self, env_params):  # Normalize Q, R
        '''
        Parameters
        =========
        self.dim: (int)
            Dimension of the A and B matrices.
        self.eval_matrix: (np.ndarray)
            Hard-coded in matrix used to generate some of the figures
        self.full_ls: (bool)
            Whether to use all the input-output pairs (if true) in LS, or only the last
            input-output pair of each rollout
        self.gaussian_actions: (bool)
            If true the actions are simply sampled for a Gaussian with zero mean and identity
            covariance. If false, the Neural Network is used to generate the actions.
        self.reward_threshold: (float)
            The value we clip the reward at.
        self.analytic_optimal_cost: (bool)
            If true, we compute the analytic optimal cost for our estimated \hat{K}. If false
            we just unroll the K for exp_length steps
        self.full_ls: (bool)
            If true, all the samples from the trials are used in the least squares
        self.gaussian_actions: (bool)
            If true, actions are sampled from a Gaussian with mean 0 and covariance cov_w * I where
            I is the identity matrix
        self.cov_w: (float)
            This is the scaling factor on the covariance of the Gaussian noise. The gaussian is
            N(0, cov_w * I).
        self.rand_num_exp: (bool)
            If true, the total number of trials is randomly sampled between Uniform(2 * dim, horizon / exp_length)
        self.exp_length: (int)
            How long each trial is. Defaults to 6.
        self.done_norm_cond: (float)
            If the norm of the state exceeds this value, the experiment ends
        '''
        self.params = env_params
        self.eigv_low, self.eigv_high = self.params["eigv_low"], self.params["eigv_high"]
        self.num_exp_bound = int(self.params["horizon"] / self.params["exp_length"])
        self.dim = self.params["dim"]
        self.eval_matrix = self.params["eval_matrix"]
        self.eval_mode = self.params["eval_mode"]
        self.analytic_optimal_cost = self.params["analytic_optimal_cost"]
        self.full_ls = self.params["full_ls"]
        self.gaussian_actions = self.params["gaussian_actions"]
        #self.obs_norm = self.params.get("obs_norm", 1.0)  # Value we normalize the observations by
        self.cov_w = self.params.get("cov_w")
        self.rand_num_exp = self.params["rand_num_exp"]
        self.exp_length = self.params["exp_length"]
        self.done_norm_cond = self.params.get("done_norm_cond")

        # We set the bounds of the box to be sqrt(2/pi) so that the norm matches the norm of sampling from
        # an actual Gaussian with covariance being the identity matrix.
        self.action_space = spaces.Box(low=-np.sqrt(2 / np.pi), high=np.sqrt(2 / np.pi), shape=(self.dim,))
        self.action_offset = self.dim * (self.exp_length + 1) * int(
            self.params["horizon"] / self.exp_length)
        # 2 at end is for 1. num_exp 2. exp_length param pass-in to NN
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf,
                                            shape=(self.action_offset + (self.params["horizon"] + 1) * self.dim + 2,))
        # Track trajectory
        self.reward_threshold = -abs(self.params["reward_threshold"])

    def generate_system(self):
        '''Generates the square A and B matrices. Guarantees that A and B form a controllable system'''
        # Make generate_system configurable/randomized

        # TODO(@evinitsky) switch the conditions from observability and controllability to stabilizable/detectable
        self.Q, self.R = 0.001 * np.eye(self.dim), np.eye(self.dim)
        if not self.eval_matrix:
            # If true, we sample the elements randomly with maximal element self.eigv_high
            A = sample_matrix(self.dim, self.eigv_high)
            B = sample_matrix(self.dim, self.eigv_high)
            Q_sqrt = scipy.linalg.sqrtm(self.Q)
            while not check_controllability(A, B) or not check_observability(A, Q_sqrt):
                A = sample_matrix(self.dim, self.eigv_high)
                B = sample_matrix(self.dim, self.eigv_high)
            self.A, self.B = A, B
        else:
            self.A = np.array([[1.01, 0.01, 0], [0.01, 1.01, 0.01], [0, 0.01, 1.01]])
            self.B = np.eye(self.dim)

    def step(self, action):
        '''Take in an action, step the dynamic, compute a reward, and check if the horizon is reached
        self.state is a padded matrix consisting of all the action, output pairs
        Padding is used since the experiments can be of variable length
        Returns
        -------
        state: x_t
        reward: J* - J (distance to optimal reward)
        completion: True is horizon number of steps is taken. If hit, we call reset_exp
        '''
        self.timestep += 1
        mean = [0] * self.dim
        cov = self.cov_w * np.eye(self.dim)
        noise = np.random.multivariate_normal(mean, cov)
        if self.gaussian_actions:
            a = np.random.multivariate_normal(mean, cov)
        else:
            a = action
        curr_state = self.states[self.curr_exp][-1]
        new_state = self.A @ curr_state + self.B @ a + noise
        self.update_state(new_state)
        self.update_action(a)
        self.states[self.curr_exp].append(list(new_state))
        self.inputs[self.curr_exp].append(a)
        completion = False
        if np.linalg.norm(new_state) > self.done_norm_cond:
            completion = True
        if self.horizon == self.timestep:
            completion = True
        if (self.timestep % self.exp_length == 0) and (not completion):
            self.reset_exp()
        # we got to the end of the horizon, compute the error on your estimates
        if completion and self.horizon == self.timestep:
            reward = self.calculate_reward()
        # we didn't get to the end, so we are penalized
        elif completion and self.horizon != self.timestep:
            reward = self.reward_threshold
        else:
            reward = 0
        return self.state, reward, completion, {}

    def reset_exp(self):
        '''Restarts the rollout process for a given experiment'''
        # Not global reset, only for each experiment
        self.curr_exp += 1
        new_state = np.zeros(self.dim)
        self.update_state(new_state)
        self.states[self.curr_exp].append(list(new_state))

    def update_state(self, new_state):
        '''Keep internal track of the state for plotting purposes'''
        start = self.timestep * self.dim
        for idx in range(self.dim):
            self.state[start + idx] = new_state[idx]

    def update_action(self, action):
        '''Keep internal track of the action for plotting purposes'''
        start = self.action_offset + self.timestep * self.dim
        for idx in range(self.dim):
            self.state[start + idx] = action[idx]

    def create_state(self):
        '''Initialize the zero padded state of the system'''
        self.state = [0 for _ in range(self.action_offset +
                                       (self.params["horizon"] + 1) * self.dim)] + \
                     [self.exp_length, self.num_exp]

    def reset(self):
        '''Reset the A and B matrices, reset the state matrix'''
        self.timestep = 0
        self.curr_exp = 0
        if self.rand_num_exp:
            self.num_exp = int(np.random.uniform(low=2 * self.dim, high=self.num_exp_bound))
        else:
            self.num_exp = self.num_exp_bound
        self.horizon = self.num_exp * self.exp_length
        self.states, self.inputs = [[] for i in range(self.num_exp)], [[] for i in range(self.num_exp)]
        self.create_state()
        new_state = np.zeros(self.dim)
        self.update_state(new_state)
        self.states[self.curr_exp].append(list(new_state))
        self.generate_system()
        e_A, _ = np.linalg.eig(self.A)
        e_B, _ = np.linalg.eig(self.B)
        self.max_EA = max([abs(e) for e in e_A])
        self.max_EB = max([abs(e) for e in e_B])
        return self.state

    ###############################################################################################
    #                                   UTILITY FUNCTIONS
    ###############################################################################################

    def ls_estimate(self):
        """Compute an LS estimate of the A and B matrices"""

        # NOTE: 2*self.dim is baked into the assumption that A,B are square of shape (self.dim, self.dim)
        if self.full_ls:
            X, Z = np.zeros((self.horizon, self.dim)), np.zeros((self.horizon, 2 * self.dim))
            for i in range(self.num_exp):
                for j in range(self.exp_length):
                    x_idx, z_idx = j + 1, j
                    pos = i * self.exp_length + j
                    X[pos] = self.states[i][x_idx]
                    z_layer = np.hstack([self.states[i][z_idx], self.inputs[i][z_idx]])
                    Z[pos] = z_layer
        if not self.full_ls:
            X, Z = np.zeros((self.num_exp, self.dim)), np.zeros((self.num_exp, 2 * self.dim))
            for i in range(self.num_exp):
                j = self.exp_length - 1
                x_idx, z_idx = j + 1, j
                X[i] = self.states[i][x_idx]
                z_layer = np.hstack([self.states[i][z_idx], self.inputs[i][z_idx]])
                Z[i] = z_layer
        try:
            theta = (inv(Z.T @ Z) @ (Z.T @ X)).T
        except:
            with open("err.txt", 'a') as f:
                f.write("lse" + '\n')
            return self.A, self.B
        # Reshape ensures that for the self.dim=1 case it is still in a matrix form to ensure consistency
        A, B = theta[:, :self.dim].reshape((self.dim, -1)), theta[:, self.dim:].reshape((self.dim, -1))
        return A, B

    def calculate_reward(self):
        '''Return the difference between J and J*'''
        # Assumes termination Q is same as regular Q
        Q, R, A, B = self.Q, self.R, self.A, self.B
        self.A_est, self.B_est = self.ls_estimate()
        try:
            K_hat = sda_estimate(self.A_est, self.B_est, Q, R)
        except:
            with open("err.txt", "a") as f:
                f.write("e" + '\n')
            return self.reward_threshold
        K_true = sda_estimate(self.A, self.B, Q, R)
        r_true, r_hat = 0, 0
        x0 = np.random.multivariate_normal([0] * self.dim, np.eye(self.dim))
        state_true = x0
        state_hat = np.copy(x0)
        if self.analytic_optimal_cost:
            is_stable = not check_stability(self.A, self.B, K_hat)
            if is_stable:
                r_hat = LQR_cost(A, B, K_hat, Q, R, self.cov_w)
                r_true = LQR_cost(A, B, K_true, Q, R, self.cov_w)
            # if we aren't stable under the true dynamics we go off to roughly infinity
            else:
                r_hat = 1e6
                r_true = 1
        else:
            for _ in range(self.exp_length):
                # Update r_hat
                noise = np.random.multivariate_normal([0] * self.dim, np.eye(self.dim))
                u_hat = K_hat @ state_hat
                r_hat += state_hat.T @ Q @ state_hat + u_hat.T @ R @ u_hat
                state_hat = A @ state_hat + B @ u_hat + noise
                # Update r_true
                u_true = K_true @ state_true
                r_true += state_true.T @ Q @ state_true + u_true.T @ R @ u_true
                state_true = A @ state_true + B @ u_true + noise
            r_hat += state_hat.T @ Q @ state_hat
            r_true += state_true.T @ Q @ state_true
        # Negative to turn into maximization problem for RL
        reward = -abs(r_hat - r_true)
        self.rel_reward = (-reward) / abs(r_true)
        # Don't hit this when we're creating graphs, only during training
        if not self.eval_mode:
            self.reward = max(self.reward_threshold, reward)
        else:
            self.reward = reward
        self.stable_res = not check_stability(self.A, self.B, K_hat)
        _, e_A, _ = np.linalg.svd(A - self.A_est)
        _, e_B, _ = np.linalg.svd(B - self.B_est)
        self.epsilon_A = max([abs(e) for e in e_A])
        self.epsilon_B = max([abs(e) for e in e_B])
        return self.reward
