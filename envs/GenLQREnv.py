import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from numpy.linalg import inv
import math
from scipy.linalg import solve_discrete_are as sda


class GenLQREnv(gym.Env):
    def __init__(self, env_params):  # Normalize Q, R
        '''
        self.es: (bool)
            If true randomly sample the elements of A and B with top eigenvalues of
            self.eigv_high and min of -eigv_low. If false, sample diagonalizable
            A and B with eigenvalues between eigv_low and eigv_high
        self.dim: (int)
            Dimension of the A and B matrices.
        self.eval_matric: (np.ndarray)
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
        '''
        self.params = env_params
        self.eigv_low, self.eigv_high = self.params["eigv_low"], self.params["eigv_high"]
        self.num_exp_bound = int(self.params["horizon"] / self.params["exp_length"])
        self.dim = self.params["dim"]
        self.es = self.params["elem_sample"]
        self.eval_matrix = self.params["eval_matrix"]
        self.eval_mode = self.params["eval_mode"]
        self.analytic_optimal_cost = self.params["analytic_optimal_cost"]
        self.full_ls = self.params["full_ls"]
        self.gaussian_actions = self.params["gaussian_actions"]
        # self.generate_system()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.dim,))
        self.action_offset = self.dim * (self.params["exp_length"] + 1) * int(
            self.params["horizon"] / self.params["exp_length"])
        # 2 at end is for 1. num_exp 2. exp_length param pass-in to NN
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf,
                                            shape=(self.action_offset + (self.params["horizon"] + 1) * self.dim + 2,))
        # Track trajectory
        self.reward_threshold = self.params["reward_threshold"]

    def generate_system(self):
        '''Generates the square A and B matrices. Guarantees that A and B form a controllable system'''
        # Make generate_system configurable/randomized
        self.Q, self.R = 0.001 * np.eye(self.dim), np.eye(self.dim)
        if not self.eval_matrix:
            if not self.es:
                self.eigv_bound = math.ceil(np.random.uniform(low=self.eigv_low,
                                                              high=self.eigv_high))
                self.a_eigv = np.random.uniform(low=self.eigv_low, high=self.eigv_bound,
                                                size=self.dim)
                self.b_eigv = np.random.uniform(low=self.eigv_low, high=self.eigv_bound,
                                                size=self.dim)
                A, B = self.a_eigv * np.eye(self.dim), self.b_eigv * np.eye(self.dim)
                # Ensure PD A, controllable system
                while not self.check_controllability(A, B):
                    self.a_eigv = np.random.uniform(low=self.eigv_low, high=self.eigv_high,
                                                    size=self.dim)
                    self.b_eigv = np.random.uniform(low=self.eigv_low, high=self.eigv_high,
                                                    size=self.dim)
                    A, B = self.a_eigv * np.eye(self.dim), self.b_eigv * np.eye(self.dim)
                P_A, P_B = self.rvs(self.dim), self.rvs(self.dim)
                self.A, self.B = P_A @ A @ P_A.T, P_B @ B @ P_B.T
            else:
                A = self.sample_matrix(self.eigv_high)
                B = self.sample_matrix(self.eigv_high)
                while not self.check_controllability(A, B):
                    A = self.sample_matrix(self.eigv_high)
                    B = self.sample_matrix(self.eigv_high)
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
        cov = np.eye(self.dim)
        noise = np.random.multivariate_normal(mean, cov)
        if self.gaussian_actions:
            a = np.random.multivariate_normal(mean, cov)
        if not self.gaussian_actions:
            a = action
        curr_state = self.states[self.curr_exp][-1]
        new_state = self.A @ curr_state + self.B @ a + noise
        self.update_state(new_state)
        self.update_action(a)
        self.states[self.curr_exp].append(list(new_state))
        self.inputs[self.curr_exp].append(a)
        completion = False
        if self.horizon == self.timestep:
            completion = True
        if (self.timestep % self.exp_length == 0) and (not completion):
            self.reset_exp()
        if completion:
            reward = self.calculate_reward()
        else:
            reward = 0
        if completion:
            print(self.timestep)
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
        # MODIFICATION, change back (only for eigv generalization replay experiment)
        if not self.params["gen_num_exp"]:
            self.num_exp = int(np.random.uniform(low=2 * self.dim, high=self.num_exp_bound))
        if self.params["gen_num_exp"]:
            self.num_exp = self.params["gen_num_exp"]
        self.exp_length = int(self.params["exp_length"])
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

    # Generate random orthornormal matrix of given dim
    def rvs(self, dim):
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())  # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat = np.eye(dim)
            mat[n - 1:, n - 1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H

    def ls_estimate(self):
        '''Compute an LS estimate of the A and B matrices'''

        # NOTE: 2*self.dim is baked into the assumption that A,B are square of shape (self.dim, self.dim)
        # if num_exp == 3 and exp_length == 3:
        # import ipdb;ipdb.set_trace()
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

    def sda_estimate(self, A, B):
        '''Solve the discrete algebraic ricatti equation to compute the optimal feedback'''
        Q, R = self.Q, self.R
        X = sda(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A
        return X, -K

    def estimate_K(self, horizon, A, B):
        '''Solve for K recursively. Not used.'''
        Q, R = self.Q, self.R
        # Calculate P matrices first for each step
        P_matrices = np.zeros((horizon + 1, Q.shape[0], Q.shape[1]))
        P_matrices[horizon] = Q
        for i in range(horizon - 1, 0, -1):
            P_t = P_matrices[i + 1]
            P_matrices[i] = Q + (A.T @ P_t @ A) - (A.T @ P_t @ B @ np.matmul(inv(R + B.T @ P_t @ B), B.T @ P_t @ A))
        # Hardcoded shape of K, change to inferred shape for diverse testing
        K_matrices = np.zeros((horizon, self.dim, self.dim))
        for i in range(horizon):
            P_i = P_matrices[i + 1]
            K_matrices[i] = -np.matmul(inv(R + B.T @ P_i @ B), B.T @ P_i @ A)
        return K_matrices

    def calculate_reward(self):
        '''Return the difference between J and J*'''
        # Assumes termination Q is same as regular Q
        Q, R, A, B = self.Q, self.R, self.A, self.B
        A_est, B_est = self.ls_estimate()
        try:
            P_ss_hat, K_hat = self.sda_estimate(A_est, B_est)
        except:
            with open("err.txt", "a") as f:
                f.write("e" + '\n')
            return self.reward_threshold
        P_ss_true, K_true = self.sda_estimate(self.A, self.B)
        r_true, r_hat = 0, 0
        # Evolve trajectory based on computing input using both K
        # synth_traj, true_traj = [[],[]], [[],[]]
        # for i in range(self.num_exp):
        # state_true, state_hat = self.states[i][0], self.states[i][0]
        # true_traj[0].append(state_true); synth_traj[0].append(state_hat)
        x0 = np.random.multivariate_normal([0] * self.dim, np.eye(self.dim))
        state_true = x0
        state_hat = np.copy(x0)
        if self.analytic_optimal_cost:
            is_stable = not self.check_stability(K_hat)
            if is_stable:
                cov = np.eye(self.dim)
                r_hat = np.trace(cov @ P_ss_hat)
                r_true = np.trace(cov @ P_ss_true)
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
        print(self.reward)
        self.inv_reward = -reward
        self.stable_res = not self.check_stability(K_hat)
        _, e_A, _ = np.linalg.svd(A - A_est)
        _, e_B, _ = np.linalg.svd(B - B_est)
        self.epsilon_A = max([abs(e) for e in e_A])
        self.epsilon_B = max([abs(e) for e in e_B])
        return self.reward

    def check_controllability(self, A, B):
        '''Check that the controllability matrix is full rank'''
        dim = self.dim
        stack = []
        for i in range(dim - 1):
            term = B @ np.linalg.matrix_power(A, i)
            stack.append(term)
        gramian = np.hstack(stack)
        return np.linalg.matrix_rank(gramian) == dim

    def check_stability(self, control):
        '''Confirm that the feedback matrix stabilizes the system'''
        mat = self.A + self.B @ control
        return np.any([abs(e) > 1 for e in np.linalg.eigvals(mat)])

    def sample_matrix(self, bound):
        '''Return a random matrix'''

        def generate():
            mat = np.eye(self.dim)
            for i in range(self.dim):
                elems = np.random.uniform(low=-bound, high=bound, size=self.dim)
                mat[i] = elems
            return mat

        rv = generate()
        while np.linalg.matrix_rank(rv) != self.dim:
            rv = generate()
        return rv
