import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from numpy.linalg import inv
import math
from scipy.linalg import solve_discrete_are as sda


class KEstimationEnv(gym.Env):
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
        self.horizon = self.params["horizon"]
        self.exp_length = self.params["exp_length"]
        self.eigv_low, self.eigv_high = self.params["eigv_low"], self.params["eigv_high"]
        self.dim = self.params["dim"]
        self.es = self.params["elem_sample"]
        self.stability_scaling = self.params["stability_scaling"]
        # self.generate_system()
        self.action_space = spaces.Box(low=-1, high=1, shape=(int(math.pow(self.dim, 2)),))
        # 2 at end is for 1. num_exp 2. exp_length param pass-in to NN
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf,
                                            shape=(self.dim * (self.horizon + 1),))
        # Track trajectory
        self.reward_threshold = self.params["reward_threshold"]

    def generate_system(self):
        '''Generates the square A and B matrices. Guarantees that A and B form a controllable system'''
        # Make generate_system configurable/randomized
        #TODO: What to make Q, R?
        self.Q, self.R = np.eye(self.dim), np.eye(self.dim)
        A = self.sample_matrix(self.eigv_high)
        B = np.eye(self.dim)
        while not self.check_controllability(A, B):
            A = self.sample_matrix(self.eigv_high)
        self.A, self.B = A, B

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
        curr_state = self.state[self.timestep * self.dim: (self.timestep + 1) * self.dim]
        self.timestep += 1
        mean = [0] * self.dim
        cov = np.eye(self.dim)
        noise = np.random.multivariate_normal(mean, cov)
        action = np.reshape(action, (self.dim, self.dim))
        a = action @ curr_state
        action_norm = np.linalg.norm(a)
        if action_norm > 1:
            a /= (np.linalg.norm(a) + 1e-5)
        new_state = self.A @ curr_state + self.B @ a + noise
        self.update_state(new_state)
        reward = - new_state.T @ self.Q @ new_state - a.T @ self.R @ a
        completion = False
        if self.horizon == self.timestep:
            completion = True
        if (self.timestep % self.exp_length == 0) and (not completion):
            self.reset_exp()
        if completion:
            stable = not self.check_stability(action)
            s_reward = self.stability_scaling if stable else -self.stability_scaling
            reward += s_reward
        return self.state, max(reward, self.reward_threshold), completion, {}

    def reset_exp(self):
        '''Restarts the rollout process for a given experiment'''
        # Not global reset, only for each experiment
        new_state = np.zeros(self.dim)
        self.update_state(new_state)

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
        self.state = [0 for _ in range(self.dim * (self.horizon + 1))]

    def reset(self):
        '''Reset the A and B matrices, reset the state matrix'''
        self.timestep = 0
        self.create_state()
        new_state = np.zeros(self.dim)
        self.update_state(new_state)
        self.generate_system()
        e_A, _ = np.linalg.eig(self.A)
        e_B, _ = np.linalg.eig(self.B)
        self.max_EA = max([abs(e) for e in e_A])
        self.max_EB = max([abs(e) for e in e_B])
        return self.state

    ###############################################################################################


    def check_controllability(self, A, B):
        '''Check that the controllability matrix is full rank'''
        dim = self.dim
        stack = []
        for i in range(dim):
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
