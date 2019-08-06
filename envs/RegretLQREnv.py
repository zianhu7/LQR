import gym
from gym import spaces
import numpy as np
import scipy

from matni_compare.python.utils import solve_least_squares
from utils.lqr_utils import sample_matrix, check_controllability, check_observability, dlqr, sda_estimate, LQR_cost

class RegretLQREnv(gym.Env):
    """In this env, after each step we synthesize a new K matrix and return its cost suboptimality (e.g. the regret)
        as the reward.
    """

    def __init__(self, env_params):
        self.params = env_params
        self.eigv_low, self.eigv_high = self.params["eigv_low"], self.params["eigv_high"]
        self.dim = self.params["dim"]
        self.eval_matrix = self.params["eval_matrix"]
        self.gaussian_actions = self.params["gaussian_actions"]
        self.obs_norm = self.params.get("obs_norm", 1.0)  # Value we normalize the observations by
        self.initial_samples = self.params.get("initial_samples", 100)  # how many samples we initialize the buffer with
        self.prime_excitation_low = self.params.get("prime_excitation_low", 0.5)  # lower bound on the exploration factor for initial rollouts
        self.prime_excitation_high = self.params.get("prime_excitation_high", 3.0)  # upper bound on the exploration factor for initial rollouts
        self.prime_excitation = 0
        self.cov_w = self.params.get("cov_w", 1.0)  # the std-dev of the actions gaussian if gaussian_actions is True
        self.dynamics_w = self.params.get("dynamics_w", 1.0)  # the std-dev of the dynamics gaussian
        self.action_space = spaces.Box(low=-np.sqrt(2 / np.pi), high=np.sqrt(2 / np.pi), shape=(self.dim,))
        # State is state, action, unrolled A_nom matrix, unrolled B_nom matrix, cov matrix
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(2 * self.dim + 2 * (self.dim ** 2) + ((2 * self.dim) ** 2),))
        self.state_buffer = []
        self.next_state_buffer = []
        self.action_buffer = []
        self._state_cur = np.zeros((self.dim, ))

    def reset(self):
        """Initialize the buffer with some sample states"""
        self.generate_system()
        self._state_cur = np.zeros((self.dim, ))
        self.prime_excitation = np.random.uniform(low=self.prime_excitation_low, high=self.prime_excitation_high)
        self.prime()
        K_true = sda_estimate(self.A, self.B, self.Q, self.R)
        self.J_star = LQR_cost(self.A, self.B, K_true, self.Q, self.R, self.cov_w)
        return self.construct_obs(self._state_cur, np.zeros(self.dim))

    def step(self, action):
        if self.gaussian_actions:
            xnext = self.update_dynamics(self.cov_w * np.random.normal(size=(self.dim)))
        else:
            xnext = self.update_dynamics(action)
        self.state_buffer.append(self._state_cur)
        self.action_buffer.append(action)
        self.next_state_buffer.append(xnext)
        # WARNING: we cannot provide the controller with J_nom or the regret, as you need the true system to do this
        self.A_hat, self.B_hat, self.cov = solve_least_squares(self.state_buffer, self.action_buffer,
                                                                 self.next_state_buffer)
        cost =  self._state_cur.dot(self.Q.dot(self._state_cur)) + action.dot(self.R.dot(action))
        regret = cost - self.J_star
        self._state_cur = xnext
        return self.construct_obs(self._state_cur, action) / self.obs_norm, -regret, False, {}

    def construct_obs(self, state, input):
        return np.concatenate((self.A_hat.reshape(-1), self.B_hat.reshape(-1), self.cov.reshape(-1), state, input))

    def update_dynamics(self, input):
        return self.A.dot(self._state_cur) + self.B.dot(input) + self.dynamics * np.random.normal(size=(self.dim))

    def prime(self):
        """Initialize the state-action buffer with 100 random samples"""
        _, K_init = dlqr(self.A, self.B, self.Q, self.R)
        for _ in range(self.initial_samples):
            inp = K_init.dot(self._state_cur) + self.prime_excitation * np.random.normal(size=(self.dim,))
            xnext = self.update_dynamics(inp)

            self.state_buffer.append(self._state_cur)
            self.action_buffer.append(inp)
            self.next_state_buffer.append(xnext)

            self._state_cur = xnext

        # compute the nominal estimates
        self.A_hat, self.B_hat, self.cov = solve_least_squares(self.state_buffer, self.action_buffer,
                                                                 self.next_state_buffer)

    def generate_system(self):
        """Generates the square A and B matrices. Guarantees that A and B form a controllable system"""
        # Make generate_system configurable/randomized

        # TODO(@evinitsky) switch the conditions from observability and controllability to stabilizable/detectable
        self.Q, self.R = 0.001 * np.eye(self.dim), np.eye(self.dim)
        if self.eval_matrix:
            self.A = np.array([[1.01, 0.01, 0], [0.01, 1.01, 0.01], [0, 0.01, 1.01]])
            self.B = np.eye(self.dim)
        else:
            # If true, we sample the elements randomly with maximal element self.eigv_high
            A = sample_matrix(self.dim, self.eigv_high)
            B = sample_matrix(self.dim, self.eigv_high)
            Q_sqrt = scipy.linalg.sqrtm(self.Q)
            while not check_controllability(A, B) and not check_observability(A, Q_sqrt):
                A = sample_matrix(self.dim, self.eigv_high)
                B = sample_matrix(self.dim, self.eigv_high)
            self.A, self.B = A, B
