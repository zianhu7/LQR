import gym
from gym import spaces
import numpy as np

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
        self.cov_w = self.params.get("cov_w", 1.0)
        self.action_space = spaces.Box(low=-np.sqrt(2 / np.pi), high=np.sqrt(2 / np.pi), shape=(self.dim,))
        # 2 at end is for 1. num_exp 2. exp_length param pass-in to NN
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(2 * self.dim,))

    def reset(self):
        pass

    def step(self, action):
        pass
