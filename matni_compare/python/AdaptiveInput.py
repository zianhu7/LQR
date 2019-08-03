"""Here we use the adaptive input designer to schedule the rollouts. Since we currently have a max horizon length
   of 120, we stop after that many iterations.
"""

"""nominal.py

"""
import logging
import math
import os
import pickle

import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class

from envs import GenLQREnv
from matni_compare.python.adaptive import AdaptiveMethod
import matni_compare.python.utils as utils
from matni_compare.python.constants import horizon
from utils.rllib_utils import merge_dicts


class AdaptiveInputStrategy(AdaptiveMethod):
    """Adaptive control based on nominal estimates of the dynamics

    """

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w,
                 rls_lam,
                 sigma_explore,
                 reg,
                 epoch_multiplier,
                 checkpoint_path,
                 epoch_schedule='linear',):
        """

        :param Q:
        :param R:
        :param A_star:
        :param B_star:
        :param sigma_w:
        :param rls_lam:
        :param sigma_explore:
        :param reg:
        :param epoch_multiplier:
        :param checkpoint_path: Path to rllib checkpoint
        :param epoch_schedule:
        """
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._sigma_explore = sigma_explore
        self._reg = reg
        self._epoch_multiplier = epoch_multiplier
        if not epoch_schedule in ('linear', 'exponential'):
            raise ValueError("invalid epoch_schedule: {}".format(epoch_schedule))
        self._epoch_schedule = epoch_schedule
        self._logger = logging.getLogger(__name__)

        # instantiate the agent
        ray.init()
        # Get the arguments

        env_params = {"horizon": horizon, "exp_length": 6,
                      "reward_threshold": -10,
                      "eigv_low": 0, "eigv_high": 20,
                      "elem_sample": 0, "eval_matrix": 1, "full_ls": 1,
                      "dim": 3, "eval_mode": 1, "analytic_optimal_cost": 1,
                      "gaussian_actions": 0, "rand_num_exp": 0}

        # Instantiate the env
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
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

        cls = get_agent_class('PPO')
        config["env_config"] = env_params
        self.agent = cls(env=GenLQREnv, config=config)
        self.agent.restore(os.path.join(checkpoint_path, os.path.basename(checkpoint_path).replace('_', '-')))

        self.env = GenLQREnv(env_params)
        self.env.reset()

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._explore_stddev_history = []

    def _on_iteration_completion(self):
        self._explore_stddev_history.append(self._explore_stddev())

    def _design_controller(self, states, inputs, transitions, rng):
        # TODO(@evinitsky) how often is this actually called?
        import ipdb; ipdb.set_trace()
        logger = self._get_logger()

        # do a least squares fit and controller based on the nominal
        Anom, Bnom, _ = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)
        _, K = utils.dlqr(Anom, Bnom, self._Q, self._R)
        self._current_K = K

        # compute the infinite horizon cost of this controller
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._current_K, self._Q, self._R, self._sigma_w)

        # for debugging purposes,
        # check to see if this controller will stabilize the true system
        rho_true = utils.spectral_radius(self._A_star + self._B_star.dot(self._current_K))
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))

        return (Anom, Bnom, Jnom)

    def _epoch_length(self):
        if self._epoch_schedule == 'linear':
            return self._epoch_multiplier * (self._epoch_idx + 1)
        else:
            return self._epoch_multiplier * math.pow(2, self._epoch_idx)

    def _explore_stddev(self):
        if self._epoch_schedule == 'linear':
            sigma_explore_decay = 1/math.pow(self._epoch_idx + 1, 1/3)
            return self._sigma_explore * sigma_explore_decay
        else:
            sigma_explore_decay = 1/math.pow(2, self._epoch_idx/6)
            return self._sigma_explore * sigma_explore_decay

    def _should_terminate_epoch(self):
        if (self._iteration_within_epoch_idx >= self._epoch_length()):
            return True
        else:
            return False

    # We use the nominal controller but with the exploratory action selected from our policy
    def _get_input(self, state, rng):
        import ipdb; ipdb.set_trace()
        rng = self._get_rng(rng)
        ctrl_input = self._current_K.dot(state)
        explore_input = self.agent.compute_action(self.env.state)

        # Explore in a weighted mixture between the agent and the nominal controller
        explore_input *= self._explore_stddev() / np.linalg.norm(explore_input)
        final_action = ctrl_input + explore_input

        # Update the state of the agent
        self.env.update_state(state)
        self.env.update_action(final_action)
        # Update the env so all of the tracking is done correctly
        self.env.step(final_action)

        return final_action

def _main():
    import matni_compare.python.examples as examples
    A_star, B_star = examples.unstable_laplacian_dynamics()

    # define costs
    Q = 1e-3 * np.eye(3)
    R = np.eye(3)

    # initial controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(3), np.eye(3))

    rng = np.random

    env = NominalStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=1,
                          sigma_explore=0.1,
                          reg=1e-5,
                          epoch_multiplier=10,
                          rls_lam=None)

    env.reset(rng)
    env.prime(100, K_init, 0.1, rng)
    for idx in range(500):
        env.step(rng)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(linewidth=200)
    _main()
