"""Here we use the adaptive input designer to schedule the rollouts. Since we currently have a max horizon length
   of 120, we stop after that many iterations.
"""

"""nominal.py

"""
import collections
import logging
import math

import numpy as np
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from matni_compare.python.adaptive import AdaptiveMethod
import matni_compare.python.utils as utils


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
                 agent,
                 env_params,
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

        # use these to track the lstm state
        # what should these be initialized to?
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
        self.agent_states = {}
        self.prev_actions = {}
        self.agent_states[DEFAULT_POLICY_ID] = state_init[DEFAULT_POLICY_ID]
        self.prev_actions[DEFAULT_POLICY_ID] = action_init[DEFAULT_POLICY_ID]

        self.agent = agent

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._explore_stddev_history = []

    def _on_iteration_completion(self):
        self._explore_stddev_history.append(self._explore_stddev())

    def _design_controller(self, states, inputs, transitions, rng):
        # TODO(@evinitsky) how often is this actually called? Not as often as you'd expect. TBD.
        logger = self._get_logger()

        # do a least squares fit and controller based on the nominal
        # TODO(@evinitsky) should we allow this to be updated?
        self.Anom, self.Bnom, self.cov = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)
        _, K = utils.dlqr(self.Anom, self.Bnom, self._Q, self._R)
        self._current_K = K

        # compute the infinite horizon cost of this controller
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._current_K, self._Q, self._R, self._sigma_w)

        # for debugging purposes,
        # check to see if this controller will stabilize the true system
        rho_true = utils.spectral_radius(self._A_star + self._B_star.dot(self._current_K))
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))

        return (self.Anom, self.Bnom, Jnom)

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
        obs = np.concatenate((self.Anom.reshape(-1), self.Bnom.reshape(-1), self.cov.reshape(-1),
                              state, self.prev_actions[DEFAULT_POLICY_ID]))
        a_action, p_state, _ = self.agent.compute_action(
                            obs,
                            state=self.agent_states[DEFAULT_POLICY_ID],
                            prev_action=self.prev_actions[DEFAULT_POLICY_ID],
                            policy_id=DEFAULT_POLICY_ID)
        import ipdb; ipdb.set_trace()
        self.agent_states[DEFAULT_POLICY_ID] = p_state
        self.prev_actions[DEFAULT_POLICY_ID] = a_action
        print(state)
        print(self.agent_states)

        return a_action
