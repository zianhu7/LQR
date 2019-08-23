import argparse
import collections
import pickle
import os

import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
import seaborn as sns
sns.set_style('ticks')

from envs import RegretLQREnv
from utils.parsers import regret_env_args

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


checkpoint_path = "/Users/eugenevinitsky/Desktop/Research/Data/cdc_lqr_paper/08-16-2019/dim3_full_ls_regret_d100/dim3_full_ls_regret_d100/PPO_RegretLQREnv-v0_1_lr=0.001_2019-08-16_03-21-2872__udt8/checkpoint_1000"
# Run the adaptive input methods
# TODO parallelize. It's kind of hard though because of rllib.
# set up the agent
ray.init()
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

# Set up the env
# figure out a way not to hard code this
parser = argparse.ArgumentParser()
regret_env_args(parser)
args = parser.parse_args()
env_params = {"eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
              "eval_matrix": True, "initial_samples": args.initial_samples,
              "dim": args.dim, "prime_excitation_low": args.prime_excitation_low,
              "prime_excitation_high": args.prime_excitation_high, "cov_w": args.cov_w,
              "gaussian_actions": args.gaussian_actions, "dynamics_w": args.dynamics_w,
              "obs_norm": args.obs_norm, "done_norm_cond": 100}
cls = get_agent_class('PPO')
config["env_config"] = env_params
agent = cls(env=RegretLQREnv, config=config)
agent.restore(os.path.join(checkpoint_path, os.path.basename(checkpoint_path).replace('_', '-')))

env = RegretLQREnv(env_params)
multiagent = isinstance(env, MultiAgentEnv)
if agent.workers.local_worker().multiagent:
    policy_agent_mapping = agent.config["multiagent"][
        "policy_mapping_fn"]

policy_map = agent.workers.local_worker().policy_map
state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
action_init = {
    p: _flatten_action(m.action_space.sample())
    for p, m in policy_map.items()
}


# while steps < (env_params["horizon"] or steps + 1):
mapping_cache = {}  # in case policy_agent_mapping is stochastic

obs = env.reset()
policy_agent_mapping = default_policy_agent_mapping
agent_states = DefaultMapping(
    lambda agent_id: state_init[mapping_cache[agent_id]])
prev_actions = DefaultMapping(
    lambda agent_id: action_init[mapping_cache[agent_id]])
prev_rewards = collections.defaultdict(lambda: 0.)
done = False
reward_total = 0.0
steps = 0
while not done and steps < args.horizon:
    multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
    action_dict = {}
    for agent_id, a_obs in multi_obs.items():
        if a_obs is not None:
            policy_id = mapping_cache.setdefault(
                agent_id, policy_agent_mapping(agent_id))
            p_use_lstm = use_lstm[policy_id]
            if p_use_lstm:
                import ipdb; ipdb.set_trace()
                a_action, p_state, _ = agent.compute_action(
                    a_obs,
                    state=agent_states[agent_id],
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    policy_id=policy_id)
                agent_states[agent_id] = p_state
            else:
                a_action = agent.compute_action(
                    a_obs,
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    policy_id=policy_id)
            a_action = _flatten_action(a_action)  # tuple actions
            action_dict[agent_id] = a_action
            prev_actions[agent_id] = a_action
    action = action_dict

    action = action if multiagent else action[_DUMMY_AGENT_ID]
    next_obs, reward, done, _ = env.step(action)
    if multiagent:
        for agent_id, r in reward.items():
            prev_rewards[agent_id] = r
    else:
        prev_rewards[_DUMMY_AGENT_ID] = reward

    if multiagent:
        done = done["__all__"]
        reward_total += sum(reward.values())
    else:
        reward_total += reward
    steps += 1
    obs = next_obs
print("Episode reward", reward_total)