"""Script version of regret comparison notebook because I hate Jupyter notebooks (for actually running code).
Great for prototyping though. No hate."""
import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pickle
import sys
import time

import matplotlib.pylab as plt
import ray
from ray.rllib.agents.registry import get_agent_class
import seaborn as sns
sns.set_style('ticks')

from envs import RegretLQREnv
from matni_compare.python import utils
from matni_compare.python import examples
from matni_compare.python.adaptiveinput import AdaptiveInputStrategy
from matni_compare.python.constants import rng, horizon, trials_per_method
from matni_compare.python.nominal import NominalStrategy
from matni_compare.python.ofu import OFUStrategy
from matni_compare.python.optimal import OptimalStrategy
from matni_compare.python.sls import SLS_FIRStrategy, SLS_CommonLyapunovStrategy, sls_common_lyapunov, SLSInfeasibleException
from matni_compare.python.ts import TSStrategy

logging.basicConfig(level=logging.WARN)

####################################################################################################################
#                            DEFINING TRUE SYSTEM DYNAMICS
####################################################################################################################

def set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation, sigma_excitation=0.1):
    n,p = B_star.shape
    # design a stabilizing controller
    _, K_init = utils.dlqr(A_star, B_star, 1e-3*np.eye(n), np.eye(p))
    assert utils.spectral_radius(A_star + B_star.dot(K_init)) < 1
    Q = qr_ratio * np.eye(n)
    R = np.eye(p)
    sigma_w = 1
    return A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w

def laplacian_dynamics(qr_ratio=1e1, prime_horizon=100, prime_excitation=1):
    A_star, B_star = examples.unstable_laplacian_dynamics()
    return set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation)

def unstable_dynamics(qr_ratio=1e1, prime_horizon=250, prime_excitation=2):
    A_star, B_star = examples.transient_dynamics(diag_coeff=2, upperdiag=4)
    return set_up_example(A_star, B_star, qr_ratio, prime_horizon, prime_excitation, sigma_excitation=0.1)


example = laplacian_dynamics() # unstable_dynamics()
A_star, B_star, K_init, Q, R, prime_horizon, prime_excitation, sigma_excitation, sigma_w = example

print(A_star)
print(B_star)
print(K_init)
print("prime_horizon", prime_horizon)
print("prime_excitation", prime_excitation)
print("sigma_excitation", sigma_excitation)


# # Constructors for different adaptive methods

def optimal_ctor():
    return OptimalStrategy(Q=Q, R=R, A_star=A_star, B_star=B_star, sigma_w=sigma_w)

def nominal_ctor():
    return NominalStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=sigma_w,
                          sigma_explore=sigma_excitation,
                          reg=1e-5,
                          epoch_multiplier=10, rls_lam=None)

def ofu_ctor():
    return OFUStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  actual_error_multiplier=1, rls_lam=None)

def ts_ctor():
    return TSStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  tau=500,
                  actual_error_multiplier=1, rls_lam=None)

def sls_fir_ctor():
    return SLS_FIRStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  truncation_length=12,
                  actual_error_multiplier=1, rls_lam=None)

def sls_cl_ctor():
    return SLS_CommonLyapunovStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  actual_error_multiplier=1, rls_lam=None)


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
env_params = {"horizon": args.horizon,
                  "eigv_low": args.eigv_low, "eigv_high": args.eigv_high,
                  "eval_matrix": args.eval_matrix, "initial_samples": args.initial_samples,
                  "dim": args.dim, "prime_excitation_low": args.prime_excitation_low,
                  "prime_excitation_high": args.prime_excitation_high, "cov_w": args.cov_w,
                  "gaussian_actions": args.gaussian_actions, "dynamics_w": args.dynamics_w,
                  "obs_norm": args.obs_norm}
cls = get_agent_class('PPO')
config["env_config"] = env_params
agent = cls(env=RegretLQREnv, config=config)
agent.restore(os.path.join(checkpoint_path, os.path.basename(checkpoint_path).replace('_', '-')))


def adaptive_input_actor():
    return AdaptiveInputStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=sigma_w,
                          sigma_explore=sigma_excitation,
                          reg=1e-5,
                          epoch_multiplier=10,
                          rls_lam=None,
                          agent=agent,
                          env_params=env_params,
                                 epoch_schedule='linear')

# # Helper methods for running in parallel

prime_seed = 45727
def run_one_trial(new_env_ctor, seed, prime_fixed=False):
    rng = np.random.RandomState(seed)
    if prime_fixed: # reducing variance
        rng_prime = np.random.RandomState(prime_seed) 
    else:
        rng_prime = rng
    env = new_env_ctor()
    env.reset(rng_prime)
    env.prime(prime_horizon, K_init, prime_excitation, rng_prime)
    regret = np.array([env.step(rng) for _ in range(horizon)])
    env.complete_epoch(rng)
    err, cost = env.get_statistics(iteration_based=True)
    return regret, err, cost

def spawn_invocation(method, p, prime_fixed=False):
    seed = np.random.randint(0xFFFFFFFF)
    ctor = {
        'optimal': optimal_ctor,
        'nominal': nominal_ctor,
        'ofu': ofu_ctor,
        'ts': ts_ctor,
        'sls_fir': sls_fir_ctor,
        'sls_cl': sls_cl_ctor,
    }[method]
    return (p.apply_async(run_one_trial, (ctor, seed, prime_fixed)), seed)

def process_future_list(ftchs):
    regrets = []
    errors = []
    costs = []
    seeds = []
    bad_invocations = 0
    for ftch, seed in ftchs:
        try:
            reg, err, cost = ftch.get()
        except Exception as e:
            bad_invocations += 1
            continue
        regrets.append(reg)
        errors.append(err)
        costs.append(cost)
        seeds.append(seed)
    return np.array(regrets), np.array(errors), np.array(costs), np.array(seeds), bad_invocations


# # Running experiments and plotting results
# Run the comparison methods
# TODO(add bad invocations)
strategies = ['optimal', 'nominal', 'ofu', 'ts', 'sls_cl', 'sls_fir']
start_time = time.time()
with Pool(processes=cpu_count()) as p:
    all_futures = [[spawn_invocation(method, p, prime_fixed=True)
                    for _ in range(trials_per_method)] for method in strategies]
    list_of_results = [process_future_list(ftchs) for ftchs in all_futures]
print("finished execution in {} seconds".format(time.time() - start_time))

start_time = time.time()
# Run the adaptive input designer
adapt_regret = []
adapt_cost = []
for i in range(trials_per_method):
    seed = np.random.randint(0xFFFFFFFF)
    regret, err, cost = run_one_trial(adaptive_input_actor, seed)
    adapt_regret.append(regret)
    adapt_cost.append(cost)
print("finished execution in {} seconds".format(time.time() - start_time))


def get_errorbars(regrets, q=10, percent_bad=0.0):
    median = np.percentile(regrets, q=max(50-percent_bad, 0), axis=0)
    low10 = np.percentile(regrets, q=q, axis=0)
    high90 = np.percentile(regrets, q=max(100-(q-percent_bad), 0), axis=0)
    return median, low10, high90


def plot_list_medquantile(datalist, title, legendlist=None, xlabel=None, ylabel=None, semilogy=False,
                          loc='upper left', alpha=0.1, figsize=(8,4)):
    rgblist = sns.color_palette('viridis', len(datalist))
    plt.figure(figsize=figsize)
    for idx, data in enumerate(datalist):
        median, lower, higher = data
        if semilogy:
            plt.semilogy(range(len(median)), median, color=rgblist[idx], label=legendlist[idx])
        else:
            plt.plot(range(len(median)), median, color=rgblist[idx], label=legendlist[idx])
        plt.fill_between(np.array(np.arange(len(median))), median.astype(np.float), 
                        higher.astype(np.float), color=rgblist[idx], alpha=alpha)
    if legendlist is not None:
        plt.legend(loc=loc)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(title)


# In[12]:

regretlist = []
costs_list = []

strat_rearranged = [strategies[2], strategies[3], strategies[5], strategies[1], strategies[0], 'adaptive']
res_rearranged = [list_of_results[2], list_of_results[3], list_of_results[5], list_of_results[1], list_of_results[0]]

for name, result in zip(strat_rearranged, res_rearranged):
    regrets, errors, costs, _, bad_invocations = result
    print(name, "bad_invocations", bad_invocations)
    percent_bad = bad_invocations / trials_per_method * 100
    regretlist.append(get_errorbars(regrets, q=10, percent_bad=percent_bad))
    costs_list.append(get_errorbars(costs, q=10, percent_bad=percent_bad))

regretlist.append(get_errorbars(adapt_regret, q=10))
costs_list.append(get_errorbars(adapt_cost, q=10))

# TODO(@evinitsky) there's a bug in here!!! The plots don't match
sns.set_palette("muted")
plot_list_medquantile(regretlist, 'regret_v_iter.png', legendlist=strat_rearranged, xlabel="Iteration", ylabel="Regret")
del costs_list[-2]
del strat_rearranged[-2]
plot_list_medquantile(costs_list, 'cost_opt_v_iter.png', legendlist=strat_rearranged, xlabel="Iteration",
                      ylabel="Cost Suboptimality", semilogy=True, loc='upper right')


# In[ ]:




