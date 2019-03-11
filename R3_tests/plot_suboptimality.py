from matplotlib import pyplot as plt
import math
import numpy as np
fname = "../generalization_es_rewards_A.txt"

def plot_subopt(fname):
    num_iters, rel_rewards  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21))
    for line in open(fname, 'r'):
        data = line.split(' ')
        num_iter, rel_reward = int(data[0]), float(data[1])
        num_iters[num_iter] += 1
        rel_rewards[num_iter] +=  rel_reward

    rollout_len, avg_subopt = [], []
    for k,v in rel_rewards.items():
        if num_iters[k]:
            rollout_len.append(k)
            avg_subopt.append(v/num_iters[k])

    #fig, ax = plt.subplots()
    plt.plot(rollout_len[2:], avg_subopt[2:])
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("Gen LQR Relative Cost Suboptimality on Recht System")
    plt.xlabel("Rollout length")
    plt.ylabel("Relative Cost Suboptimality")
    plt.savefig("rel_es_finalLS_recht.png")
    plt.show()


def plot_stability(fname):
    num_iters, stabilities  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        num_iter, stability = int(data[0]), bool(data[2])
        num_iters[num_iter] += 1
        stabilities[num_iter] += stability

    rollout_len, percent_stable = [], []
    for k,v in stabilities.items():
        if num_iters[k]:
            rollout_len.append(k)
            percent_stable.append(v/num_iters[k])

    #fig, ax = plt.subplots()
    plt.plot(rollout_len[2:], percent_stable[2:])
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("Gen LQR Stabilization on Recht System")
    plt.xlabel("Rollout length")
    plt.ylabel("Percent Stable")
    plt.savefig("gen_es_stability_recht.png")
    plt.show()

def plot_opnorms(fname):
    num_iters, opnorm_A, opnorm_B  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6,21))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        num_iter, ep_A, ep_B = int(data[0]), float(data[1]), float(data[2])
        num_iters[num_iter] += 1
        opnorm_A[num_iter] += ep_A
        opnorm_B[num_iter] += ep_B

    rollout_len, avg_EA, avg_EB  = [], [], []
    for k,v in opnorm_A.items():
        if num_iters[k]:
            rollout_len.append(k)
            avg_EA.append(v/num_iters[k])
            avg_EB.append(opnorm_B[k]/num_iters[k])

    #fig, ax = plt.subplots()
    plt.plot(rollout_len, avg_EA)
    plt.plot(rollout_len, avg_EB)
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("Operator Norm Error on Recht System")
    plt.xlabel("Rollout length")
    plt.ylabel("Operator Norm Error")
    plt.legend(['epsilon_A', 'epsilon_B'])
    plt.savefig("gen_es_opnorm_recht.png")
    plt.show()

def plot_generalization_rewards(fname, eig_high):
    top_eigs, rel_reward, stability = dict((k,0) for k in range(eig_high*3)), dict((k,0) for k in range(eig_high*3)), dict((k,0) for k in range(eig_high*3))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        top_eig, reward, stable = math.ceil(float(data[0])), float(data[1]), bool(data[2])
        top_eigs[top_eig] += 1
        rel_reward[top_eig] += reward
        stability[top_eig] += stable

    eig_range, reward, percent_stable = [], [], []
    for k, v in rel_reward.items():
        if top_eigs[k]:
            eig_range.append(k)
            reward.append(v/top_eigs[k])
            percent_stable.append(stability[k]/top_eigs[k])


    plt.plot(eig_range, reward)
    plt.title("Generalization Rewards as a Function of Top Eigenvalue of A")
    plt.xlabel("Maximum Eigenvalue of A")
    plt.ylabel("Relative Reward")
    plt.savefig("generalization_es_A.png")
    plt.show()
    

if __name__ == "__main__":
    plot_generalization_rewards(fname, 8)
