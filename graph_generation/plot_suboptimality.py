from matplotlib import pyplot as plt
from matplotlib import rc
import math
import numpy as np
fname = "../generalization_es_rewards_A.txt"

font = {'size': 18}
rc('font', **font)
def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%1i' % (x)

def plot_subopt(fname):
    num_iters, rel_rewards  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        num_iter, rel_reward = int(data[0]), float(data[1])
        num_iters[num_iter] += 1
        rel_rewards[num_iter] +=  rel_reward

    rollout_len, avg_subopt = [], []
    for k,v in rel_rewards.items():
        if num_iters[k]:
            rollout_len.append(k)
            avg_subopt.append(v/num_iters[k])

    #fig, ax = plt.subplots()
    plt.figure(figsize=(14, 10))

    plt.plot(rollout_len, avg_subopt, linestyle='-', marker='o', color='b')
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("LQR Relative Cost Suboptimality")
    plt.xlabel("Rollout length", labelpad=20)
    plt.ylabel("Log Relative Cost Suboptimality")
    plt.yscale('log')
    formatter = plt.FuncFormatter(log_10_product)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plot_name = fname[:-4] + "_subopt.png"
    plt.savefig(plot_name)
    plt.show()


def plot_stability(fname):
    num_iters, stabilities  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        # FIXME(ev) Hack to support the way other code was writte
        if data[2] == 'False':
            data[2] = 0
        else:
            data[2] = 1
        num_iter, stability = int(data[0]), bool(data[2])
        num_iters[num_iter] += 1
        stabilities[num_iter] += stability

    rollout_len, percent_stable = [], []
    for k,v in stabilities.items():
        if num_iters[k]:
            rollout_len.append(k)
            percent_stable.append(v/num_iters[k])

    #fig, ax = plt.subplots()
    plt.figure(figsize=(14, 10))
    plt.plot(rollout_len, np.array(percent_stable) * 100, linestyle='-', marker='o', color='b')
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("LQR Stabilization Percentage")
    plt.xlabel("Rollout length", labelpad=10)
    plt.ylabel("Percent Stable")
    plt.grid(True)
    plot_name = fname[:-4] + "_stability.png"
    plt.savefig(plot_name)
    plt.show()


def plot_opnorms(fname, iters, env_params=None):
    '''Plot the operatornorm as function of horizon or top eigenvalue. The latter occurs if iters = True'''
    if iters:
        num_iters, opnorm_A, opnorm_B  = dict((k,0) for k in range(6, 50)), \
                                         dict((k,0) for k in range(6, 50)), \
                                         dict((k,0) for k in range(6,50))
        for line in open(fname, 'r'):
            data = line.strip().split(' ')
            num_iter, ep_A, ep_B = int(data[0]), float(data[1]), float(data[2])
            num_iters[num_iter] += 1
            opnorm_A[num_iter] += ep_A
            opnorm_B[num_iter] += ep_B
        idx_nums, avg_EA, avg_EB = [], [], []
        for k, v in opnorm_A.items():
            if num_iters[k]:
                idx_nums.append(k)
                avg_EA.append(v / num_iters[k])
                avg_EB.append(opnorm_B[k] / num_iters[k])

        # fig, ax = plt.subplots()
        plt.figure(figsize=(14, 10))
        plt.plot(idx_nums, avg_EA, linestyle='-', marker='o', color='b')
        plt.plot(idx_nums, avg_EB, linestyle='-', marker='d', color='r')
        # y_ticks = np.arange(10e-2,10e2,1)
        # ax.set_ylim([0,10e2])
        # ax.set_yticklabels(y_ticks)
        plt.title("Operator Norm Error on Eval System")
        xlabel = "Rollout length" if iters else r'$\lambda_{\text{max}}(A)$'
        plt.xlabel(xlabel, labelpad=20)
        plt.ylabel("Log Operator Norm Error")
        plt.yscale('log')
        formatter = plt.FuncFormatter(log_10_product)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(formatter)
        plt.grid(True)
        plt.legend([r'$\epsilon_A$', r'$\epsilon_B$'])
        plot_name = fname[:-4] + ".png"
        plt.savefig(plot_name)
        plt.show()
    else:
        top_eigs, opnorm_A, opnorm_B = dict((k, 0) for k in range(env_params['eigv_high']
                                                                     * env_params['dim'])), \
                                          dict((k, 0) for k in range(env_params['eigv_high']
                                                                     * env_params['dim'])), \
                                          dict((k, 0) for k in range(env_params['eigv_high']
                                                                     * env_params['dim']))
        for line in open(fname, 'r'):
            data = line.strip().split(' ')
            top_eig, ep_A, ep_B = math.ceil(float(data[0])), float(data[1]), float(data[2])
            top_eigs[top_eig] += 1
            opnorm_A[top_eig] += ep_A
            opnorm_B[top_eig] += ep_B

        eig_range, avg_EA, avg_EB = [], [], []
        for k, v in opnorm_A.items():
            if top_eigs[k]:
                eig_range.append(k)
                avg_EA.append(v / top_eigs[k])
                avg_EB.append(opnorm_B[k] / top_eigs[k])

        plt.figure(figsize=(14, 10))
        plt.plot(eig_range, avg_EA, linestyle='-', marker='o', color='b')
        plt.plot(eig_range, avg_EB, linestyle='-', marker='r', color='d')
        plt.title(r'$\text{Operator Norm vs.} \lambda_{\text{max}}(A)$')
        plt.xlabel("r'\lambda_{\text{max}}(A)'", labelpad=20)
        plt.ylabel("Log Operator Norm")
        plt.yscale('log')
        formatter = plt.FuncFormatter(log_10_product)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(formatter)
        plot_name = fname[:-4] + ".png"
        plt.savefig(plot_name)
        plt.show()


def plot_generalization_rewards(fname, eig_high, dim):
    top_eigs, rel_reward, stability = dict((k,0) for k in range(eig_high * dim)), \
                                      dict((k,0) for k in range(eig_high * dim)), \
                                      dict((k,0) for k in range(eig_high * dim))
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

    plt.figure(figsize=(14, 10))
    plt.plot(eig_range, reward, linestyle='-', marker='o', color='b')
    plt.title(r'$\text{Generalization Rewards vs.} \lambda_{\text{max}}(A)$')
    plt.xlabel(r'\lambda_{\text{max}}(A)', labelpad=20)
    plt.ylabel("Log Relative Reward")
    plt.yscale('log')
    formatter = plt.FuncFormatter(log_10_product)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.savefig("output_files/generalization_es_A.png")
    plt.show()
    

if __name__ == "__main__":
    plot_generalization_rewards(fname, 8)
