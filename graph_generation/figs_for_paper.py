import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import math
import pandas as pd

font = {'size': 18}
rc('font', **font)


def stability_plot(fname, marker, color):
    num_iters, stabilities = dict((k, 0) for k in range(6, 21)), dict((k, 0) for k in range(6, 21))
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
    for k, v in stabilities.items():
        if num_iters[k]:
            rollout_len.append(k)
            percent_stable.append(v / num_iters[k])

    # fig, ax = plt.subplots()
    plt.plot(rollout_len, np.array(percent_stable) * 100, linestyle='-', marker=marker, color=color)


def plot_subopt(fname, marker, color):
    num_iters = dict((k, 0) for k in range(6, 21))
    rel_rewards = dict((k, []) for k in range(6, 21))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        num_iter, rel_reward = int(data[0]), float(data[1])
        num_iters[num_iter] += 1
        rel_rewards[num_iter].append(rel_reward)

    rollout_len, median_subopt = [], []
    for k, v in rel_rewards.items():
        if num_iters[k]:
            rollout_len.append(k)
            median_subopt.append(np.median(v))

    plt.plot(rollout_len, median_subopt, linestyle='-', marker=marker, color=color)


def plot_stable_txt(fname, marker, color):
    data = np.loadtxt(fname, delimiter=',')
    plt.plot(data[:, 0], np.array(data[:, 1]) * 100, linestyle='-', marker=marker, color=color)


def plot_rel_cost_txt(fname, marker, color):
    data = np.loadtxt(fname, delimiter=',')
    plt.plot(data[:, 0], data[:, 1], linestyle='-', marker=marker, color=color)


def plot_opnorms(fname, iters, marker, color, eigv_high, dim):
    '''Plot the operatornorm as function of horizon or top eigenvalue. The latter occurs if iters = True'''
    top_eigs, opnorm_A, opnorm_B = dict((k, 0) for k in range(eigv_high * dim)), \
                                   dict((k, []) for k in range(eigv_high * dim)), \
                                   dict((k, []) for k in range(eigv_high * dim))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        top_eig, ep_A, ep_B = math.ceil(float(data[0])), float(data[1]), float(data[2])
        # data is sparse outside of that point so the performance is not reflective
        if top_eig < 40:
            top_eigs[top_eig] += 1
            opnorm_A[top_eig].append(ep_A)
            opnorm_B[top_eig].append(ep_B)

    eig_range, median_EA, median_EB = [], [], []
    for k, v in opnorm_A.items():
        if top_eigs[k]:
            eig_range.append(k)
            median_EA.append(np.median(v))
            median_EB.append(np.median(opnorm_B[k]))

    plt.plot(eig_range, median_EA, linestyle='-', marker=marker[0], color=color[0])
    plt.plot(eig_range, median_EB, linestyle='-', marker=marker[1], color=color[1])


def plot_generalization_rewards(fname, eig_high, dim, marker, color):
    top_eigs, rel_reward, stability = dict((k, 0) for k in range(eig_high * dim)), \
                                      dict((k, []) for k in range(eig_high * dim)), \
                                      dict((k, 0) for k in range(eig_high * dim))
    for line in open(fname, 'r'):
        data = line.strip().split(' ')
        top_eig, reward, stable = math.ceil(float(data[0])), float(data[1]), bool(data[2])
        # data is sparse outside of that point so the performance is not reflective
        if top_eig < 40:
            top_eigs[top_eig] += 1
            rel_reward[top_eig].append(reward)
            stability[top_eig] += stable

    eig_range, reward, percent_stable = [], [], []
    for k, v in rel_reward.items():
        if top_eigs[k]:
            eig_range.append(k)
            reward.append(np.median(v))
            percent_stable.append(stability[k] / top_eigs[k])

    plt.plot(eig_range, reward, linestyle='-', marker=marker, color=color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help="If true, the figures are saved")
    args = parser.parse_args()
    # # first construct the stability plot
    plt.figure(figsize=(14, 10))
    # y_ticks = np.arange(10e-2,10e2,1)
    # ax.set_ylim([0,10e2])
    # ax.set_yticklabels(y_ticks)
    stability_plot('output_files/dim3_full_constrained_eval_eval_matrix_benchmark.txt', 'o', 'b')
    stability_plot('output_files/dim3_partial_constrained_eval_eval_matrix_benchmark.txt', 'd', 'r')
    stability_plot('output_files/dim3_full_constrained_eval_gauss_eval_matrix_benchmark.txt', '<', 'y')
    stability_plot('output_files/full_ls_fiechter_eval_matrix_benchmark.txt', 'x', 'g')
    plot_stable_txt('output_files/fir_true_stable.txt', '+', 'c')
    plot_stable_txt('output_files/fir_bootstrap_stable.txt', 'x', 'b')
    plt.legend(['LS Controller Full', 'LS Controller Partial', 'Gaussian Full', 'Fiechter',
                'FIR True', 'FIR Bootstrap'])
    plt.title("LQR Stabilization Percentage")
    plt.xlabel("Rollout length", labelpad=10)
    plt.ylabel("Avg. Percent Stable")
    plt.grid(True)
    plot_name = 'output_files/stability_compare.png'
    if args.save:
        plt.savefig(plot_name)
    plt.show()

    ##############################################################################################
    # Generate the suboptimality plots
    plt.figure(figsize=(14, 10))
    plot_subopt('output_files/dim3_full_constrained_eval_eval_matrix_benchmark.txt', 'o', 'b')
    plot_subopt('output_files/dim3_partial_constrained_eval_eval_matrix_benchmark.txt', 'd', 'r')
    plot_subopt('output_files/dim3_full_constrained_eval_gauss_eval_matrix_benchmark.txt', '<', 'y')
    plot_rel_cost_txt('output_files/FIR_true.txt', '+', 'c')
    # plot_rel_cost_txt('output_files/CL_true.txt', 'x', 'k')
    plt.legend(['LS Controller Full', 'LS Controller Partial', 'Gaussian Full',
                'FIR True', 'FIR Bootstrap'])
    plt.title("LQR Relative Cost Suboptimality")
    plt.xlabel("Rollout length", labelpad=20)
    plt.ylabel("Median Log Relative Cost Suboptimality")
    plt.yscale('log')
    # formatter = plt.FuncFormatter(log_10_product)
    ax = plt.gca()
    # ax.yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend(['LS Controller Full', 'LS Controller Partial', 'Gaussian Full',
                'FIR True'])
    plot_name = 'output_files/rel_cost_compare.png'
    if args.save:
        plt.savefig(plot_name)
    plt.show()

    ###############################################################################################
    # Generate the generalization plots
    # Generate the plot of reward as function of top eigenvalue
    plt.figure(figsize=(14, 10))
    plot_generalization_rewards('output_files/dim3_full_constrained_gen_eigv_generalization.txt', 20,
                                3, 'o', 'b')
    plot_generalization_rewards('output_files/dim3_partial_constrained_gen_eigv_generalization.txt', 20,
                                3, 'd', 'r')
    plot_generalization_rewards('output_files/dim3_full_constrained_gen_gauss_eigv_generalization.txt',
                                20, 3, '>', 'g')

    plt.title('Top eigenvalue vs. log median relative reward')
    plt.xlabel('Top eigenvalue', labelpad=20)
    plt.ylabel("Log median relative Reward")
    plt.legend(['LS Controller Full', 'LS Controller Partial', 'Gaussian Full'])
    plt.yscale('log')
    # formatter = plt.FuncFormatter(log_10_product)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plot_name = "figures/lqr_cost_v_eigenval.png"
    if args.save:
        plt.savefig(plot_name)
    plt.show()

    ###############################################################################################
    # Generate the operator norm plots
    plt.figure(figsize=(14, 10))
    plot_opnorms('output_files/dim3_full_constrained_gen_opnorm_error.txt', False,
                 ['o', 'd'], ['b', 'r'], 20, 3)
    plot_opnorms('output_files/dim3_partial_constrained_gen_opnorm_error.txt', False,
                 ['<', '>'], ['y', 'g'], 20, 3)
    plot_opnorms('output_files/dim3_full_constrained_gen_gauss_opnorm_error.txt',
                 False, ['+', 'x'], ['c', 'k'], 20, 3)

    plt.title('Operator norm vs. top eigenvalue')
    plt.xlabel('Top eigenvalue', labelpad=20)
    plt.ylabel("Median Log Operator Norm")
    plt.yscale('log')
    plt.legend([r'LS Controller Full $\epsilon_A$', r'LS Controller Full $\epsilon_B$',
                r'LS Controller Partial $\epsilon_A$', r'LS Controller Partial $\epsilon_B$',
                r'LS Gaussian Full $\epsilon_A$', r'LS Gaussian Full $\epsilon_B$'])
    # plt.legend([r'$\epsilon_A$', r'$\epsilon_B$'])
    plot_name = "figures/eig_v_operator_norm.png"
    if args.save:
        plt.savefig(plot_name)
    plt.show()

    # plt.figure(figsize=(14, 10))
    # plt.title("Operator Norm Error on Eval System")
    # xlabel = "Rollout length"
    # plt.xlabel(xlabel, labelpad=20)
    # plt.ylabel("Median Log Operator Norm Error")
    # plt.yscale('log')
    # plt.grid(True)
    # plt.legend([r'$\epsilon_A$', r'$\epsilon_B$'])
    # plot_name = fname[:-4] + ".png"
    # plt.savefig(plot_name)
    # plt.show()

    #################################################################################################
    # Generate the training curve
    df = pd.read_csv('../trained_policies/partial_constrained_R3/progress.csv')
    training_iters = df['training_iteration']
    mean_rew = df['episode_reward_mean']
    plt.figure(figsize=(14, 10))
    plt.plot(training_iters, mean_rew)
    plt.title("Avg. Reward vs. training iterations")
    plt.xlabel('Training iteration')
    plt.ylabel('Avg. Reward')
    plot_name = "figures/training_curve.png"
    if args.save:
        plt.savefig(plot_name)
    plt.show()
