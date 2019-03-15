import matplotlib.pyplot as plt

def plot_stability_compare(no_gauss_fname,  gauss_fname):
    num_iters, stabilities  = dict((k,0) for k in range(6, 21)), dict((k,0) for k in range(6, 21))
    for line in open(no_gauss_fname, 'r'):
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

    num_iters, stabilities = dict((k, 0) for k in range(6, 21)), dict((k, 0) for k in range(6, 21))
    for line in open(gauss_fname, 'r'):
        data = line.strip().split(' ')
        num_iter, stability = int(data[0]), bool(data[2])
        num_iters[num_iter] += 1
        stabilities[num_iter] += stability

    rollout_len, percent_stable = [], []
    for k, v in stabilities.items():
        if num_iters[k]:
            rollout_len.append(k)
            percent_stable.append(v / num_iters[k])

    # fig, ax = plt.subplots()
    plt.plot(rollout_len[2:], percent_stable[2:])
    #y_ticks = np.arange(10e-2,10e2,1)
    #ax.set_ylim([0,10e2])
    #ax.set_yticklabels(y_ticks)
    plt.title("LQR Stabilization Percentage")
    plt.xlabel("Rollout length")
    plt.ylabel("Percent Stable")
    plot_name = no_gauss_fname[:-4] + "_stability.png"
    plt.legend(['Input Synthesis', 'Random Actions'])
    plt.savefig(plot_name)
    plt.show()