from matplotlib import pyplot as plt
import numpy as np
fname = '../genR3_rel_reward_recht.txt'

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
plt.savefig("relative_cost_subopt_recht.png")
plt.show()

