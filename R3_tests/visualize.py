import json
from matplotlib import pyplot as plt

fnames = ['gen_R3/result.json']

for f in fnames:
    data = []
    for line in open(f, 'r'):
        data.append(json.loads(line))
    rewards = [data[i]['episode_reward_mean'] for i in range(len(data))]
    x = list(range(1, len(rewards)+1))
    plt.plot(x, rewards)
    plt.title(f)
    plt.savefig('reward.png')
    plt.show()



