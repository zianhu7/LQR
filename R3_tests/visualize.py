import json
from matplotlib import pyplot as plt

fpath = "/home/zian/Desktop/research/LQR/R3_tests/gen_full_ls_3/GenLQR_tests/"
fnames = ["8e4_full/result.json", "8e4_consts/result.json"]
#fnames = ["./gen_es_R3/8e4/result.json"]

for i in range(len(fnames)):
    f = fnames[i]
    ftot = fpath + f
    data = []
    for line in open(ftot, 'r'):
        data.append(json.loads(line))
    rewards = [data[i]['episode_reward_mean'] for i in range(len(data))]
    x = list(range(1, len(rewards)+1))
    plt.plot(x, rewards)
    plt.title(f)
    #plt.savefig("reward_es_8e4.png")
    plt.show()



