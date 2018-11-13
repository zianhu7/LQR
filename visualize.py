from matplotlib import pyplot as plt

def visualize(file, title, save=False):
    data = []
    with open(file) as f:
        for line in f:
            data.append(float(line.rstrip()))
    x = list(range(1, len(data)+1))
    plt.plot(x, data) 
    plt.title(title)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Mean Reward")
    if save:
        plt.savefig(title+'.png')
    plt.show()


def gaussian_visualize(file, title, save=False):
    x, y = [], []
    with open(file) as f:
        for line in f:
            elem = line.split(' ')
            x.append(elem[0])
            y.append(elem[1])
    plt.plot(x, y) 
    plt.title(title, fontsize=16)
    plt.xlabel("Number of Rollouts")
    plt.ylabel("Proportion of Finding Stable Controller")
    if save:
        plt.savefig(title+'.png')
    plt.show()

if __name__ == '__main__':
    f2 = 'rewards_stability_R3_120_5.txt'
    visualize(f2, 'Rewarding Stability: 1.0-1.5 eigenvalue range (actual)', save=True)




