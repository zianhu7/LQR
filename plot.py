import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#3D plotting
def plot_trajectory(states, inputs, idx, traj_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for state in states:
        ax.scatter(state[0],state[1],state[2],c='r',marker='o')
    for inpt in inputs:
        ax.scatter(inpt[0],inpt[1],inpt[2],c='b',marker='^')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig('plot_{}_{}.png'.format(idx, traj_type))
    plt.close(fig)
