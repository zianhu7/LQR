import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math

class LQR_Env(gym.Env)

    def __init__(self):
        #Normalize Q, R
        self.rScaling = 1
        self.dim = 2
        self.generate_system()
        self.action_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.dim,))
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.dim,))
        #Track trajectory
        self.states, self.inputs = [], []
        self.state = None

    def generate_system(self):
        #Make generate_system configurable/randomized
        self.Q = np.eye(self.dim)
        self.R = self.rScaling * np.eye(self.dim)
        self.eigv = np.random.uniform(low=1.0, high=7.0, size=self.dim)
        #Generate A, B matrices
        P = self.rvs(self.dim)
        self.A = P @ self.eigv*np.eye @ P.T
        #Test basic case of B=[[0,0][0,1]]
        self.B = np.zeros((self.dim,self.dim))
        self.B[-1][-1] = 1

    def step(self, u):
        new_state = self.A @ self.state + self.B @ u
        all_zeros = not np.any(new_state)
        if all_zeros or self.max_episode_steps == self.timestep:
            done = True
        self.state = new_state
        self.states.append(self.state)
        self.inputs.append(u)
        self.timestep += 1
        if done: reward = self.calculate_reward()
        else: reward = 0
        return self.state, reward, done, {}

    def reset(self):
        #Assumes reset is called right after __init__
        self.timestep = 0
        rand_values = np.random.randint(low=1, high=100, size=self.dim)
        norm_factor = np.sqrt(sum([e**2 for e in rand_values]))
        self.state = norm_factor * rand_values.reshape((self.dim, ))
        self.states.append(self.state)
        self.generate_system()
        return np.array(self.state)

###############################################################################################

    #Generate random orthornormal matrix of given dim
    def rvs(dim):
         random_state = np.random
         H = np.eye(dim)
         D = np.ones((dim,))
         for n in range(1, dim):
             x = random_state.normal(size=(dim-n+1,))
             D[n-1] = np.sign(x[0])
             x[0] -= D[n-1]*np.sqrt((x*x).sum())
             # Householder transformation
             Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
             mat = np.eye(dim)
             mat[n-1:, n-1:] = Hx
             H = np.dot(H, mat)
             # Fix the last sign such that the determinant is 1
         D[-1] = (-1)**(1-(dim % 2))*D.prod()
         # Equivalent to np.dot(np.diag(D), H) but faster, apparently
         H = (D*H.T).T
         return H

    def ls_estimate(self):
        #NOTE: 2*self.dim is baked into the assumption that A,B are square of shape (self.dim, self.dim)
        X, Z = np.zeros((self.max_episode_steps, self.dim)), np.zeros((self.max_episode_steps, 2*self.dim))
        for i in range(self.timestep):
            #x_idx starts at tstep 1
            x_idx, z_idx = i+1, i
            X[i] = self.states[x_idx]
            z_layer = np.hstack([self.states[i], self.inputs[i]])
            Z[i] = z_layer
        theta = np.inv(Z.T@Z)@(Z.T@X)
        #Reshape ensures that for the self.dim=1 case it is still in a matrix form to ensure consistency
        A, B = theta[:,:self.dim].reshape((self.dim, -1)), theta[:,self.dim:].reshape((self.dim, -1))
        return A, B

    def estimate_K(self, A, B):
        Q, R = self.Q, self.R
        #Calculate P matrices first for each step
        P_matrices = np.zeros((self.max_episode_steps+1, Q.shape[0], Q.shape[1]))
        P_matrices[self.max_episode_steps] = Q
        for i in range(self.max_episode_steps-1, 0, -1):
            P_t = P_matrices[i+1]
            P_matrices[i] = Q + (A.T@P_t@A)-(A.T@P_t@B @ np.matmul(np.inv(R+B.T@P_t@B), B.T@P_t@A))
        #Hardcoded shape of K, change to inferred shape for diverse testing
        K_matrices = np.zeros((self.max_episode_steps, self.dim, self.dim))
        for i in range(self.max_episode_steps):
            P_i = P_matrices[i+1]
            K_matrices[i] = -np.matmul(np.inv(R+B.T@P_i@B), B.T@P_i@A)
        return K_matrices

    def calculate_reward(self):
        #Assumes termination Q is same as regular Q
        Q, R, A, B = self.Q, self.R, self.A, self.B
        K_hat = self.estimate_K(self.ls_estimate())
        K_true = self.estimate_K(self.A, self.B)
        r_true, r_hat = 0, 0
        #Evolve trajectory based on computing input using both K
        state_true, state_hat = self.states[0], self.states[0]
        for i in range(self.max_episode_steps):
            #Update r_hat
            u_hat = K_hat[i] @ state_hat
            r_hat += state_hat.T @ Q @ state_hat + u_hat.T @ R @ u_hat
            state_hat = A @ state_hat + B @ u_hat
            #Update r_true
            u_true = K_true[i] @ state_true
            r_true += state_true.T @ Q @ state_true + u_true.T @ R @ u_true
            state_true = A @ state_true + B @ u_true
        r_hat += state_hat.T @ Q @ state_hat
        r_true += state_true.T @ Q @ state_true
        #Negative to turn into maximization problem for RL
        return -math.abs(r_hat - r_true)
