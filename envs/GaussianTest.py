import numpy as np
from numpy.linalg import inv
from plot import plot_trajectory
from scipy.linalg import solve_discrete_are

class LQRExp():

    def __init__(self, exp_params):
        #Normalize Q, R
        self.params = exp_params
        self.horizon, self.exp_length = self.params["horizon"], self.params["exp_length"]
        self.rScaling = 1
        self.dim = 3
        self.generate_system()
        #Track trajectory
        self.num_exp = int(self.horizon / self.exp_length)
        self.states, self.inputs = [[] for i in range(self.num_exp)], [[] for i in range(self.num_exp)]
        self.state = None


    def generate_system(self):
        #Make generate_system configurable/randomized
        self.Q = 0.001*np.eye(self.dim)
        self.R = self.rScaling * np.eye(self.dim)
        self.eigv = np.random.uniform(low=-1.2, high=1.2, size=self.dim)
        #Ensure PD, stable A
        while np.count_nonzero(self.eigv) != self.dim:
            self.eigv = np.random.uniform(low=-1.2, high=1.2, size=self.dim)
        #Generate A, B matrices
        P = self.rvs(self.dim)
        #self.A = P @ self.eigv*np.eye(self.dim) @  P.T
        self.A = np.array([[1.01,0.01,0],[0.01,1.01,0],[0,0.01,1.01]])
        #Test basic case of B=[[0,0][0,1]]
        #self.B = np.zeros((self.dim,self.dim))
        #self.B[-1][-1] = 1
        self.B = np.eye(self.dim)

    def step(self, u):
        mu = [0]*self.dim
        cov = np.eye(self.dim)
        noise = np.random.multivariate_normal(mu, cov)
        new_state = self.A @ self.state + self.B @ u + noise
        self.state = new_state
        self.states[self.curr_exp].append(self.state)
        self.inputs[self.curr_exp].append(u)
        self.timestep += 1
        completion = False
        if self.horizon == self.timestep:
            completion = True
        if (self.timestep % self.exp_length == 0) and (not completion):
            self.reset_exp()
        if completion:
            reward, hat_stability, true_stability = self.calculate_reward()
            stability_info = {"hat_stable": 1-int(hat_stability), "true_stable": 1-int(true_stability)}
        else: reward, stability_info = 0, {}
        return self.state, reward, completion, stability_info

    def reset_exp(self):
        #Not global reset, only for each experiment
        self.curr_exp += 1
        rand_values = np.random.randint(low=1, high=100, size=self.dim)
        norm_factor = np.sqrt(sum([e**2 for e in rand_values]))
        self.state = (1 / norm_factor) * rand_values
        self.states[self.curr_exp].append(self.state)

    def reset(self):
        #Assumes reset is called right after __init__
        self.timestep = 0
        self.curr_exp = 0
        rand_values = np.random.randint(low=1, high=100, size=self.dim)
        norm_factor = np.sqrt(sum([e**2 for e in rand_values]))
        self.state = (1 / norm_factor) * rand_values
        self.states[self.curr_exp].append(self.state)
        self.generate_system()
        return np.array(self.state)

###############################################################################################

    #Generate random orthornormal matrix of given dim
    def rvs(self, dim):
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
        X, Z = np.zeros((self.num_exp, self.dim)), np.zeros((self.num_exp, 2*self.dim))
        for i in range(self.num_exp):
            j = self.exp_length - 1
            #x_idx starts at tstep 1
            x_idx, z_idx = j+1, j
            X[i] = self.states[i][x_idx]
            z_layer = np.hstack([self.states[i][z_idx], self.inputs[i][z_idx]])
            Z[i] = z_layer
        theta = (inv(Z.T@Z)@(Z.T@X)).T
        #Reshape ensures that for the self.dim=1 case it is still in a matrix form to ensure consistency
        A, B = theta[:,:self.dim].reshape((self.dim, -1)), theta[:,self.dim:].reshape((self.dim, -1))
        return A, B

    #def ls_estimate(self):
    #    #NOTE: 2*self.dim is baked into the assumption that A,B are square of shape (self.dim, self.dim)
    #    X, Z = np.zeros((self.timestep, self.dim)), np.zeros((self.timestep, 2*self.dim))
    #    for i in range(self.num_exp):
    #        for j in range(self.exp_length):
    #            #x_idx starts at tstep 1
    #            x_idx, z_idx = j+1, j
    #            X[i] = self.states[i][x_idx]
    #            z_layer = np.hstack([self.states[i][z_idx], self.inputs[i][z_idx]])
    #            Z[i] = z_layer
    #    theta = (inv(Z.T@Z)@(Z.T@X)).T
    #    #Reshape ensures that for the self.dim=1 case it is still in a matrix form to ensure consistency
    #    A, B = theta[:,:self.dim].reshape((self.dim, -1)), theta[:,self.dim:].reshape((self.dim, -1))
    #    return A, B

    def estimate_K(self, horizon, A, B):
        Q, R = self.Q, self.R
        #Calculate P matrices first for each step
        P_matrices = np.zeros((horizon+1, Q.shape[0], Q.shape[1]))
        P_matrices[horizon] = Q
        for i in range(horizon-1, 0, -1):
            P_t = P_matrices[i+1]
            P_matrices[i] = Q + (A.T@P_t@A)-(A.T@P_t@B @ np.matmul(inv(R+B.T@P_t@B), B.T@P_t@A))
        #Hardcoded shape of K, change to inferred shape for diverse testing
        K_matrices = np.zeros((horizon, self.dim, self.dim))
        for i in range(horizon):
            P_i = P_matrices[i+1]
            K_matrices[i] = -np.matmul(inv(R+B.T@P_i@B), B.T@P_i@A)
        return K_matrices

    def solve_DARE(self, A, B):
        Q, R = self.Q, self.R
        X = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T@X@B)@B.T@X@A
        return -K

    def calculate_reward(self):
        #Assumes termination Q is same as regular Q
        Q, R, A, B = self.Q, self.R, self.A, self.B
        A_est, B_est = self.ls_estimate()
        K_t =  self.solve_DARE(A, B)
        K_h = self.solve_DARE(A_est, B_est)
        r_true, r_hat = 0, 0
        #Evolve trajectory based on computing input using both K
        #synth_traj, true_traj = [[],[]], [[],[]]
        #for i in range(self.num_exp):
            #state_true, state_hat = self.states[i][0], self.states[i][0]
            #true_traj[0].append(state_true); synth_traj[0].append(state_hat)
        choice_idx = np.random.choice(self.num_exp, 1)[0]
        state_true, state_hat = self.states[choice_idx][0], self.states[choice_idx][0]
        for j in range(self.exp_length):
            #Update r_hat
            u_hat = K_h @ state_hat
            r_hat += state_hat.T @ Q @ state_hat + u_hat.T @ R @ u_hat
            state_hat = A @ state_hat + B @ u_hat
            #Update r_true
            u_true = K_t @ state_true
            r_true += state_true.T @ Q @ state_true + u_true.T @ R @ u_true
            state_true = A @ state_true + B @ u_true
        r_hat += state_hat.T @ Q @ state_hat
        r_true += state_true.T @ Q @ state_true
        #Negative to turn into maximization problem for RL
        reward = -abs(r_hat-r_true)
        hat_stability = self.check_instability(K_h)
        true_stability = self.check_instability(K_t)
        return reward, hat_stability, true_stability

    def check_controllability(self):
        dim = self.dim
        stack = []
        for i in range(dim - 1):
            term = self.B @ np.linalg.matrix_power(self.A, i)
            stack.append(term)
        gramian = np.hstack(stack)
        return np.linalg.matrix_rank(gramian) == dim

    def check_instability(self, control):
        mat = self.A + self.B @ control
        return np.any([abs(e) > 1 for e in np.linalg.eigvals(mat)])

    def run_exp(self):
        self.reset()
        mean = [0]*self.dim
        cov = np.eye(self.dim)
        min_reward = 0
        for _ in range(self.horizon):
            input = np.random.multivariate_normal(mean, cov)
            _, reward, _, stability_info = self.step(input)
            if reward < min_reward: min_reward = reward
        return min_reward, stability_info

if __name__ == "__main__":
    num_rollouts = range(10, 101, 5)
    rewards, stability = {}, {}
    stats = {}
    for nr in num_rollouts:
        exp_length = 6
        exp_params = {"horizon":exp_length*nr, "exp_length":exp_length}
        hat_stable, true_stable, total_reward = 0, 0, 0
        for _ in range(100):
            exp = LQRExp(exp_params)
            reward, stability = exp.run_exp()
            hat_stable += stability["hat_stable"]
            true_stable += stability["true_stable"]
            total_reward += reward
        stats[exp_length*nr] = [hat_stable/100, true_stable/100, total_reward/100]
    import ipdb;ipdb.set_trace()
    with open('gaussian_stability.txt', 'w') as f:
        for k,v in stats.items():
            f.write(str(k) + ' ' + str(v[0]) + ' ' + str(v[1]) + '\n')
    #with open('gaussian_results.txt', 'w') as f:
        #for k,v in stats.items():
            #f.write(str(k)+" ")
            #for e in v:
                #f.write(str(e) + " ")
            #f.write("\n")

        #'''
        #if min_val < -10e8:
            #plot_trajectory(exp.true_trajectory[0], exp.true_trajectory[1], idx, 't')
            #plot_trajectory(exp.synth_trajectory[0], exp.synth_trajectory[1], idx, 's')
            #idx += 1
            #irr_states[0].append(exp.true_trajectory[0])
            #irr_states[1].append(exp.synth_trajectory[0])
            #irr_inputs[0].append(exp.true_trajectory[1])
            #irr_inputs[1].append(exp.synth_trajectory[1])
        #'''
