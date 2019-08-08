"""These utils are adapted from https://github.com/modestyachts/robust-adaptive-lqr/blob/master/python/utils.py"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are as sda
import scipy


def sda_estimate(A, B, Q, R):
    """Solve the discrete algebraic ricatti equation to compute the optimal feedback. """
    X = sda(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A
    return -K


def spectral_radius(A):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    return max(np.abs(np.linalg.eigvals(A)))


def solve_discrete_lyapunov(A, Q, method=None):
    """Solve A^T P A - P + Q = 0
    """

    # newer versions of scipy solve A P A^T - P + Q = 0,
    # while older ones solve A^T P A - P + Q = 0. I do not
    # remember exactly which version of scipy made the change.

    # I am going to assume you have the newer version installed.
    # If the assertion below fails, please add an if statement
    # that branches on your version.

    P = scipy.linalg.solve_discrete_lyapunov(A.T, Q, method)

    assert np.allclose(A.T.dot(P).dot(A) - P, -Q)

    return P


def LQR_cost(A, B, K, Q, R, sigma_w):
    """Compute infinite time horizon average LQR cost.
    Returns 1e6 if A+BK is not stable
    """

    L = A + B.dot(K)
    if spectral_radius(L) >= 1:
        return 1e6

    M = Q + K.T.dot(R).dot(K)

    P = solve_discrete_lyapunov(L, M)

    return (sigma_w ** 2) * np.trace(P)


def estimate_K(self, horizon, A, B):
    """Solve for K recursively. Not used."""
    Q, R = self.Q, self.R
    # Calculate P matrices first for each step
    P_matrices = np.zeros((horizon + 1, Q.shape[0], Q.shape[1]))
    P_matrices[horizon] = Q
    for i in range(horizon - 1, 0, -1):
        P_t = P_matrices[i + 1]
        P_matrices[i] = Q + (A.T @ P_t @ A) - (A.T @ P_t @ B @ np.matmul(inv(R + B.T @ P_t @ B), B.T @ P_t @ A))
    # Hardcoded shape of K, change to inferred shape for diverse testing
    K_matrices = np.zeros((horizon, self.dim, self.dim))
    for i in range(horizon):
        P_i = P_matrices[i + 1]
        K_matrices[i] = -np.matmul(inv(R + B.T @ P_i @ B), B.T @ P_i @ A)
    return K_matrices


def dlqr(A,B,Q=None,R=None):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    if Q is None:
        Q = np.eye(A.shape[0])
    if R is None:
        R = np.eye(B.shape[1])

    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = -scipy.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A), sym_pos=True)

    A_c = A + B.dot(K)
    TOL = 1e-5
    if spectral_radius(A_c) >= 1 + TOL:
        print("WARNING: spectral radius of closed loop is:", spectral_radius(A_c))

    return P, K


def check_observability(A, C):
    """Check that the observability matrix is full rank.
       We check this by using the standard condition that
       [C
        CA
        CA^2
        .
        .
        .
        CA^{n-1}] is full rank
    """
    assert len(A.shape) == 2
    dim = A.shape[0]
    stack = []
    for i in range(dim):
        term = C @ np.linalg.matrix_power(A, i)
        stack.append(term)
    obs_grammian = np.vstack(stack)
    return np.linalg.matrix_rank(obs_grammian) == dim


def check_controllability(A, B):
    """Check that the controllability matrix [B, BA, ..., BA^{n-1}] is full rank"""
    assert len(A.shape) == 2
    dim = A.shape[0]
    stack = []
    for i in range(dim):
        term = B @ np.linalg.matrix_power(A, i)
        stack.append(term)
    grammian = np.hstack(stack)
    return np.linalg.matrix_rank(grammian) == dim


def check_stability(A, B, control):
    '''Confirm that the feedback matrix stabilizes the system'''
    mat = A + B @ control
    return np.any([abs(e) > 1 for e in np.linalg.eigvals(mat)])


def sample_matrix(dim, bound):
    """Returns a random dim x dim matrix with top eigenvalue bounded by bound"""
    return np.random.uniform(low=-bound / dim, high=bound / dim, size=(dim, dim))
