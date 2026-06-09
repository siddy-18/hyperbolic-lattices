import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar

def grid_adjacency_periodic(n):
    N = n * n
    A = np.zeros((N, N), dtype=int)

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            u = idx(i, j)

            neighbors = [
                ((i - 1) % n, j),
                ((i + 1) % n, j),
                (i, (j - 1) % n),
                (i, (j + 1) % n),
            ]

            for ni, nj in neighbors:
                v = idx(ni, nj)
                A[u, v] = 1

    return A


def grid_adjacency_open(n):
    N = n*n
    A = np.zeros((N,N), dtype = int)

    def idx(i, j):
        return i*n + j
    
    def exists(i, j):
        return i >= 0 and i < n and j >= 0 and j < n
    
    for i in range(n):
        for j in range(n):
            u = idx(i, j)
            neighbors = []
            if(exists(i+1, j)): neighbors.append(idx(i+1, j))
            if(exists(i-1, j)): neighbors.append(idx(i-1, j))
            if(exists(i, j+1)): neighbors.append(idx(i, j+1))
            if(exists(i, j-1)): neighbors.append(idx(i, j-1))

            for k in neighbors:
                A[u, k] = 1

    return A


def eigenthings(H):
    return np.linalg.eigh(H)


def timeEvolve(state, eigvals, eigvecs, t):
    phases = np.exp(-1j * eigvals * t / hbar)
    return eigvecs @ (phases * (eigvecs.conj().T @ state))


def simulate_random_walk(n, t, eigvals, eigvecs):
    num_sites = n * n

    psi0 = np.zeros(num_sites, dtype=complex)
    cx, cy = n // 2, n // 2
    center_index = cx * n + cy
    psi0[center_index] = 1.0

    psi_t = timeEvolve(psi0, eigvals, eigvecs, t)

    probabilities = np.abs(psi_t)**2
    probabilities /= probabilities.sum()

    return probabilities.reshape((n, n))



# --- Standard deviation vs time ---
def std_dev_with_t(n, t_max):
    times = np.arange(t_max)
    stds = []

    cx = n // 2

    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

    H = -1.0 * grid_adjacency_open(n)
    eigvals, eigvecs = eigenthings(H)

    for t in times:
        prob = simulate_random_walk(n, t, eigvals, eigvecs)

        # ✅ Use x-direction variance (cleaner ballistic signal)
        mean_x = np.sum(prob * xs)
        var_x = np.sum(prob * (xs - mean_x)**2)
        stds.append(np.sqrt(var_x))

    plt.plot(times, stds)
    plt.xlabel("t")
    plt.ylabel("σ(t)")
    plt.title("Spread of quantum walk on periodic lattice")
    plt.show()


std_dev_with_t(100, 15)