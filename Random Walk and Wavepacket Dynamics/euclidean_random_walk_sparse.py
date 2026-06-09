import numpy as np
import matplotlib.pyplot as plt
hbar = 1.0
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply


def grid_adjacency_periodic_sparse(n):
    N = n * n
    A = lil_matrix((N, N), dtype=np.float64)

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

    return csr_matrix(A)  # efficient format


def timeEvolve_sparse(psi0, H, t):
    return expm_multiply((-1j * H * t / hbar), psi0)


def simulate_random_walk_sparse(n, t, H):
    num_sites = n * n

    psi0 = np.zeros(num_sites, dtype=complex)
    cx, cy = n // 2, n // 2
    psi0[cx * n + cy] = 1.0

    psi_t = timeEvolve_sparse(psi0, H, t)

    prob = np.abs(psi_t)**2
    prob /= prob.sum()

    return prob.reshape((n, n))


# --- Standard deviation vs time ---
def std_dev_with_t_sparse(n, t_max):
    times = np.arange(t_max)
    stds = []

    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

    H = -1.0 * grid_adjacency_periodic_sparse(n)  # sparse Hamiltonian

    for t in times:
        prob = simulate_random_walk_sparse(n, t, H)
        cx, cy = n // 2, n // 2
        dx = xs - cx
        dy = ys - cy
        r2 = dx**2 + dy**2

        sigma = np.sqrt(np.sum(prob * r2))
        stds.append(sigma)

    plt.plot(times, stds)
    plt.xlabel("t")
    plt.ylabel("σ(t)")
    plt.title("Quantum walk (sparse evolution)")
    plt.show()


std_dev_with_t_sparse(100, 30)