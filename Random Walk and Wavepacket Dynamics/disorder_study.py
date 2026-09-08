import numpy as np
import matplotlib.pyplot as plt
hbar = 1.0
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import hypertiling as ht

import seaborn as sns
import logging
import os
import scipy.sparse as sp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', filename='randomwalk_disordered.log')

logger = logging.getLogger(__name__)

PLOT_DIR = "plots_disordered"
os.makedirs(PLOT_DIR, exist_ok=True)
Nds = 100

def hyperbolic_adjacency_sparse(p, q, n):
    T = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')

    size = len(T)
    A = lil_matrix((size, size), dtype=np.float64)

    for i in range(size):
        neighbors = find_radius_optimized_single(T, i, radius=None, eps=1e-5)
        for j in neighbors:
            A[i, j] = 1
            A[j, i] = 1

    H_clean = csr_matrix(-1.0 * A)
    
    return H_clean, T  # return tiling too

def add_disorder(H_clean, T, W):
    rng = np.random.default_rng()
    size = len(T)
    disorder_vector = rng.uniform(-W/2.0, W/2.0, size=size)
    H_disordered = H_clean + sp.diags(disorder_vector, format='csr')
    return H_disordered

def timeEvolve_sparse(psi0, H, t):
    #Trace: timeEvolve_sparse -> simulate_random_walk_sparse -> std_dev_with_t_sparse -> p_q_slope
    psi_t = expm_multiply((-1j * H * t / hbar), psi0)
    psi_t /= np.sqrt(np.vdot(psi_t, psi_t))
    E_t = np.vdot(psi_t, H @ psi_t)
    return psi_t, np.real(E_t)


def simulate_random_walk_sparse(T, t, H):
    num_sites = H.shape[0]

    psi0 = np.zeros(num_sites, dtype=complex)
    site = 0  # start at origin
    psi0[site] = 1.0

    psi_t, E_t = timeEvolve_sparse(psi0, H, t)

    prob = np.abs(psi_t)**2
    prob /= prob.sum()

    return prob, E_t


# --- Standard deviation vs time ---
def std_dev_with_t_sparse(p, q, n, t_max, H, T):
    times = np.arange(0, t_max, 0.05)
    stds = []
    energies = []

    # Build geometry + Hamiltonian
    # A, T = hyperbolic_adjacency_sparse(p, q, n)
    # H = -1.0 * A  # tight-binding Hamiltonian

    # Precompute coordinates
    coords = [T.get_center(i) for i in range(len(T))]
    site = 0

    # Precompute squared distances
    r2 = np.zeros(len(coords))
    for i in range(len(coords)):
        d = ht.distance.poincare_distance(coords[i], coords[site])
        r2[i] = d**2

    for t in times:
        prob, E_t = simulate_random_walk_sparse(T, t, H)

        sigma = np.sqrt(np.sum(prob * r2))
        stds.append(sigma)
        energies.append(E_t)

    return times, stds, energies


from scipy.stats import linregress

def linear_region_study(times, stds, r2_threshold=0.99):
    times = np.array(times)
    stds = np.array(stds)
    
    best_slope = None
    best_intercept = None
    limit_idx = 2
    
    for i in range(3, len(times)):
        slope, intercept, r_value, p_value, std_err = linregress(times[:i], stds[:i])
        r2 = r_value**2
        
        if r2 >= r2_threshold:
            best_slope = slope
            best_intercept = intercept
            limit_idx = i - 1
        else:
            break
    
    if best_slope is None:
        # Nothing cleared the threshold — fall back to the first two points
        slope, intercept, r_value, p_value, std_err = linregress(times[:2], stds[:2])
        best_slope, best_intercept = slope, intercept
        limit_idx = 1

    return limit_idx, times[limit_idx], best_slope, best_intercept


#Now, we sweep across a {p, q} grid and compute the slopes
def p_q_slope(p, q, n=4, t_max=7):
    # Initialize an array of NaNs to store the slopes
    # Sized to allow direct indexing: slopes[p, q]
    # slopes = np.full((p_max+1, q_max+1), np.nan)
    # lin_times = np.full((p_max+1, q_max+1), np.nan)
    # energies = np.full((p_max+1, q_max+1, len(np.arange(0, t_max, 0.05))), np.nan)
    energies = []
    stds = []
    times = []

    # Ensure the geometry is strictly hyperbolic
    if (p - 2) * (q - 2) > 4:
        try:
            H_clean, T = hyperbolic_adjacency_sparse(p, q, 4)

            for _ in tqdm(range(Nds)):
                H_disordered = add_disorder(H_clean, T, 40.0)
                time, std, e_t = std_dev_with_t_sparse(p, q, 4, t_max, H_disordered, T)
                times.append(time)

                energies.append(e_t)
                stds.append(std)

            # Average over realisations
            ave_energy = np.mean(energies, axis=0)
            ave_std = np.mean(stds, axis=0)
            ave_time = np.mean(times, axis=0)

            _, lin_time, m, _ = linear_region_study(ave_time, ave_std)

            # --- plot spread vs time and save ---
            plt.figure()
            plt.plot(ave_time, ave_std, marker='o')
            plt.xlabel("t")
            plt.ylabel("std dev of spread")
            plt.title(f"Spread vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"spread_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()
        
            plt.figure()
            plt.plot(ave_time, ave_energy, marker='o')
            plt.xlabel("t")
            plt.ylabel("energy profile")
            plt.title(f"Energy vs t (p={p}, q={q}, n={n})")
            plt.savefig(os.path.join(PLOT_DIR, f"energy_vs_t_p{p}_q{q}_n{n}.png"))
            plt.close()

            logger.info(f"Computed slope for p={p}, q={q}: {m}, linear time: {lin_time}")
            return m, lin_time, ave_energy
            
        except Exception as e:
            logger.error(f"Simulation failed for p={p}, q={q} due to: {e}")
            pass # Leaves the value as NaN

    return None


if __name__ == "__main__":
    slopes, lin_times, energies = p_q_slope(4, 5)