import numpy as np
import matplotlib.pyplot as plt
hbar = 1.0
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

from hypertiling import HyperbolicTiling
from hypertiling.neighbors import find_radius_optimized_single
import hypertiling as ht

import seaborn as sns

def hyperbolic_adjacency_sparse(p, q, n):
    T = HyperbolicTiling(q, p, n, kernel='SRG', center='vertex')

    size = len(T)
    A = lil_matrix((size, size), dtype=np.float64)

    for i in range(size):
        neighbors = find_radius_optimized_single(T, i, radius=None, eps=1e-5)
        for j in neighbors:
            A[i, j] = 1
            A[j, i] = 1

    return csr_matrix(A), T  # return tiling too


def timeEvolve_sparse(psi0, H, t):
    return expm_multiply((-1j * H * t / hbar), psi0)


def simulate_random_walk_sparse(T, t, H):
    num_sites = H.shape[0]

    psi0 = np.zeros(num_sites, dtype=complex)
    site = 0  # start at origin
    psi0[site] = 1.0

    psi_t = timeEvolve_sparse(psi0, H, t)

    prob = np.abs(psi_t)**2
    prob /= prob.sum()

    return prob


# --- Standard deviation vs time ---
def std_dev_with_t_sparse(p, q, n, t_max):
    times = np.arange(t_max)
    stds = []

    # Build geometry + Hamiltonian
    A, T = hyperbolic_adjacency_sparse(p, q, n)
    H = -1.0 * A  # tight-binding Hamiltonian

    # Precompute coordinates
    coords = [T.get_center(i) for i in range(len(T))]
    site = 0

    # Precompute squared distances
    r2 = np.zeros(len(coords))
    for i in range(len(coords)):
        d = ht.distance.disk_distance(coords[i], coords[site])
        r2[i] = d**2

    for t in times:
        prob = simulate_random_walk_sparse(T, t, H)

        sigma = np.sqrt(np.sum(prob * r2))
        stds.append(sigma)
    
    return times, stds

    # plt.plot(times, stds)
    # plt.xlabel("t")
    # plt.ylabel("σ(t)")
    # plt.title("Quantum walk on hyperbolic lattice (sparse)")
    # plt.show()


# Example
# std_dev_with_t_sparse(12, 3, 3, 30)
from scipy.stats import linregress

def linear_region_study(times, stds, r2_threshold = 0.99):
    times = np.array(times)
    stds = np.array(stds)
    
    # We need at least 3 points to evaluate a linear trend meaningfully
    best_slope = None
    best_intercept = None
    limit_idx = 2
    
    for i in range(3, len(times)):
        # Fit a line through the data from t=0 to t=i
        slope, intercept, r_value, p_value, std_err = linregress(times[:i], stds[:i])
        r2 = r_value**2
        
        # As long as the fit is nearly perfect, update our metrics
        if r2 >= r2_threshold:
            best_slope = slope
            best_intercept = intercept
            limit_idx = i - 1  # Last index that satisfied the linear condition
        else:
            # The moment it drops below the threshold, we stop
            break
            
    return limit_idx, times[limit_idx], best_slope, best_intercept

# times, stds = std_dev_with_t_sparse(10, 3, 3, 20)
# plt.plot(times, stds)
# plt.xlabel("t")
# plt.ylabel("σ(t)")
# plt.title("Quantum walk on hyperbolic lattice (sparse)")

# lim_idx, _, m, c = linear_region_study(times, stds)
# plt.plot(times[:lim_idx+1], times[:lim_idx+1]*m + c)
# plt.show()

#Now, we sweep across a {p, q} grid and compute the slopes
def p_q_slope(p_min=4, p_max=20, q_min=4, q_max=20, t_max=20):
    # Initialize an array of NaNs to store the slopes
    # Sized to allow direct indexing: slopes[p, q]
    slopes = np.full((p_max, q_max), np.nan)

    for p in range(p_min, p_max):
        for q in range(q_min, q_max):
            
            # Ensure the geometry is strictly hyperbolic
            if (p - 2) * (q - 2) > 4:
                try:
                    # 3 layers is kept as hardcoded in your prompt
                    times, stds = std_dev_with_t_sparse(p, q, 3, t_max)
                    
                    # Ensure this unpacking matches your actual linear_region_study function
                    _, _, m, _ = linear_region_study(times, stds)
                    
                    # Store the slope
                    slopes[p, q] = m
                    
                except Exception as e:
                    print(f"Simulation failed for p={p}, q={q} due to: {e}")
                    pass # Leaves the value as NaN

    # --- Plotting the Heatmap ---
    plt.figure(figsize=(12, 9))
    
    # Slice the array to remove the unused 0-3 indices for cleaner plotting
    plot_data = slopes[p_min:p_max, q_min:q_max]
    
    # Create the heatmap
    ax = sns.heatmap(
        plot_data, 
        xticklabels=range(q_min, q_max), 
        yticklabels=range(p_min, p_max),
        cmap="magma",       # 'magma' or 'viridis' are great for scientific data
        annot=False,        # Change to True if you want the exact numbers printed in each cell
        fmt=".2f",          # Formats the annotations to 2 decimal places if annot=True
        cbar_kws={'label': 'Ballistic Velocity (Slope m)'},
        linewidths=0.5,     # Adds gridlines between cells
        na_color='lightgray' # Colors the non-hyperbolic / failed combinations gray
    )
    
    # Invert Y-axis so p increases upwards (optional, but standard for cartesian grids)
    ax.invert_yaxis()

    plt.title("Heatmap of Quantum Walk Dispersion Slopes on {p, q} Lattices", pad=20, fontsize=14)
    plt.xlabel("q (Polygons meeting at a vertex)", fontsize=12)
    plt.ylabel("p (Sides per polygon)", fontsize=12)
    
    plt.tight_layout()
    plt.show()

    return slopes

if __name__ == "__main__":
    slope_matrix = p_q_slope()