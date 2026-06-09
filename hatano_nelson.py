import numpy as np
from pbc_adj import build_hyperbolic_lattice, visualize_lattice 

def introduce_asymmetry(adj, delta_t) :
    adj_copy = np.array(adj)
    for i in range(np.array(adj).shape[0]):
        for j in range(i, np.array(adj).shape[0]):
            if i != j:
                adj_copy[i][j] += delta_t
                adj_copy[j][i] -= delta_t
    return adj_copy

def introduce_nonhermiticity(adj, eta):
    adj_copy = np.array(adj, dtype=complex, copy=True)
    for i in range(adj_copy.shape[0]):
        adj_copy[i][i] += 1j * eta
    return adj_copy

p, q, n = 3, 7, 4

# for p in [3, 4]:
#     for q in [6, 7, 8, 9, 10]:
#         try:
#             adj, G, circles = build_hyperbolic_lattice(Nc=n, k=q, m=p, seed=1)
#         except ValueError:
#             print(f"{p, q} does not work")
#             continue
#         print(f"{p,q} successful")
        
adj, _, _ = build_hyperbolic_lattice(Nc=n, k=q, m=p, seed=1)
non_hermitian_adj = introduce_nonhermiticity(adj, 0.1)
