import pickle
from randomwalk import std_dev_with_t_sparse, linear_region_study

# DEFINE PARAMETERS
T_MAX = 100
res = {}

for p in range(4, 10):
    for q in range(4, 10):
        if (p - 2) * (q - 2) <= 4:
            continue

        lin_limits_for_n = [] 
        
        for n in range(3, 11):
            try:
                times_array, stds = std_dev_with_t_sparse(p, q, n, T_MAX)
                
                _, lin_time, _, _ = linear_region_study(times_array, stds)
                lin_limits_for_n.append(lin_time)
                
            except Exception as e:
                # Catch out-of-memory errors for massive n values
                print(f"Simulation failed at p={p}, q={q}, n={n}. Error: {e}")
                lin_limits_for_n.append(None) # Pad with None to keep lengths consistent

        # Store in the dictionary
        res[(p, q)] = lin_limits_for_n
        print(f"Completed p={p}, q={q}: {lin_limits_for_n}")

with open('hyperbolic_linear_limits.pkl', 'wb') as f:
    pickle.dump(res, f)

print("Results successfully saved to 'hyperbolic_linear_limits.pkl'")