import numpy as np
import random
import math 
import matplotlib.pyplot as plt

def simulate_classical_walk(n, t):
    #We simulate 'n' walkers for time 't'
    xs = [0] * n
    ys = [0] * n
    for _ in range(t):
        outcomes = [random.randint(1, 4) for _ in range(n)]
        for i in range(n):
            if(outcomes[i] == 1):
                xs[i] += 1
            elif(outcomes[i] == 2):
                xs[i] -= 1
            elif(outcomes[i] == 3):
                ys[i] += 1
            else:
                ys[i] -= 1
    
    distances_sq = [xs[i]**2 + ys[i]**2 for i in range(n)]
    rms = math.sqrt(np.mean(distances_sq))
    return rms

def std_plot_with_t(n, t_max):
    x = np.arange(t_max)
    y = [] 
    for t in range(t_max):
        std = simulate_classical_walk(n, t)  
        y.append(std)
    plt.plot(x, np.array(y))
    plt.show()

std_plot_with_t(100,100)