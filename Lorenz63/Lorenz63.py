# %%
#!/usr/bin/env python3
## Lorenz 63

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %%
# Derivative function

def dydt_func(t,y,params):
    sigma = params[0]
    beta = params[1]
    rho = params[2]
    dydt = np.zeros_like(y)

    dydt[0] = sigma * (y[1] - y[0] )
    dydt[1] = y[0] * (rho - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - beta * y[2]

    return dydt

if __name__ == "__main__":
    # %%
    params_ex = np.array([10,8/3,28])
    dy = lambda t,y: dydt_func(t,y,params_ex)

    y0 = np.array([1,2,2])
    T = 10000
    N_samples = 1000000
    teval = np.sort((T-100) * np.random.rand(1,N_samples)+100)
    sol = solve_ivp(dy,[0,10000], y0,method = 'DOP853',t_eval = teval[0])

    # %%
    dataset = sol.y

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dataset[0,:],dataset[1,:],dataset[2,:],s=0.1)
    
    dataset = dataset.T
    np.save(f'lorenzdataset.npy',dataset)
    #np.save(f'lorenzdataset_{N_samples}.npy',dataset)
    plt.show()

    # %%


def return_data(y0, random_seed):
    np.random.seed(random_seed)
    params_ex = np.array([10,8/3,28])
    dy = lambda t,y: dydt_func(t,y,params_ex)

    #y0 = np.array([1,2,2])
    T = 10000
    N_samples = 5000# 1000000
    teval = np.sort((T-100) * np.random.rand(1,N_samples)+100)
    sol = solve_ivp(dy,[0,10000], y0,method = 'DOP853',t_eval = teval[0])

    # %%
    dataset = sol.y
    return dataset

    
def generate_invariant_lorenz63(
    size, #: tuple[int], # (N, d)
    y0, #: tuple[float],
    random_seed: int,
):
        
    y0 = np.array(y0)
    X = return_data(y0, random_seed)
    
    X = X.T
    X = X[0:size[0]]
    
    return X
    
