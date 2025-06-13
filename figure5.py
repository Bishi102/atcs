"""
Executable to obtain Figure 5
"""
from plots import Plots
from des import CoffeeShopSimulator
from optimisation import Optimisation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(69)

def obj(params):
    default_values = {
        'numBaristas': 2,
        'coffeePrice': 5.0,
        'wagePerHour': 20.0,
        'shopStartTime': 6.0,
        'shopOpenHours': 12,
        'maxQueueLength': 5
    }
    param_keys = list(default_values.keys())
    for i in range(len(params)):
        key = param_keys[i]
        default_values[key] = params[i]
    sim = CoffeeShopSimulator(**default_values)
    result = sim.run()
    return result['profit']

def clamp_bounds(x, bounds):
    return np.clip(x, bounds[:, 0], bounds[:, 1])

def scipy_optimizer(obj, bounds, method="L-BFGS-B", n_iter=100):
    dim = bounds.shape[0]
    x0 = np.mean(bounds, axis=1)
    history = []

    def wrapped_obj(x):
        x_clamped = clamp_bounds(x, bounds)
        val = obj(x_clamped)
        history.append(val)
        return -val  # negate for maximization

    result = minimize(
        fun=wrapped_obj,
        x0=x0,
        method=method,
        bounds=None if method in ["Nelder-Mead", "Powell"] else bounds,
        options={'maxfev': n_iter, 'disp': False}
    )

    best_params = clamp_bounds(result.x, bounds)
    best_profit = -result.fun
    return best_params, best_profit, history

if __name__ == "__main__":

    bounds = np.array([
        [1, 4],    # num of baristas
        [1.0, 7.0],# coffee price
        [20, 40]   # wage per hour
    ])

    # Kalman-Delaunay Optimisation
    optimiser = Optimisation(
        obj=obj,
        bounds=bounds,
        nInit=50,
        nIter=950,
        nCand=1000,
        beta=1000.0,
        epsilonStart=0.99,
        epsilonEnd=0.1,
        epsilonDecay=0.99745
    )
    bestParams_kf, bestProfit_kf = optimiser.optimise()
    print("Kalman Optimiser Best:", bestParams_kf, "Profit:", bestProfit_kf)

    # SciPy L-BFGS-B
    params_lbfgs, profit_lbfgs, history_lbfgs = scipy_optimizer(obj, bounds, method="L-BFGS-B", n_iter=1000)
    print("L-BFGS-B Best:", params_lbfgs, "Profit:", profit_lbfgs)

    # SciPy Nelder-Mead
    params_nm, profit_nm, history_nm = scipy_optimizer(obj, bounds, method="Nelder-Mead", n_iter=1000)
    print("Nelder-Mead Best:", params_nm, "Profit:", profit_nm)

    # SciPy Powell
    params_powel, profit_powel, history_powel = scipy_optimizer(obj, bounds, method="Powell", n_iter=1000)
    print("Powell Best:", params_powel, "Profit:", profit_powel)

    # Plotting
    plt.figure(figsize=(20, 12))
    def running_max(seq):
        return np.maximum.accumulate(seq)
    plt.plot(running_max(optimiser.history), label="Kalman-Delaunay Optimiser", marker='o')
    plt.plot(running_max(history_lbfgs), label="L-BFGS-B (scipy)", marker='x')
    plt.plot(running_max(history_nm), label="Nelder-Mead (scipy)", marker='s')
    plt.plot(running_max(history_powel), label="Powell (scipy)", marker='^')
    plt.xlabel("Function Evaluations")
    plt.ylabel("Estimated Profit")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure5_convergence.png", dpi=800)
    plt.show()
