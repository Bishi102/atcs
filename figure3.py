"""
Excecutable to obtain Figure 3
Please uncomment lines 114 and 115 in optimisation.py
"""
from plots import Plots
from des import CoffeeShopSimulator
from optimisation import Optimisation

import numpy as np
np.random.seed(69)

def obj(params):
    # defaults
    default_values = {
        'numBaristas': 2,
        'coffeePrice': 5.0,
        'wagePerHour': 20.0,
        'shopStartTime': 6.0,
        'shopOpenHours': 12,
        'maxQueueLength': 5
    }
    # replacing defaults with given parameters
    param_keys = list(default_values.keys())
    for i in range(len(params)):
        key = param_keys[i]
        default_values[key] = params[i]

    # getting result
    sim = CoffeeShopSimulator(**default_values)
    result = sim.run()

    return result['profit']

if __name__ == "__main__":

    bounds = np.array([
        [1, 4], # num of baristas
        [1.0, 7.0] # coffee price
    ])

    optimiser = Optimisation(
        obj = obj,
        bounds = bounds,
        nInit = 10,
        nIter = 10,
        nCand = 50,
        beta = 2.0,
        epsilonStart=0.99,
        epsilonEnd=0.1,
        epsilonDecay=0.95
    )

    bestParams, bestProfit = optimiser.optimise()
