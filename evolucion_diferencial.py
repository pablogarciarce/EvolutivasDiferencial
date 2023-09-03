import json
import random
import math
import numpy as np
import pandas as pd
import statistics
import time
import multiprocessing
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
from itertools import product

import functions


def main(f, ejs):
    # Parámetros
    params = {
        "init": ['latinhypercube', 'sobol', 'random'],
        "strategy": ['rand1bin', 'best1bin'],
        "F": np.linspace(0.5, 1.9, 15),
        "CR": np.linspace(0.2, 1.0, 9)
    }
    for ej in ejs:
        print('EJECUCION ', ej)
        df = pd.DataFrame(columns=['init', 'strategy', 'F', 'CR', 'x', 'valor'])
        for param in list(product(*params.values())):
            init = param[0]
            strategy = param[1]
            F = param[2]
            CR = param[3]
            pop_size = 3
            num_generations = 1200
            if f.__name__ == 'powell':
                bounds = np.array([[-1, 1] for _ in range(10)])
            else:
                bounds = np.array([[-10, 10] for _ in range(10)])
            t = time.time()
            result = differential_evolution(
                f,
                bounds,
                strategy=strategy,
                tol=1e-6,
                popsize=pop_size,
                maxiter=num_generations,
                mutation=F,
                recombination=CR,
                init=init
            )
            print(param, 'Tiempo: ', time.time() - t)

            df.loc[len(df)] = [*param, result.x, result.fun]
        df.to_csv('results/dif/' + f.__name__ + '_e' + str(ej) + '.csv')


if __name__ == '__main__':
    ejecuciones = [0, 1, 2, 3, 4]
    f = functions.xinshe
    main(f, ejecuciones)
