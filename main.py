from estrategia_evolutiva import Poblacion

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from itertools import product

import functions


class RunSimulation:
    def __init__(self, config_path, fnc):
        with open(config_path) as f:
            config = json.load(f)
        self.num_individuos = config['num_individuos']
        self.num_padres = config['num_padres']
        self.num_hijos = config['num_hijos']
        self.dimension = config['dimension']
        self.box = config['box']
        self.n_sigmas = config['n_sigmas']
        self.discreto = config['discreto']
        self.eps = config['eps']
        self.tau = config['tau']
        self.tau_i = config['tau_i']
        self.elitismo = config['elitismo']
        self.max_ite = config['max_ite']

        self.individuos = Poblacion(
            self.dimension, self.num_individuos, self.num_padres, self.num_hijos, self.n_sigmas, self.box)
        self.f = fnc

    def simulate(self, paciencia=10):
        mejor = []
        media = []
        stdev = []
        self.individuos.evaluar(self.f, padres=True)
        mejor_media = np.inf
        cont_paciencia = 0
        for i in range(self.max_ite):
            print(i)
            self.individuos.ejecutar_iteracion(self.discreto, self.eps, self.tau, self.tau_i, self.f, self.elitismo)
            mejor.append(self.individuos.mejor_individuo.valor)
            media_y_desviacion = self.individuos.media_y_desviacion()
            print(media_y_desviacion[0])
            media.append(media_y_desviacion[0])
            stdev.append(media_y_desviacion[1])
            if media_y_desviacion[0] < mejor_media:
                mejor_media = media_y_desviacion[0]
                cont_paciencia = 0
            else:
                cont_paciencia += 1
                if cont_paciencia > paciencia:
                    break
        print(self.individuos.mejor_individuo.solucion)
        plt.plot(mejor)
        plt.errorbar(range(len(mejor)), media, yerr=stdev)
        plt.show()
        print(mejor[len(mejor)-1])

    def simulate_probs(self, ejecuciones, paciencia=10):
        params = {
            "num_padres": np.arange(3, 10, 3),
            "eps": (10 * np.ones(3)) ** (-np.arange(3, 6)),
            "tau": np.linspace(0.2, 0.5, 4),
            "n_sigmas": [True, False],
            "discreto": [True, False],
            "elitismo": [True, False]
        }
        for param in list(product(*params.values())):
            print(param)
            for ej in ejecuciones:
                path = 'results/evol/' + f.__name__ + '/' + '_'.join([str(p) for p in param]) + '_e' + str(ej) + '.csv'
                print('EJECUCION ', ej)
                df = pd.DataFrame(index=range(self.max_ite), columns=['Mejor', 'Media', 'Std'])
                self.individuos = Poblacion(
                    self.dimension, self.num_individuos, param[0], self.num_hijos, param[3], self.box)
                self.individuos.evaluar(self.f, padres=True)
                mejor_media = np.inf
                cont_paciencia = 0
                for i in range(self.max_ite):
                    self.individuos.ejecutar_iteracion(
                        param[4], param[1], param[2], param[2] * self.dimension ** 1/4, self.f, param[5])
                    media_desviacion = self.individuos.media_y_desviacion()
                    df.iloc[i] = [
                        self.individuos.mejor_individuo.valor,
                        *media_desviacion
                             ]
                    print("Iteracion ", i, " media ", media_desviacion[0])
                    if media_desviacion[0] < mejor_media:
                        mejor_media = media_desviacion[0]
                        cont_paciencia = 0
                    else:
                        cont_paciencia += 1
                        if cont_paciencia > paciencia:
                            break

                df.to_csv(path)


if __name__ == '__main__':
    conf_path = 'config.json'
    f = functions.powell
    # RunSimulation(conf_path, f).simulate(paciencia=100)

    RunSimulation(conf_path, f).simulate_probs(ejecuciones=[2, 3], paciencia=100)


