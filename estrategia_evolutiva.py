import json
import random
import math
import numpy as np
import statistics
import time
import multiprocessing
from joblib import Parallel, delayed


class Individuo:
    def __init__(self, dim, n_sigmas):
        self.dim = dim
        self.n_sigmas = n_sigmas
        self.solucion = np.array
        if n_sigmas:
            self.sigmas = np.array
        else:
            self.sigmas = None
        self.valor = None

    def init_aleatorio(self, box):
        self.solucion = - box + 2 * box * np.random.random(self.dim)
        if self.n_sigmas:
            self.sigmas = np.random.random(self.dim)
        else:
            self.sigmas = np.random.random(1)
        return self

    def asignar_solucion(self, solucion):
        self.solucion = solucion.copy()
        return self

    def asignar_sigmas(self, sigmas):
        if self.n_sigmas:
            self.sigmas = sigmas.copy()
        else:
            self.sigmas = sigmas
        return self

    def asignar_valor(self, valor):
        self.valor = valor
        return self

    def mutar(self, eps, tau, tau_i=None):
        if self.n_sigmas:
            self.sigmas *= np.exp(tau * np.random.normal(0, 1, 1))
            self.sigmas *= np.exp(tau_i * np.random.normal(0, 1, self.dim))
            self.sigmas[self.sigmas < eps] = eps
            self.solucion += self.sigmas * np.random.normal(0, 1, self.dim)
        else:
            self.sigmas *= np.exp(tau * np.random.normal(0, 1, 1))
            if self.sigmas < eps:
                self.sigmas = eps
            self.solucion += self.sigmas * np.random.normal(0, 1, self.dim)

    def eval(self, f):
        self.valor = f(self.solucion)

    def copiar(self):
        ind = Individuo(self.dim, self.n_sigmas)
        return ind.asignar_sigmas(self.sigmas).asignar_solucion(self.solucion).asignar_valor(self.valor)


class Poblacion:
    def __init__(self, dimension, num_individuos, num_padres, num_hijos, n_sigmas, box):
        self.dim = dimension
        self.n_sigmas = n_sigmas
        self.num_individuos = num_individuos
        self.num_padres = num_padres
        self.num_hijos = num_hijos
        self.poblacion = [Individuo(dimension, n_sigmas).init_aleatorio(box) for _ in range(num_individuos)]
        self.hijos = None
        self.mejor_individuo = None

    def seleccion_padres(self):
        padres = []
        for i in range(self.num_hijos):
            selec = random.sample(self.poblacion, self.num_padres)
            padres.append([s.copiar() for s in selec])
        return padres

    def cruce(self, discreto):
        padres = self.seleccion_padres()
        if discreto:
            self.hijos = Parallel(n_jobs=2)(delayed(self.cruce_discreto)(p) for p in padres)
        else:
            self.hijos = Parallel(n_jobs=2)(delayed(self.cruce_promedio)(p) for p in padres)

    def cruce_discreto(self, padres):
        matriz_sol = np.vstack([p.solucion for p in padres])
        matriz_sig = np.vstack([p.sigmas for p in padres])
        num_padres, _ = matriz_sol.shape
        indices_sol = np.random.choice(num_padres, size=self.dim)
        indices_sig = np.random.choice(num_padres, size=self.dim if self.n_sigmas else 1)
        hijo = Individuo(self.dim, self.n_sigmas)
        return (hijo.asignar_solucion(matriz_sol[indices_sol, range(self.dim)])
                .asignar_sigmas(matriz_sig[indices_sig, range(self.dim if self.n_sigmas else 1)]))

    def cruce_promedio(self, padres):
        matriz_soluciones = np.vstack([p.solucion for p in padres])
        matriz_sigmas = np.vstack([p.sigmas for p in padres])
        hijo = Individuo(self.dim, self.n_sigmas)
        return hijo.asignar_solucion(np.mean(matriz_soluciones, axis=0)).asignar_sigmas(np.mean(matriz_sigmas, axis=0))

    def mutacion(self, eps, tau, tau_i):
        for ind in self.hijos:
            ind.mutar(eps, tau, tau_i)

    def evaluar(self, f, padres=False):
        if padres:
            for ind in self.poblacion:
                ind.eval(f)
        else:
            for ind in self.hijos:
                ind.eval(f)

    def seleccion_supervivientes(self, elitismo):
        if elitismo:
            self.poblacion = sorted(self.poblacion + self.hijos, key=lambda x: x.valor)[:self.num_individuos]
        else:
            self.poblacion = sorted(self.hijos, key=lambda x: x.valor)[:self.num_individuos]
        self.mejor_individuo = self.poblacion[0]

    def ejecutar_iteracion(self, discreto, eps, tau, tau_i, f, elitismo):
        t = time.time()
        self.cruce(discreto)
        self.mutacion(eps, tau, tau_i)
        self.evaluar(f)
        self.seleccion_supervivientes(elitismo)
        # print('Tiempo iteracion: ', time.time() - t)

    def media_y_desviacion(self):
        valores = [ind.valor for ind in self.poblacion]
        return sum(valores) / len(valores), statistics.stdev(valores)


def seleccionar(lista_individuos):
    seleccionado = lista_individuos[0]
    for ind in lista_individuos:
        if ind.valor < seleccionado.valor:
            seleccionado = ind
    return seleccionado

