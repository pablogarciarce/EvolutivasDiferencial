import pandas as pd
import matplotlib.pyplot as plt
import glob
import xarray as xr
import numpy as np

import functions


def juntar_evol(func):
    files = glob.glob('results/evol/' + func + '/*.csv')
    arrs = [xr.Dataset.from_dataframe(pd.read_csv(f)).expand_dims(dim={
        "num_padres": [int(f.split('\\')[1].split('_')[0])],
        "eps": [float(f.split('\\')[1].split('_')[1])],
        "tau": [float(f.split('\\')[1].split('_')[2])],
        "n_sigmas": [f.split('\\')[1].split('_')[3] == 'True'],
        "discreto": [f.split('\\')[1].split('_')[4] == 'True'],
        "elitismo": [f.split('\\')[1].split('_')[5] == 'True'],
        "ej": [int(f.split('\\')[1].split('_e')[1].split('.')[0])]
    }, axis=[0, 1, 2, 3, 4, 5, 6]) for f in files]

    # arr = xr.merge(arrs)

    df = pd.DataFrame(columns=['num_padres', 'eps', 'tau', 'n_sigmas', 'discreto', 'elitismo', 'ej', 'index', 'Mejor'])
    for ar in arrs:
        sin_nan = ar.dropna(dim='index')
        df.loc[len(df)] = sin_nan['Mejor'].sel(index=sin_nan.sizes['index'] - 1).to_dataframe().reset_index().values[0]
    df.to_csv('results/evol/' + func + '.csv')


def juntar_dif(func):
    files = glob.glob('results/dif/' + func + '*.csv')
    dfs = []
    for f in files:
        aux = pd.read_csv(f)
        aux['ej'] = int(f.split('_e')[1].split('.')[0])
        dfs.append(aux)
    df = pd.concat(dfs)
    df.to_csv('results/dif/' + func + '.csv')


def plot_functions():
    x1_powell, x2_powell = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    x1_xinshe, x2_xinshe = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

    # Calcular los valores de las funciones
    ys_powell = np.array(
        [functions.powell([x1_powell[i, j], x2_powell[i, j]]) for i in range(x1_powell.shape[0]) for j in
         range(x1_powell.shape[1])])
    ys_xinshe = np.array(
        [functions.xinshe([x1_xinshe[i, j], x2_xinshe[i, j]]) for i in range(x1_xinshe.shape[0]) for j in
         range(x1_xinshe.shape[1])])

    # Reformatear ys_powell y ys_xinshe para que coincidan con las dimensiones de x1 y x2
    ys_powell = ys_powell.reshape(x1_powell.shape)
    ys_xinshe = ys_xinshe.reshape(x1_xinshe.shape)

    # Crear las figuras y gráficos 3D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    surf1 = ax1.plot_surface(x1_powell, x2_powell, ys_powell)
    ax1.set_title('Superficie de la función Powell')
    surf2 = ax2.plot_surface(x1_xinshe, x2_xinshe, ys_xinshe)
    ax2.set_title('Superficie de la función Xinshe')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    contour = plt.contour(x1_powell, x2_powell, ys_powell, levels=20, cmap='viridis')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Contorno de la función Powell')

    plt.subplot(1, 2, 2)
    contour = plt.contour(x1_xinshe, x2_xinshe, ys_xinshe, levels=20, cmap='viridis')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Contorno de la función Xinshe')

    plt.tight_layout()
    plt.show()


def main(func):
    df_dif = pd.read_csv('results/dif/' + func + '.csv').dropna()
    df_evol = pd.read_csv('results/evol/' + func + '.csv').dropna()
    sorted_dif = df_dif.sort_values(by='Mejor').reset_index()
    sorted_evol = df_evol.sort_values(by='Mejor').reset_index()

    cut1 = 1000
    sorted_dif['Mejor'][:cut1].plot()
    sorted_evol['Mejor'][:cut1].plot()
    plt.legend(['Dif', 'Evol'])
    plt.show()

    cut2 = 100
    sorted_dif['Mejor'][:cut2].plot()
    sorted_evol['Mejor'][:cut2].plot()
    plt.legend(['Dif', 'Evol'])
    plt.show()

    sorted_dif['Mejor'][:cut1].plot()
    plt.show()


def analysis_evol(func, umbral=1e-6):
    df = pd.read_csv('results/evol/' + func + '.csv').dropna()

    if func == 'xinshe':
        df['exito'] = abs(df['Mejor'] + 1) < umbral
    elif func == 'powell':
        df['exito'] = abs(df['Mejor']) < umbral

    print('TASA DE EXITO')
    print('Valor medio: ', df['exito'].values.mean())

    print(df.groupby('elitismo')['exito'].mean())
    print(df.groupby('discreto')['exito'].mean())
    print(df.groupby('n_sigmas')['exito'].mean())
    print(df.groupby('num_padres')['exito'].mean())
    print(df.groupby('eps')['exito'].mean())
    print(df.groupby('tau')['exito'].mean())

    df.groupby('eps')['exito'].mean().plot()
    plt.ylabel('Tasa de exito')
    plt.figure()
    df.groupby('num_padres')['exito'].mean().plot()
    plt.ylabel('Tasa de exito')
    plt.figure()
    df.groupby('tau')['exito'].mean().plot()
    plt.ylabel('Tasa de exito')
    plt.show()

    print('VALOR DE ADAPTACION')
    print('Valor medio: ', df['Mejor'].values.mean())

    print(df.groupby('elitismo')['Mejor'].mean())
    print(df.groupby('discreto')['Mejor'].mean())
    print(df.groupby('n_sigmas')['Mejor'].mean())
    print(df.groupby('num_padres')['Mejor'].mean())
    print(df.groupby('eps')['Mejor'].mean())
    print(df.groupby('tau')['Mejor'].mean())

    df.groupby('eps')['Mejor'].mean().plot()
    plt.ylabel('Valor de adaptación')
    plt.figure()
    df.groupby('num_padres')['Mejor'].mean().plot()
    plt.ylabel('Valor de adaptación')
    plt.figure()
    df.groupby('tau')['Mejor'].mean().plot()
    plt.ylabel('Valor de adaptación')
    plt.show()


def analysis_dif(func, umbral=1e-6):
    df = pd.read_csv('results/dif/' + func + '.csv').dropna()

    if func == 'xinshe':
        df['exito'] = abs(df['valor'] + 1) < umbral
    elif func == 'powell':
        df['exito'] = abs(df['valor']) < umbral

    print('TASA DE EXITO')
    print('Valor medio: ', df['exito'].values.mean())

    print(df.groupby('init')['exito'].mean())
    print(df.groupby('strategy')['exito'].mean())
    print(df.groupby('F')['exito'].mean())
    print(df.groupby('CR')['exito'].mean())

    df.groupby('F')['exito'].mean().plot()
    plt.ylabel('Tasa de exito')
    plt.figure()
    df.groupby('CR')['exito'].mean().plot()
    plt.ylabel('Tasa de exito')
    plt.show()

    print('VALOR DE ADAPTACION')
    print('Valor medio: ', df['valor'].values.mean())

    print(df.groupby('init')['valor'].mean())
    print(df.groupby('strategy')['valor'].mean())
    print(df.groupby('F')['valor'].mean())
    print(df.groupby('CR')['valor'].mean())

    df.groupby('F')['valor'].mean().plot()
    plt.ylabel('Valor de adaptación')
    plt.figure()
    df.groupby('CR')['valor'].mean().plot()
    plt.ylabel('Valor de adaptación')
    plt.show()


if __name__ == '__main__':
    # plot_functions()
    # main('xinshe')
    func = 'powell'
    print('\n FUNCION:' + func + ' \n\n')

    print('\n\n ANALISIS ESTRATEGIA EVOLUTIVA \n\n')
    analysis_evol(func)

    print('\n\n ANALISIS EVOLUCION DIFERENCIAL \n\n')
    analysis_dif(func)
