import csv

import numpy as np
from multiprocessing.pool import ThreadPool

from nn import net_1capa, net_2capas


def ejecuta(funcion_nn, rate, neuronas, noise, funcion_bp, momentum, neuronas_2):
    for i in range(3):
        itera = funcion_nn(rate, neuronas, noise, funcion_bp, momentum, neuronas_2)
        if itera >= 0:
            return itera
    return 'Max_iter'


csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = list(range(7, 26))

for capas in [net_1capa, net_2capas]:
    for func in [1, 2]:
        for momentum in (np.arange(0.1, 1.1, 0.1) if func == 2 else [0]):
            for noise in np.arange(0, 0.3, 0.05):
                writer.writerow(["Ejecución con %s ruido %01.2f y funcion %s" % (
                    "una capa oculta" if capas == net_1capa else "dos capas ocultas", noise,
                    "gradiente" if func == 1 else "gradiente con momento %01.1f" % momentum)])

                neuronas_por_capa = [(capa1, 0) for capa1 in n_neuronas] if net_1capa == capas else \
                    [(capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas]

                writer.writerow(['ratio'] +
                                [str(capa1) for capa1 in n_neuronas] if capas == net_1capa else
                                ["%d - %d" % (capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas])
                for ratio in np.arange(0.2, 1.1, 0.1):
                    with ThreadPool(processes=1024) as pool:
                        results = [
                            pool.apply_async(ejecuta, (capas, ratio, neu_1, noise, func, momentum, neu_2)) for
                            neu_1, neu_2 in neuronas_por_capa]
                        pool.close()
                        pool.join()
                        writer.writerow([ratio] + [r.get() for r in results])
                        csvfile.flush(

                        )
                writer.writerow([])
                csvfile.flush()
csvfile.flush()
csvfile.close()

"""
neuronas: 7 a 20
ruido: 0 a 0.25 en pasos de 0.05
ratio de 0.2 a 1.2 en pasos de 0.1
maximo iteraciones en 750

"""
