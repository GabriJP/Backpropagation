import csv

import numpy as np
from multiprocessing.pool import ThreadPool

from nn import net_1capa, net_2capas


def ejecuta(funcion_nn, rate, neuronas, noise, funcion_bp):
    for i in range(3):
        itera = funcion_nn(rate, neuronas, noise, funcion_bp)
        if itera >= 0:
            return itera
    return 'Max_iter'


csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = list(range(7, 25))

for capas in [net_1capa, net_2capas]:
    for func in [1]:
        for noise in np.arange(0, 0.25, 0.05):
            writer.writerow(["Ejecución con %s ruido %f y funcion %s" % (
                "una capa oculta" if capas == net_1capa else "dos capas ocultas", noise,
                str(func)[str(func).rfind('.') + 1:-2])])

            writer.writerow([])
            writer.writerow(['ratio'] + list(map(str, n_neuronas)))
            csvfile.flush()
            for ratio in np.arange(0.2, 1.2, 0.1):
                pool = ThreadPool()
                results = [pool.apply_async(ejecuta, (capas, ratio, neuronas_capa, noise, func)) for neuronas_capa in
                           n_neuronas]
                pool.close()
                pool.join()
                writer.writerow([ratio] + [result.get() for result in results])
                csvfile.flush()
            writer.writerow([])

csvfile.flush()
csvfile.close()

"""
neuronas: 7 a 20
ruido: 0 a 0.25 en pasos de 0.05
ratio de 0.2 a 1.2 en pasos de 0.1
maximo iteraciones en 750

"""
