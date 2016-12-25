import csv
import numpy as np
from multiprocessing.pool import ThreadPool

from nn import net_1capa, net_2capas


def ejecuta(funcion_nn, rate, neuronas, ruido, funcion_bp, moment, neuronas_2):
    for i in range(3):
        itera = funcion_nn(rate, neuronas, ruido, funcion_bp, moment, neuronas_2)
        if itera >= 0:
            return itera
    return 'Max_iter'


csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = [11, 14, 17,  20]

capas = net_1capa
for func in [1, 2]:
    for momentum in ([0.5, 0.7, 0.9] if func == 2 else [0]):
        for noise in [0.0]:
            writer.writerow(["Ejecución con %s ruido %01.2f y funcion %s" % (
                "una capa oculta" if capas == net_1capa else "dos capas ocultas", noise,
                "gradiente" if func == 1 else "gradiente con momento %01.1f" % momentum)])

            if net_1capa == capas:
                neuronas_por_capa = [(capa1, 0) for capa1 in n_neuronas]
                writer.writerow(['ratio'] + [str(capa1) for capa1 in n_neuronas])
            else:
                neuronas_por_capa = [(capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas]
                writer.writerow(['ratio'] + ["%d - %d" % (capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas])

            for ratio in [0.4, 0.6, 0.8]:
                with ThreadPool(processes=32) as pool:
                    results = [
                        pool.apply_async(ejecuta, (capas, ratio, neu_1, noise, func, momentum, neu_2)) for
                        neu_1, neu_2 in neuronas_por_capa]
                    pool.close()
                    pool.join()
                    writer.writerow([ratio] + [r.get() for r in results])
                    csvfile.flush()
            writer.writerow([])

csvfile.flush()
csvfile.close()