# coding=utf-8
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

# 1 Layer
'''
# Gradient noise 0
for ratio in np.arange(0.2, 1.3, 0.1):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_1capa, ratio, neu_1, 0, 1, 0, 0)) for
            neu_1 in range(7, 25, 1)]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])


# Momento 0.9 noise 0
for ratio in np.arange(0.2, 1.3, 0.1):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_1capa, ratio, neu_1, 0, 2, 0.9, 0)) for
            neu_1 in range(7, 25, 1)]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])


# Momento 0.3 noise 7%
for ratio in np.arange(0.2, 1.3, 0.1):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_1capa, ratio, neu_1, 0.07, 2, 0.3, 0)) for
            neu_1 in range(7, 25, 1)]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])

# Momento 0.1 noise 14%
for ratio in np.arange(0.2, 1.3, 0.1):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_1capa, ratio, neu_1, 0.14, 2, 0.1, 0)) for
            neu_1 in range(7, 25, 1)]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])


# 2 Layer

# Gradiente noise 0%
for ratio in np.arange(0.2, 1, 0.4):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_2capas, ratio, neu_1, 0, 1, 0.3, neu_2)) for
            neu_1, neu_2 in [(10,15),(10,20),(15,20),(20,15),(20,20)]]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])

# Momento 0.3 noise 0%
for ratio in np.arange(0.2, 1, 0.4):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_2capas, ratio, neu_1, 0, 2, 0.3, neu_2)) for
            neu_1, neu_2 in [(10,15),(10,20),(15,20),(20,15),(20,20)]]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])

# Ratio 0.5 noise 7%
for momento in np.arange(0.01, 0.8, 0.15):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_2capas, 0.5, neu_1, 0.07, 2, momento, neu_2)) for
            neu_1, neu_2 in [(15,20),(20,15),(20,20)]]
        pool.close()
        pool.join()
        writer.writerow([ratio] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])
'''
# Ratio 0.2 noise 14%
for momento in np.arange(0.01, 0.5, 0.15):
    with ThreadPool(processes=32) as pool:
        results = [
            pool.apply_async(ejecuta, (net_2capas, 0.2, neu_1, 0.14, 2, momento, neu_2)) for
            neu_1, neu_2 in [(15, 20), (20, 15), (20, 20)]]
        pool.close()
        pool.join()
        writer.writerow([momento] + [r.get() for r in results])
        csvfile.flush()
writer.writerow([])

csvfile.flush()
csvfile.close()


def bigAnalysis():
    n_neuronas = [10, 15, 20]

    for capas in [net_1capa, net_2capas]:
        for func in [1, 2]:
            for momentum in ([0.5, 0.7, 1] if func == 2 else [0]):
                for noise in [0, 0.07, 0.14]:
                    writer.writerow(["Ejecución con %s ruido %01.2f y funcion %s" % (
                        "una capa oculta" if capas == net_1capa else "dos capas ocultas", noise,
                        "gradiente" if func == 1 else "gradiente con momento %01.1f" % momentum)])

                    if net_1capa == capas:
                        neuronas_por_capa = [(capa1, 0) for capa1 in n_neuronas]
                        writer.writerow(['ratio'] + [str(capa1) for capa1 in n_neuronas])
                    else:
                        neuronas_por_capa = [(capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas]
                        writer.writerow(
                            ['ratio'] + ["%d - %d" % (capa1, capa2) for capa1 in n_neuronas for capa2 in n_neuronas])

                    for ratio in [0.5, 0.8, 1.1]:
                        with ThreadPool(processes=32) as pool:
                            results = [
                                pool.apply_async(ejecuta, (capas, ratio, neu_1, noise, func, momentum, neu_2)) for
                                neu_1, neu_2 in neuronas_por_capa]
                            pool.close()
                            pool.join()
                            writer.writerow([ratio] + [r.get() for r in results])
                            csvfile.flush()
                    writer.writerow([])
