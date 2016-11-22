import csv

import numpy as np
from tensorflow import train

from nn import net_1capa, net_2capas

csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = list(range(5, 25))

for capas in [net_1capa, net_2capas]:
    for func in [train.GradientDescentOptimizer, train.MomentumOptimizer]:
        for noise in np.arange(0, 0.26, 0.02):
            writer.writerow(["Ejecución con una capa oculta ruido %.00f y funcion %s" % (
                noise, str(func)[str(func).rfind('.') + 1:-2])])
            writer.writerow([])
            writer.writerow(['ratio'] + list(map(str, n_neuronas)))
            for ratio in np.arange(0.05, 1.5, 0.05):
                n_iteraciones = []
                for neuronas_capa_1 in n_neuronas:
                    control = False
                    for i in range(3):
                        itera = capas(ratio, neuronas_capa_1, noise, train.GradientDescentOptimizer)
                        if itera >= 0:
                            control = True
                            n_iteraciones.append(itera)
                            break
                    if not control:
                        n_iteraciones.append('Max_iter')
                writer.writerow([ratio] + n_iteraciones)
            writer.writerow([])

csvfile.close()
