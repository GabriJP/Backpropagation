import csv

import numpy as np
from tensorflow import train

from nn import net_1capa, net_2capas

csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = list(range(7, 20))

for capas in [net_1capa, net_2capas]:
    for func in [train.GradientDescentOptimizer, train.MomentumOptimizer]:
        for noise in np.arange(0, 0.25, 0.05):
            writer.writerow(["Ejecución con % ruido %f y funcion %s" % (
                "una capa oculta" if capas == net_1capa else "dos capas ocultas", noise,
                str(func)[str(func).rfind('.') + 1:-2])])

            writer.writerow([])
            writer.writerow(['ratio'] + list(map(str, n_neuronas)))
            csvfile.flush()
            for ratio in np.arange(0.2, 1.2, 0.1):
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