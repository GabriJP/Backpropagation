import csv

import numpy as np

from nn import net_gradiente

csvfile = open('datos.csv', 'w')
writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

n_neuronas = list(range(5, 25))

for noise in np.arange(0, 0.26, 0.02):
    writer.writerow(["Ejecución con una capa oculta y ruido %.00f" % noise])
    writer.writerow([noise])
    writer.writerow(['ratio'] + n_neuronas)
    for ratio in np.arange(0.05, 1.5, 0.05):
        n_iteraciones = []
        for neuronas_capa_1 in n_neuronas:
            control = False
            for i in range(3):
                itera = net_gradiente(ratio, neuronas_capa_1, noise)
                if itera >= 0:
                    control = True
                    n_iteraciones.append(itera)
                    break
            if not control:
                n_iteraciones.append('Max_iter')
        writer.writerow([ratio] + n_iteraciones)
    writer.writerow([])

csvfile.close()
