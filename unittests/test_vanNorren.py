from __future__ import division
import matplotlib.pylab as plt
import numpy as np

from base.optics import filters as f

spectrum = np.arange(390, 650)

plt.figure()
for age in range(20, 81, 10):
    lens = f.vanNorren(age, spectrum, 'large')

    plt.plot(spectrum, lens, 'k-')

plt.show()
