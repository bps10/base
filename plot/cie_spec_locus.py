import numpy as np
import matplotlib.pylab as plt
import os
try:
	from scipy.interpolate import splrep, splev
	routine = 'scipy'
except ImportError, e:
	routine = 'numpy'

this_dir = os.path.dirname(os.path.realpath(__file__))


def add_cie_spec_locus(ax, purpleLine=True):
	'''
	'''
	spectrum = np.arange(390, 720, 1)

	xyzbar = np.genfromtxt(this_dir + '/ciexyzjv.csv', delimiter=',')
	xyz_sp = xyzbar[:, 0]
	xyzbar = xyzbar[:, 1:4]

	newxyzbar = np.zeros((3, len(spectrum)))
	for i in range(0, 3):
		if routine == 'scipy':
			fun = splrep(xyz_sp, xyzbar[:, i], s=0)
			newxyzbar[i, :] = splev(spectrum, fun, der=0)
		else:
			newxyzbar[i, :] = np.interp(spectrum, 
				xyz_sp, xyzbar[:, i])

	xyzbar = newxyzbar
	xyzbar = np.dot(np.diag(1 / xyzbar.sum(0)), xyzbar.T)

	ax.plot(xyzbar[:, 0], xyzbar[:, 1], 'k-')

	if purpleLine:
		ax.plot([xyzbar[0, 0], xyzbar[-1, 0]],
			[xyzbar[0, 1], xyzbar[-1, 1]], 'k-')

