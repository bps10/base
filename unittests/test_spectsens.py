# -*- coding: utf-8 -*-
import os 
import unittest
import numpy as np

<<<<<<< HEAD
os.path.os.chdir('../')

from base import spectsens as sp
=======
import spectsens as sp
>>>>>>> 387604c0eafb860db222f4e7a8825452599199f9

import sys

sys.path.append('../../')

class TestSpec(unittest.TestCase):

    def test_neitz(self):
        
        sens = sp.neitz(LambdaMax=559, OpticalDensity=0.5, LOG=False,
                StartWavelength=390, EndWavelength=770, resolution=1)
        self.assertTrue(len(sens) == 381)
        self.assertTrue(round(np.max(sens), 2) == 1.0)

    def test_stockman(self):
        sens = sp.stockman()
        self.assertTrue(len(sens[:, 0]) == 381)
        
        self.assertTrue(round(np.max(sens[:, 0]), 2) == 1.0)

        sens = sp.stockman(resolution=10)
        self.assertTrue(len(sens) == 39)

    def test_stockmanfund(self):
        sens = sp.stockmanfund()
        self.assertTrue(len(sens[:, 0]) == 381)
        self.assertTrue(round(np.max(sens[:, 0]), 2) == 1.0)

        sens = sp.stockmanfund(resolution=10)
        self.assertTrue(len(sens[:, 0]) == 39)

os.path.os.chdir('base/')        

if __name__ == '__main__':
    unittest.main()