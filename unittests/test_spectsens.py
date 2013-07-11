# -*- coding: utf-8 -*-
import os 
import unittest
import numpy as np

from base import spectsens as sp


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
        
if __name__ == '__main__':
    unittest.main()