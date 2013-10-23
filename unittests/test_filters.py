# -*- coding: utf-8 -*-
import unittest
import numpy as np

import sys
sys.path.append('../')

from base import optics as op


class TestFilters(unittest.TestCase):

    def test_filters(self):
        
        filt = op.filters.stockman(minLambda=390, maxLambda=770, 
            ONLY_LENS=False, ONLY_MACULA=False, RETURN_SPECTRUM=False, 
            resolution=1)
        self.assertTrue(len(filt) == 381)
        self.assertTrue(np.max(filt) > 1.0)
        filt = op.filters.stockman(minLambda=390, maxLambda=770, 
            ONLY_LENS=False, ONLY_MACULA=False, RETURN_SPECTRUM=False, 
            resolution=10)
        self.assertTrue(len(filt) == 39)
        filt = op.filters.stockman(minLambda=390, maxLambda=770, 
            ONLY_LENS=False, ONLY_MACULA=True, RETURN_SPECTRUM=False, 
            resolution=10)
        print np.max(filt)
        self.assertTrue(np.max(filt) < 2)


        
if __name__ == '__main__':
    unittest.main()