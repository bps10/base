# -*- coding: utf-8 -*-
import os 
import unittest
import numpy as np

import files as f


class TestFiles(unittest.TestCase):

    def test_files(self):
        
        filt = f.getAllFiles(".")


if __name__ == '__main__':
    unittest.main()