#!/usr/bin/env python

import unittest
from mlModel.aux import tfAux

class importingAux(unittest.TestCase):

    def test_import(self):

        functions = ['map',
                     'leaky_relu',
                     'tf_mat_vec_dot',
                     'jacobian']
        for testF in functions:
            self.assertTrue(testF in dir(tfAux))