#!/usr/bin/env python

import unittest
from mlModel.network import mlNetwork
from mlModel.aux import tfAux

class testMlNetwork(unittest.TestCase):

    def setUp(self):

        self.testNetwork = mlNetwork.mlNetwork([2, 3, 4, 2, 5],
                                               tfAux.leaky_relu)

    def test_createInitWeights(self):
        """
            Only run test
        """

        self.testNetwork._createInitialWeights()
        print(self.testNetwork.wInits)

    def test_setUpCompGraph(self):
        """
            Only run test
        """

        self.testNetwork._createInitialWeights()
        self.testNetwork._createComputationalGraph()