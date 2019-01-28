#!/usr/bin/env python

import unittest
from mlModel.network import mlNetwork
from mlModel.aux import tfAux
import numpy as np

class testMlNetwork(unittest.TestCase):

    def setUp(self):

        self.layers = [2, 3, 4, 2, 5]
        self.testNetwork = mlNetwork.mlNetwork(self.layers,
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

    def test_getActionVector(self):
        """
            Test the output of the get activation vector
        """

        # set up the network for testing
        self.testNetwork._createInitialWeights()
        self.testNetwork._createComputationalGraph()

        actVec = self.testNetwork.getActionVector(np.array([0.1, 0.1]))

        # Make the checks
        self.assertTrue(len(actVec) == self.layers[-1])
        self.assertTrue(np.sum(actVec) == 1)
        self.assertTrue(np.sum(actVec == 1) == 1)
