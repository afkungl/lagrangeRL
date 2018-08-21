import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .timeContinuousClassificationSmOu import timeContinuousClassificationSmOu


class timeContinuousClassificationVerifyBackprop(timeContinuousClassificationSmOu):

    def setUpNetwork(self):
        """
            Set up the network with the lagrange dynamics
        """

        # Set up the network structure
        self.W = lagrangeRL.tools.networks.feedForwardWtaReadout(
            self.layers,
            self.wtaStrength,
            offset=self.initWeightMean,
            noiseMagnitude=self.initWeightWidth)
        # Lagrange network
        self.simClass = lagrangeRL.network.lagrangeEligTf()
        self.simClass.setLearningRate(1.)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(True)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[self.layers[0]:, self.layers[0]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

    def initLogging(self):

        self.logger = logging.getLogger(
            'timeContinuousClassificationVerifyBackprop')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])
