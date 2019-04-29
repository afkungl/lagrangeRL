import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
import json
from .expApproxLagrange import expApproxLagrange


class expApproxLagrangeVerifyBp(expApproxLagrange):
    """
        Experiment class for the optimized network
    """

    def setUpNetwork(self):
        """
            Set up the network with the lagrange dynamics
        """

        # Set up the network structure
        self.W = lagrangeRL.tools.networks.feedForwardWtaReadout(
            self.layers,
            self.wtaStrength,
            offset=self.initWeightMean,
            noiseMagnitude=self.initWeightWidth,
            noWtaMask=True)
        # Lagrange network
        self.simClass = lagrangeRL.network.lagrangeTfOptimized()
        if len(self.layers) != 3:
            self.logger.error(
                "A network with {} layers was specified! This experiment is designed only for networks with 3 layers.".format(len(self.layers)))
            sys.exit()
        Wplastic = np.zeros((self.N, self.N))
        Wplastic[self.layers[0]:self.layers[0] + self.layers[1], :self.layers[0]] = 1
        self.simClass.setPlasticSynapses(Wplastic)
        self.simClass.setLearningRate(self.learningRate)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(True)
        self.simClass.setCostWeightings(self.alphaWna,
                                        self.alphaNoise,
                                        self.beta)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[-self.layers[-1]:, -self.layers[-1]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))
        self.simClass.setRegParameters(self.uLow,
                                       self.uHigh,
                                       self.kappaDecay)
        self.simClass.setNoiseParameter(0.,
                                        self.noiseStd,
                                        self.noiseAutoCorrTime)
        self.simClass.calcWnoWta(self.layers[-1])
        self.simClass.calcOnlyWta(self.layers[-1])
