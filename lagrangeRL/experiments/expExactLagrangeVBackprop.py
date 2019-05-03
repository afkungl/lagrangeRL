import lagrangeRL
import numpy as np
import tensorflow as tf
import copy
import time
import sys
import os
import logging
import coloredlogs
import json
from .expExactLagrange import expExactLagrange


class expExactLagrangeVBackprop(expExactLagrange):
    """
        Experiment class with the direct Lagrange model to verify the 
        baackpropagation
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
        self.logger.debug(
            'The w matrix as it comes from the tool function: {}'.format(self.W.data))
        # Lagrange network
        self.simClass = lagrangeRL.network.lagrangeTfDirect()

        # plastic synapses
        # the last layer and the winner-nudges-all circuit stays fixed
        wPlastic = np.logical_not(self.W.mask)
        wPlastic[-self.layers[-1]:,
                 -(self.layers[-1] + self.layers[-2]):-self.layers[-1]] = False
        #wPlastic = False * wPlastic
        self.simClass.setPlasticSynapses(wPlastic)

        self.simClass.setLearningRate(self.learningRate)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        if self.saveOnlyReward:
            self.simClass.saveTraces(False)
        else:
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

        # set the bias
        biasVector = np.zeros(sum(self.layers))
        biasVector[-self.layers[-1]:] = 0.5
        self.simClass.setBias(biasVector)
