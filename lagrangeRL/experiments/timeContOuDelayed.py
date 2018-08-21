import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .timeContinuousClassificationDelayedRewardSmoothed import timeContinuousClassificationDelayedRewardSmoothed


class timeContOuDelayed(timeContinuousClassificationDelayedRewardSmoothed):

    def __init__(self, params):
        """
            The dictionary contains the relevant parameters for the experiment.
            Get the values from the dictionary
        """

        # list of number of neurons in layers, e.g. [4,2]
        self.layers = params['layers']
        self.N = np.sum(self.layers)
        self.tau = params['tau']  # time-constant of the membrane potential
        self.tauElig = params['tauElig']  # time-constant of the elig trace
        self.sigmaLog = params['sigmaLog']  # width of the activation function
        # learning rate in the three factor update
        self.learningRate = params['learningRate']
        self.weightDecayRate = params['weightDecayRate']
        self.weightDecayTarget = params['weightDecayTarget']
        # nudging constant of the explorational noise
        self.nudging = params['nudging']
        self.simTime = params['simTime']  # simulation time of a single trial
        self.timeStep = params['timeStep']  # timeStep of the simulation
        self.labels = params['labels']  # list of the used labels
        # decay constant in the reward averaging
        self.gammaReward = params['gammaReward']
        self.Niter = params['Niter']  # number of iteartions
        self.dataSet = params['dataSet']  # path to the dataSet
        # reward for true classification
        self.trueReward = params['trueReward']
        # reward for false classification
        self.falseReward = params['falseReward']
        self.initWeightMean = params['initWeightMean']
        self.initWeightWidth = params['initWeightWidth']
        self.wtaStrength = params['wtaStrength']
        self.figSize = tuple(params['figSize'])
        self.tRamp = params['tRamp']
        self.noiseStd = params['noiseStd']
        self.noiseAutoCorrTime = params['noiseAutoCorrTime']
        self.params = params

    def initLogging(self):

        self.logger = logging.getLogger('timeContOuDelayed')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

    def setUpExpNoise(self):
        """
            Set up the nudging on the output layer to realize the explorational noise
        """
        # Set up the nudging class
        mask = np.zeros(self.N)
        mask[self.N - self.layers[-1]:] = 1
        self.nudgingNoise = lagrangeRL.tools.targetModels.ornsteinUhlenbeckTarget(
            mask,
            mean=0.0,
            tau=self.noiseAutoCorrTime,
            standardDiv=self.noiseStd)
        self.simClass.connectTarget(self.nudgingNoise)
