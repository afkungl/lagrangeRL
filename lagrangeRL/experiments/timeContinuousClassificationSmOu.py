import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .trialBasedClassification import trialBasedClassification
from .trialBasedClassificationSmoothed import trialBasedClassificationSmoothed


class timeContinuousClassificationSmOu(trialBasedClassification):

    # Inherit the constructor from the trail based smoothed
    setUpInput = trialBasedClassificationSmoothed.__dict__["setUpInput"]

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

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getNextTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.run(self.simTime - self.tRamp, updateNudging=True)
        output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]

        # obtain reward
        trueLabel = np.argmax(example['label'])
        self.logger.debug("The true label is: {}".format(trueLabel))
        self.logger.info("The current average reward is: {}".format(self.avgR))
        R = self.rewardScheme.obtainReward(example['label'], output)
        self.avgR[trueLabel] = self.avgR[trueLabel] + \
            self.gammaReward * (R - self.avgR[trueLabel])
        self.avgRArray.append(np.mean(self.avgR.values()))
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])

        # Update the weights
        modavgR = np.min([np.max([self.avgR[trueLabel], 0.]), 1.])
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR)
        self.deltaWBatch += self.deltaW

        if index % self.layers[-1] == 0:
            self.deltaWBatch += -1. * self.weightDecayRate * \
                (copy.deepcopy(self.simClass.W.data) - self.weightDecayTarget)
            self.simClass.applyWeightUpdates(self.deltaWBatch)
            self.simClass.calcWnoWta(self.layers[-1])
            self.logger.debug(
                'The applied weigth changes: {}'.format(self.deltaWBatch))
            self.deltaWBatch = 0. * self.deltaWBatch

        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])

        self.simClass.run(self.tRamp, updateNudging=True)

        self.plotReport(index, output, example)

        self.simClass.deleteTraces()

        self.logger.info("Iteration {} is done.".format(index))

    def initLogging(self):

        self.logger = logging.getLogger('timeContinuousClassificationSmoothed')
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
