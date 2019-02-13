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


class timeContinuousClassificationSmoothed(trialBasedClassification):

    # Inherit the constructor from the trail based smoothed
    __init__ = trialBasedClassificationSmoothed.__dict__["__init__"]
    setUpInput = trialBasedClassificationSmoothed.__dict__["setUpInput"]

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getNextTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.run(self.simTime - self.tRamp)
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

        self.simClass.run(self.tRamp)

        self.plotReport(index, output, example)

        self.logger.info("Iteration {} is done.".format(index))

    def initLogging(self):

        self.logger = logging.getLogger('timeContinuousClassificationSmoothed')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])
