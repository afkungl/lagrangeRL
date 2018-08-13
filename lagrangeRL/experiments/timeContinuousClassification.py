import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .trialBasedClassification import trialBasedClassification


class timeContinuousClassification(trialBasedClassification):

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getNextTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.run(self.simTime)
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
        modavgR = np.max([self.avgR[trueLabel], 0.]), 0.95
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

        self.plotReport(index, output, example)

        self.logger.info("Iteration {} is done.".format(index))

    def initLogging(self):

        self.logger = logging.getLogger('timeContinuousClassification')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])
