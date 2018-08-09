import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .trialBasedClassification import trialBasedClassification


class timeContinuousClassificationDelayedReward(trialBasedClassification):

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.run(self.simTime)
        output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]

        # obtain the reward. The reward from the previous example is provided
        # to the network. The reward now is seved in self.previousReward to be
        # used later
        nowTrueLabel = np.argmax(example['label'])
        self.logger.debug("The true label is: {}".format(nowTrueLabel))
        self.logger.info("The current average reward is: {}".format(self.avgR))
        nowR = self.rewardScheme.obtainReward(example['label'], output)
        if index == 1:
            # In the first iteration there is no previous iteration, hence we
            # provide a reward of 0. This means that no update is made
            R = 0
            trueLabel = self.labels[0]
        else:
            R = self.previousR
            trueLabel = self.previousLabel
        self.previousR = nowR
        self.previousLabel = nowTrueLabel

        # Calculate the moving average of the reward
        self.avgR[trueLabel] = self.avgR[trueLabel] + \
            self.gammaReward * (R - self.avgR[trueLabel])
        self.avgRArray.append(np.mean(self.avgR.values()))
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])

        # Update the weights
        modavgR = np.min([np.max([self.avgR[trueLabel], 0.]), 0.95])
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR)
        self.deltaWBatch += self.deltaW

        # first try: apply the weight updates after every iteration
        if index % 1 == 0:
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

        self.logger = logging.getLogger(
            'timeContinuousClassificationDelayedReward')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])
