import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
import json
from .expOptimizedLagrange import expOptimizedLagrange


class expOptimizedLagrangeDelay(expOptimizedLagrange):
    """
        Experiment class for the optimized network
    """

    def singleIteration(self, index=0):

        # get an example as input
        inputExample = self.myData.getNextTestExample()[0]
        self.Input.value[:self.layers[0]] = inputExample['data']

        # run the simulation before the ramp downstart
        self.simClass.run(self.simTime - self.tRamp)

        # get the output and obtain the reward
        output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]
        trueLabel = self.labels[np.argmax(inputExample['label'])]
        Reward = self.rewardScheme.obtainReward(inputExample['label'], output)
        self.avgRewards[trueLabel] = self.avgRewards[trueLabel] + \
            self.gammaReward * (Reward - self.avgRewards[trueLabel])

        # save the averaged reward array
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgRewards[key])

        # Update the weights
        # The reward goes modulated into the update formula.
        # This is necessary such that the well predicted negativ reward does
        # cause the learning to stop but a well predicted positiv reward does
        if index in [1, 2]:
            self.Wnew = self.simClass.applyWeightUpdates(0.)
            self.avgRArray.append(0)
        else:
            modulatedAvgReward = np.max([self.avgRArray[-1], -0.90])
            self.Wnew = self.simClass.applyWeightUpdates(
                self.rewardOld - modulatedAvgReward)
            self.meanReward = self.meanReward + self.gammaReward * \
                (self.rewardOld - self.meanReward)
            self.avgRArray.append(self.meanReward)
        self.rewardOld = Reward

        # save the weights in an array
        self.Warray.append(self.Wnew[~self.simClass.W.mask])
        self.wToOutputArray.append(
            self.simClass.W.data[self.layers[-2]:, :self.layers[-2]].flatten())

        # Plot reports
        if index % self.reportFrequency == 0:
            self.plotReport(index, output, inputExample)

        # run the simulation of the example until ramp down
        self.simClass.run(self.tRamp)

        # Delete the traces to avoid memory-overflow
        self.simClass.deleteTraces()

        # Log intermediate results
        self.logger.info("The obtained reward is {}".format(Reward))
        self.logger.info(
            "The current average reward is: {}".format(self.avgRArray[-1]))
        self.logger.debug("The current weights: {}".format(self.Wnew))
        self.logger.info("Iteration {} is done.".format(index))
        self.logger.debug("No WTA mask: {}".format(
            self.simClass.sess.run(self.simClass.wNoWtaMask)))
