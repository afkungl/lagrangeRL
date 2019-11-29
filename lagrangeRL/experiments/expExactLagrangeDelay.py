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


class expExactLagrangeDelay(expExactLagrange):
    """
        Experiment class for the optimized network.
        The reward arrives delayed by one trial
    """

    def singleIteration(self, index=0, testing=False):

        # The od reward is assumed to be zero at the first iteration
        if index == 1:
            self.oldReward = 0

        # get an example as input
        if testing:
            inputExample = self.myData.getNextTestExample()[0]
        else:
            inputExample = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = inputExample['data']

        # run the simulation before the ramp downstart
        self.simClass.run(self.simTime - self.tRamp)

        # get the output and obtain the reward
        #output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]
        output = self.simClass.getLowPassActivity()[self.N - self.layers[-1]:]
        trueLabel = self.labels[np.argmax(inputExample['label'])]
        Reward = self.rewardScheme.obtainReward(inputExample['label'], output)
        oldReward = self.oldReward
        self.oldReward = Reward
        self.avgRewards[trueLabel] = self.avgRewards[trueLabel] + \
            self.gammaReward * (oldReward - self.avgRewards[trueLabel])

        # save the averaged reward array
        self.avgRArray.append(self.meanReward)
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgRewards[key])

        # Update the weights
        # The reward goes modulated into the update formula.
        # This is necessary such that the well predicted negativ reward does
        # cause the learning to stop but a well predicted positiv reward does
        if index == 1:
            self.Wnew = self.simClass.applyWeightUpdates(oldReward)
        else:
            modulatedAvgReward = np.max([self.meanReward, 0.0])
            if self.saveOnlyReward:
                self.simClass.applyWeightUpdates(
                    oldReward - modulatedAvgReward)
            else:
                self.Wnew = self.simClass.applyWeightUpdates(
                    oldReward - modulatedAvgReward)
        self.meanReward = self.meanReward + \
            self.gammaReward * (oldReward - self.meanReward)

        # save the weights in an array
        if not self.saveOnlyReward:
            self.Warray.append(self.Wnew[~self.simClass.W.mask])
            self.wToOutputArray.append(
                self.simClass.W.data[self.layers[-2]:, :self.layers[-2]].flatten())

        # Plot reports
        if (index % self.reportFrequency == 0) and not self.saveOnlyReward:
            self.plotReport(index, output, inputExample)

        # run the simulation of the example until ramp down
        self.simClass.run(self.tRamp)

        # Delete the traces to avoid memory-overflow
        if not self.saveOnlyReward:
            self.simClass.deleteTraces()

        # Log intermediate results
        self.logger.info("The obtained reward is {}".format(Reward))
        self.logger.info(
            "The current average reward is: {}".format(self.avgRArray[-1]))
        
        self.logger.info("Iteration {} is done.".format(index))
        
        if not self.saveOnlyReward:
            self.logger.debug("The current weights: {}".format(self.Wnew))
        self.logger.debug("No WTA mask: {}".format(
                    self.simClass.sess.run(self.simClass.wNoWtaMask)))
        self.logger.debug("The used WTA network {}".format(self.simClass.onlyWta))
        self.logger.debug("The used bias vector is {}".format(
                                self.simClass.getBias()))

        return [output, inputExample['label']]