import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
import json
from .timeContinuousClassificationSmOu import timeContinuousClassificationSmOu


class slimExperimentVerifyBp(timeContinuousClassificationSmOu):

    def initLogging(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

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
        self.simClass = lagrangeRL.network.lagrangeEligTfApprox()
        self.simClass.setLearningRate(1.)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(False)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[self.layers[0]:, self.layers[0]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

    def plotReport(self, index, output, example):

        # Plot the report
        figName = 'Output/reportIteration{}.png'.format(index)
        outputU = output
        outputRho = self.simClass.getActivities()[self.N - self.layers[-1]:]
        target = np.argmax(example['label']) + 1
        data = example['data']
        wCurrent = self.simClass.WnoWta
        eligs = self.simClass.getEligibilities().T
        signDeltaW = np.sign(self.deltaW.T)
        lagrangeRL.tools.visualization.plotReportNoTraces(
            figName,
            self.timeStep,
            outputU,
            outputRho,
            target,
            data,
            self.figSize,
            wCurrent,
            eligs,
            signDeltaW,
            simTime=self.simTime)

    def plotFinalReport(self):
        """
            Plot a final report about the results of the simulation
        """
        #W = self.simClass.W[self.layers[-1]:,:self.layers[-1]]
        Warrays = np.array(self.wToOutputArray)
        print(Warrays.shape)

        # Plot the report
        lagrangeRL.tools.visualization.plotLearningReport(Warrays,
                                                          self.avgRArray,
                                                          self.avgRArrays,
                                                          'Output/learningReport.png')

    def runSimulation(self):

        for index in range(1, self.Niter + 1):
            self.singleIteration(index)
            if index % 10 == 0:
                self.plotFinalReport()

        self.plotFinalReport()

        self.saveResults()

    def saveResults(self):

        dictToSave = {'weights': np.array(self.wToOutputArray).tolist(),
                      'P': {'mean': self.avgRArray}}

        for label in self.labels:
            dictToSave['P'][label] = self.avgRArrays[label]

        # Save to the result to output
        with open('Output/results.json', 'w') as outfile:
            json.dump(dictToSave, outfile)

    def setUpSavingArrays(self):

        # Set up arrays to save results from the simulation
        self.avgR = {}
        self.avgRArrays = {}
        for label in self.labels:
            self.avgR[label] = 0
            self.avgRArrays[label] = []
        self.avgRArray = []
        self.deltaW = copy.deepcopy(self.W.data)
        self.deltaW = 0. * self.deltaW
        self.deltaWBatch = 0. * copy.deepcopy(self.deltaW)
        self.Warray = []
        self.wToOutputArray = []
        # Save the starting arrays
        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(self.simClass.W.data[
                                        self.layers[0]:self.layers[0] + self.layers[1],
                                        0:self.layers[0]].flatten())

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.run(self.simTime - self.tRamp, updateNudging=True)
        output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]

        # obtain reward
        self.logger.debug("The label vector is {}".format(example['label']))
        self.logger.debug("The output vector is {}".format(output))
        trueLabel = self.labels[np.argmax(example['label'])]
        self.logger.debug("The true label is: {}".format(trueLabel))
        self.logger.info("The current average reward is: {}".format(self.avgR))
        R = self.rewardScheme.obtainReward(example['label'], output)
        self.logger.info("The obtained reward is {}".format(R))
        self.avgR[trueLabel] = self.avgR[trueLabel] + \
            self.gammaReward * (R - self.avgR[trueLabel])
        self.avgRArray.append(np.mean(self.avgR.values()))
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])

        # Update the weights
        modavgR = np.min([np.max([self.avgR[trueLabel], 0.]), 1.])
        self.logger.debug('The avgR for the label {0} is {1}'.format(
            trueLabel, self.avgR[trueLabel]))
        self.logger.debug('The modavgR for the label {0} is {1}'.format(
            trueLabel, modavgR))
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR)
        self.deltaWBatch += self.deltaW

        # if index % self.layers[-1] == 0:
        if index % 1 == 0:
            self.deltaWBatch += -1. * self.weightDecayRate * \
                (copy.deepcopy(self.simClass.W.data) - self.weightDecayTarget)
            self.simClass.applyWeightUpdates(self.deltaWBatch, self.cap)
            self.simClass.calcWnoWta(self.layers[-1])
            self.logger.debug(
                'The applied weigth changes: {}'.format(self.deltaWBatch))
            self.deltaWBatch = 0. * self.deltaWBatch

        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(self.simClass.W.data[
                                        self.layers[0]:self.layers[0] + self.layers[1],
                                        0:self.layers[0]].flatten())

        self.plotReport(index, output, example)

        self.simClass.run(self.tRamp, updateNudging=True)

        self.simClass.deleteTraces()

        self.logger.info("Iteration {} is done.".format(index))
        self.logger.debug(
            "The current weights are: {}".format(self.simClass.W))
