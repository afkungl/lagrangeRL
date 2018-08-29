import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
from .timeContinuousClassificationSmOu import timeContinuousClassificationSmOu


class slimExperiment(timeContinuousClassificationSmOu):

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
        wMaxFixed[-self.layers[-1]:, -self.layers[-1]:] = 1
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

        # Plot the report
        lagrangeRL.tools.visualization.plotLearningReport(Warrays,
                                                          self.avgRArray,
                                                          self.avgRArrays,
                                                          'Output/learningReport.png')
