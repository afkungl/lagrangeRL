import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs


class trialBasedClassification(object):
    """
        This class contains the implementation of the trial based classification with reward.
        The setup:
            --- in a feedforward network the input is presented as membrane voltage
            --- the last layer is a WTA layer with the same number of neurons as the number of classes
            --- the dynamics of the network runs according to the vanilla Lagrange model
            --- the weight updates are gathered in an eligibility trace
            --- weight updates are applied at the end of a trial when a reward +1/-1 is given
            --- the network is reset between the trials
            --- this implementation uses hard-coded sigmoid activation function

    """

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
        self.params = params

    def initialize(self):
        """
            Set up the simulation
        """

        self.initLogging()
        self.setUpNetwork()
        self.setUpInput()
        self.setUpExpNoise()
        self.setUpActivationFunction()
        self.setUpDataHandler()
        self.setUpRewardScheme()
        self.simClass.initCompGraph()
        self.setUpSavingArrays()
        self.makeOutputFolder()

    def initLogging(self):

        self.logger = logging.getLogger('trialBasedClassification')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

    def runSimulation(self):

        for index in range(1, self.Niter + 1):
            self.singleIteration(index)
            if index % 25 == 0:
                self.plotFinalReport()

        self.plotFinalReport()

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
        self.simClass = lagrangeRL.network.lagrangeEligTf()
        self.simClass.setLearningRate(1.)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(True)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[-self.layers[-1]:, -self.layers[-1]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

    def setUpInput(self):
        """
            Set up the input
        """
        # connect to input
        value = np.ones(self.N)
        mask = np.zeros(self.N)
        mask[:self.layers[0]] = 1.
        self.Input = lagrangeRL.tools.inputModels.constantInput(
            value,
            mask)
        self.simClass.connectInput(self.Input)

    def setUpExpNoise(self):
        """
            Set up the nudging on the output layer to realize the explorational noise
        """
        # Set up the nudging class
        value = np.ones(self.N)
        mask = np.zeros(self.N)
        mask[self.N - self.layers[-1]:] = 1
        self.nudgingNoise = lagrangeRL.tools.targetModels.constantTarget(
            value,
            mask)
        self.simClass.connectTarget(self.nudgingNoise)

    def setUpActivationFunction(self):
        """
            set up the activation function
        """
        # Connect to activation function
        self.actFunc = lagrangeRL.tools.activationFunctions.sigmoidTf(
            self.sigmaLog)
        self.simClass.connectActivationFunction(self.actFunc)

    def setUpDataHandler(self):
        # set up the dataHandler
        self.myData = lagrangeRL.tools.dataHandler.dataHandlerMnist(
            self.labels,
            self.dataSet,
            self.dataSet)
        self.myData.loadTestSet()
        self.myData.loadTrainSet()

    def setUpRewardScheme(self):
        # set up the reward scheme
        self.rewardScheme = lagrangeRL.tools.rewardSchemes.maxClassification(
            self.trueReward,
            self.falseReward)
        self.simClass.calcWnoWta(self.layers[-1])

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

    def makeOutputFolder(self):

        if os.path.exists('Output'):
            sys.exit(
                'There is a folder named Output. Delete it to run the simulation! I will not do it for you!')
        else:
            os.makedirs('Output')

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getNextTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

        # get and set random noise
        noiseVector = 2 * (np.random.randint(2, size=self.layers[-1]) - 0.5)
        self.nudgingNoise.value[self.N - self.layers[-1]:] = noiseVector

        # run the simulation
        self.simClass.resetCompGraph()
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
        modavgR = np.max([self.avgR[trueLabel], 0.])
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR)
        self.deltaWBatch += self.deltaW

        if index % self.layers[-1] == 0:
            self.deltaWBatch += -1. * self.weightDecayRate * \
                (copy.deepcopy(self.simClass.W.data) - self.weightDecayTarget)
            self.simClass.applyWeightUpdates(self.deltaWBatch)
            self.simClass.calcWnoWta(self.layers[-1])
            self.logger.debug('The applied weigth changes: {}'.format(self.deltaWBatch))
            self.deltaWBatch = 0. * self.deltaWBatch

        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])

        self.plotReport(index, output, example)

        self.logger.info("Iteration {} is done.".format(index))

    def plotReport(self, index, output, example):

        # Plot the report
        figName = 'Output/reportIteration{}.png'.format(index)
        traces = self.simClass.getTraces()
        outputU = output
        outputRho = self.simClass.getActivities()[self.N - self.layers[-1]:]
        target = np.argmax(example['label']) + 1
        data = example['data']
        """
        wCurrent = self.simClass.W.data[
            self.layers[0]:, :self.N - self.layers[1]].T
        eligs = self.simClass.getEligibilities(
        )[self.layers[0]:, :self.N - self.layers[1]].T
        signDeltaW = np.sign(
            self.deltaW[self.layers[0]:, :self.N - self.layers[1]].T)
        """
        wCurrent = self.simClass.W.data.T
        eligs = self.simClass.getEligibilities().T
        signDeltaW = np.sign(self.deltaW.T)
        lagrangeRL.tools.visualization.plotReport(
            figName,
            self.timeStep,
            traces,
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

        Warray = np.array(self.Warray)

        # Plot the report
        lagrangeRL.tools.visualization.plotLearningReport(Warray,
                                                          self.avgRArray,
                                                          self.avgRArrays,
                                                          'Output/learningReport.png')
