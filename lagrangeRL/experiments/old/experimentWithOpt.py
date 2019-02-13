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
        # nudging constant of the explorational noise
        self.nudging = params['nudging']
        self.alphaWna = params['alphaWna']
        self.alphaNoise = params['alphaNoise']
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
        
        # regularization to keep the WTA in the range
        self.uLow = params['uLow']
        self.uHigh = params['uHigh']
        self.lambdaReg = params['lambdaReg']

        # Load the params for fall-back solutions
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
        self.setUpWeightDecay()
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
            if index % 10 == 0:
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
        self.simClass.setLearningRate(self.learningRate)
        self.simClass.setNoiseParameter(mean=0.,
                                        std=self.noiseStd,
                                        corrTime=self.noiseAutoCorrTime)
        self.simClass.calcWnoWta(layers[-1])
        self.simClass.calcOnlyWta(layers[-1])
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(True)
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

    def setUpInput(self):
        """
            Set up the input
        """
        # connect to input
        value = np.ones(self.N)
        mask = np.zeros(self.N)
        mask[:self.layers[0]] = 1.
        self.Input = lagrangeRL.tools.inputModels.smoothedConstantInput(
            value,
            mask,
            self.simTime,
            self.tRamp)
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
        self.wToOutputArray = []
        # Save the starting arrays
        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(self.simClass.W.data[self.layers[-2]:,:self.layers[-2]].flatten())

    def makeOutputFolder(self):

        if os.path.exists('Output'):
            sys.exit(
                'There is a folder named Output. Delete it to run the simulation! I will not do it for you!')
        else:
            os.makedirs('Output')

    def singleIteration(self, index=0):

        # get and set example
        example = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = example['data']

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
        self.avgRFull = self.avgRFull + \
            self.gammaReward * (R - self.avgRFull)
        self.avgRArray.append(self.avgRFull)
        self.avgR[trueLabel] = self.avgR[trueLabel] + \
            self.gammaReward * (R - self.avgR[trueLabel])
        self.instantRArray.append({trueLabel: R})
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])
            

        # Update the weights
        modavgR = np.min([np.max([self.avgRFull, -0.9]), 1.])
        self.logger.debug('The avgR for the label {0} is {1}'.format(
            trueLabel, self.avgR[trueLabel]))
        self.logger.debug('The modavgR for the label {0} is {1}'.format(
            trueLabel, modavgR))
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR,
                                                           self.lambdaReg)
        self.deltaWBatch += self.deltaW

        # if index % self.layers[-1] == 0:
        if index % 1 == 0:
            self.simClass.applyWeightUpdates(self.deltaWBatch)
            self.simClass.calcWnoWta(self.layers[-1])
            self.logger.debug(
                'The applied weigth changes: {}'.format(self.deltaWBatch))
            self.deltaWBatch = 0. * self.deltaWBatch

        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(self.simClass.W.data[self.layers[-2]:,:self.layers[-2]].flatten())

        self.plotReport(index, output, example)

        self.simClass.run(self.tRamp, updateNudging=True)

        self.simClass.deleteTraces()

        self.logger.info("Iteration {} is done.".format(index))
        self.logger.debug(
            "The current weights are: {}".format(self.simClass.W))

    def plotReport(self, index, output, example):

        # Plot the report
        figName = 'Output/reportIteration{}.png'.format(index)
        traces = self.simClass.getTraces()
        outputU = output
        outputRho = self.simClass.getActivities()[self.N - self.layers[-1]:]
        target = np.argmax(example['label']) + 1
        data = example['data']
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

    def setUpWeightDecay(self):

        self.weightDecay = lagrangeRL.tools.weightDecayModels.flatValleyL2Decay(
            self.lowerValley,
            self.upperValley,
            self.kappaDecay)
        self.simClass.connectWeightDecay(self.weightDecay)

        if self.cap == "None":
            self.cap = None
            self.logger.info('The weights clipping is NOT active')