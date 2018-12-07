import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs
import json


class expOptimizedLagrange(object):
    """
        Experiment class for the optimized network
    """

    def __init__(self, params):
        """
            The dictionary contains the relevant parameters for the experiment.
            Get the values from the dictionary
        """

        # Network parameters
        self.layers = params['layers']
        self.N = np.sum(self.layers)
        self.wtaStrength = params['wtaStrength']

        # Dynamics of the network
        self.tau = params['tau']  # time-constant of the membrane potential
        self.tauElig = params['tauElig']  # time-constant of the elig trace
        self.timeStep = params['timeStep']  # timeStep of the simulation

        # Activation function
        self.sigmaLog = params['sigmaLog']  # width of the activation function

        # Training parameters
        self.learningRate = params['learningRate']
        self.Niter = params['Niter']  # number of iteartions
        self.initWeightMean = params['initWeightMean']
        self.initWeightWidth = params['initWeightWidth']

        # Input parameters (time, nudging, ramp)
        self.nudging = params['nudging']
        self.tRamp = params['tRamp']
        self.simTime = params['simTime']  # simulation time of a single trial

        # dataset parameters
        self.labels = params['labels']  # list of the used labels
        self.dataSet = params['dataSet']  # path to the dataSet
        self.figSize = tuple(params['figSize'])

        # reward parameters
        self.gammaReward = params['gammaReward']  # decay of the reward
        self.trueReward = params['trueReward']
        self.falseReward = params['falseReward']

        # Noise parameters
        self.noiseStd = params['noiseStd']
        self.noiseAutoCorrTime = params['noiseAutoCorrTime']

        # regularization parameters
        self.uLow = params['uLow']
        self.uHigh = params['uHigh']
        self.kappaDecay = params['kappaDecay']

        # cost weighting parameters
        self.alphaWna = params['alphaWna']
        self.alphaNoise = params['alphaNoise']
        self.beta = params['beta']

        # reporting
        if 'reportFrequency' in params:
            self.reportFrequency = params['reportFrequency']
        else:
            self.reportFrequency = 1

        self.params = params

    def initialize(self):
        """
            Set up the simulation
        """

        self.initLogging()
        self.setUpNetwork()
        self.setUpInput()
        self.setUpActivationFunction()
        self.setUpDataHandler()
        self.setUpRewardScheme()
        self.setUpEmptyTarget()
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
                self.saveResults()

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
            noiseMagnitude=self.initWeightWidth,
            noWtaMask=True)
        self.logger.debug('The w matrix as it comes from the tool function: {}'.format(self.W.data))
        # Lagrange network
        self.simClass = lagrangeRL.network.lagrangeTfOptimized2()
        self.simClass.setPlasticSynapses(np.logical_not(self.W.mask))
        self.simClass.setLearningRate(self.learningRate)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        #self.simClass.setNudging(self.nudging)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        self.simClass.saveTraces(True)
        self.simClass.setCostWeightings(self.alphaWna,
                                        self.alphaNoise,
                                        self.beta)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[-self.layers[-1]:, -self.layers[-1]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))
        self.simClass.setRegParameters(self.uLow,
                                       self.uHigh,
                                       self.kappaDecay)
        self.simClass.setNoiseParameter(0.,
                                        self.noiseStd,
                                        self.noiseAutoCorrTime)
        self.simClass.calcWnoWta(self.layers[-1])
        self.simClass.calcOnlyWta(self.layers[-1])

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

    def setUpEmptyTarget(self):
        """
            Set up the target nudging on the output layer. This is necessary to get a target mask
        """
        # Set up the empty target
        value = np.zeros(self.N)
        mask = np.zeros(self.N)
        mask[self.N - self.layers[-1]:] = 1
        self.target = lagrangeRL.tools.targetModels.constantTarget(
            value,
            mask)
        self.simClass.connectTarget(self.target)

    def setUpActivationFunction(self):
        """
            set up the activation function
        """
        # Connect to activation function
        self.actFunc = lagrangeRL.tools.activationFunctions.hardSigmoidTf(
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

    def setUpSavingArrays(self):

        # Set up arrays to save results from the simulation
        self.avgRewards = {}
        self.meanReward = 0.
        self.avgRArrays = {}
        for label in self.labels:
            self.avgRewards[label] = 0
            self.avgRArrays[label] = []
        self.avgRArray = []
        self.deltaW = copy.deepcopy(self.W.data)
        self.deltaW = 0. * self.deltaW
        self.deltaWBatch = 0. * copy.deepcopy(self.deltaW)
        self.Warray = []
        self.wToOutputArray = []
        # Save the starting arrays
        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(
            self.simClass.W.data[self.layers[-2]:, :self.layers[-2]].flatten())

    def makeOutputFolder(self):

        if os.path.exists('Output'):
            sys.exit(
                'There is a folder named Output. Delete it to run the simulation! I will NOT do it for you!')
        else:
            os.makedirs('Output')

    def singleIteration(self, index=0):

        # get an example as input
        inputExample = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = inputExample['data']

        # run the simulation before the ramp downstart
        self.simClass.run(self.simTime - self.tRamp)

        # get the output and obtain the reward
        #output = self.simClass.getMembPotentials()[self.N - self.layers[-1]:]
        output = self.simClass.getLowPassActivity()[self.N - self.layers[-1]:]
        trueLabel = self.labels[np.argmax(inputExample['label'])]
        Reward = self.rewardScheme.obtainReward(inputExample['label'], output)
        self.avgRewards[trueLabel] = self.avgRewards[trueLabel] + \
            self.gammaReward * (Reward - self.avgRewards[trueLabel])

        # save the averaged reward array
        self.avgRArray.append(self.meanReward)
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgRewards[key])

        # Update the weights
        # The reward goes modulated into the update formula.
        # This is necessary such that the well predicted negativ reward does
        # cause the learning to stop but a well predicted positiv reward does
        if index == 1:
            self.Wnew = self.simClass.applyWeightUpdates(Reward)
        else:
            modulatedAvgReward = np.max([self.meanReward, 0.0])
            self.Wnew = self.simClass.applyWeightUpdates(
                Reward - modulatedAvgReward)
        self.meanReward = self.meanReward + \
            self.gammaReward * (Reward - self.meanReward)

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
        self.logger.debug("The used WTA network {}".format(self.simClass.onlyWta))
        self.logger.debug("The used network without WTA {}".format(self.simClass.WnoWta))

    def plotReport(self, index, output, example):

        # Plot the report
        figName = 'Output/reportIteration{}.png'.format(index)
        traces = self.simClass.getTraces()
        outputU = self.simClass.getMembPotentials()[-self.layers[-1]:]
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
        wCurrent = self.Wnew
        eligs = self.simClass.getEligibilities()
        eligs[:self.layers[0],:self.layers[0]] = 0.
        plastNow = self.simClass.getPlastNow()
        error = self.simClass.getErrorNow()[-self.layers[-1]:]
        errorHidden = self.simClass.getErrorNow()[self.layers[0]:-self.layers[-1]]
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
            plastNow,
            error,
            errorHidden,
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

    def saveResults(self):

        dictToSave = {'weights': np.array(self.wToOutputArray).tolist(),
                      'P': {'mean': self.avgRArray}}

        for label in self.labels:
            dictToSave['P'][label] = self.avgRArrays[label]

        # Save to the result to output
        with open('Output/results.json', 'w') as outfile:
            json.dump(dictToSave, outfile)
