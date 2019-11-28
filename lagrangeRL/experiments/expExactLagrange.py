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


class expExactLagrange(object):
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
        self.learningRateB = params['learningRateB']
        self.uTarget = params['uTarget']
        self.learningRateH = params['learningRateH']

        # cost weighting parameters
        self.alphaWna = params['alphaWna']
        self.alphaNoise = params['alphaNoise']
        self.beta = params['beta']

        # reporting
        if 'reportFrequency' in params:
            self.reportFrequency = params['reportFrequency']
        else:
            self.reportFrequency = 1

        # save only reward as a default parameter
        if 'saveOnlyReward' in params:
            self.saveOnlyReward = params['saveOnlyReward']
        else:
            self.saveOnlyReward = False

        # Set the random seed for numpy and tensorflow
        np.random.seed(params['randomSeed'])
        tf.set_random_seed(params['randomSeed'])

        # checkpointing is turned off by default
        self.checkpointing = False

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

        self.logger = logging.getLogger('expExactLagrange')
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

    def runSimulation(self, startFrom=0, testing=False):
        """
            Args:
            startFrom: start from a certain iteration number, relevant for continuing from checkpoint
            testing: if True then confusion matrix is gathered
        """

        if testing:
            nLabels =  len(self.params['labels'])
            confMatrix = np.zeros((nLabels, nLabels))
            nTestExamples = self.myData.nTest
            iterRange = range(nTestExamples)
        else:
            iterRange = range(1 + startFrom, self.Niter + 1)

        for index in iterRange:
            [activityLastLayer, trueLabel] = self.singleIteration(
                                                        index,
                                                        testing = testing)
            if index % self.reportFrequency == 0:
                self.plotFinalReport()
                self.saveResults()
            if self.checkpointing and (index % self.checkPerIter == 0):
                self.saveCheckpoint(index)
                self.logger.info('Checkpoint created at {}'.format(index))

            if testing:
                actionIndex = np.argmax(activityLastLayer)
                labelIndex = np.argmax(trueLabel)
                self.logger.debug('Activity in the last layer: {}'.format(activityLastLayer))
                self.logger.debug('Action index: {}'.format(actionIndex))
                self.logger.debug('True label vector: {}'.format(trueLabel))
                self.logger.debug('True label Index: {}',format(labelIndex))
                confMatrix[labelIndex, actionIndex] += 1
                self.logger.debug('The current confusion matrix: {}'.format(confMatrix))
                self.logger.debug('Test example {} evaluated'.format(index))

        if not testing:
            self.plotFinalReport()

        self.logger.debug('The reported conf matrix is: {}'.format(confMatrix))
        return confMatrix

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
        self.simClass = lagrangeRL.network.lagrangeTfDirect()
        self.simClass.setPlasticSynapses(np.logical_not(self.W.mask))
        self.simClass.setLearningRate(self.learningRate)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.addMatrix(self.W)
        self.simClass.setTauEligibility(self.tauElig)
        if self.saveOnlyReward:
            self.simClass.saveTraces(False)
        else:
            self.simClass.saveTraces(True)

        self.simClass.setCostWeightings(self.alphaWna,
                                        self.alphaNoise,
                                        self.beta)
        wMaxFixed = np.zeros((self.N, self.N))
        wMaxFixed[-self.layers[-1]:, -self.layers[-1]:] = 1
        self.simClass.setFixedSynapseMask(wMaxFixed.astype(bool))
        self.simClass.setRegParameters(self.uTarget,
                                       self.learningRateH,
                                       self.uLow,
                                       self.uHigh,
                                       self.learningRateB)
        self.simClass.setNoiseParameter(0.,
                                        self.noiseStd,
                                        self.noiseAutoCorrTime)
        self.simClass.calcWnoWta(self.layers[-1])
        self.simClass.calcOnlyWta(self.layers[-1])

        # set the bias
        biasVector = np.zeros(sum(self.layers))
        biasVector[-self.layers[-1]:] = 0.5
        self.simClass.setBias(biasVector)

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
        self.actFunc = lagrangeRL.tools.activationFunctions.softReluTf(
            1.0, 0.0, 0.3)
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

    def makeOutputFolder(self, overwriteOutput=False):

        if os.path.exists('Output'):
            if not overwriteOutput:
                sys.exit(
                'There is a folder named Output. Delete it to run the simulation! I will NOT do it for you!')
        else:
            os.makedirs('Output')

    def singleIteration(self, index=0, testing=False):

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
            if self.saveOnlyReward:
                self.simClass.applyWeightUpdates(
                    Reward - modulatedAvgReward)
            else:
                self.Wnew = self.simClass.applyWeightUpdates(
                    Reward - modulatedAvgReward)
        self.meanReward = self.meanReward + \
            self.gammaReward * (Reward - self.meanReward)

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
        self.logger.debug("Plastic synapses: {}".format(
                    self.simClass.Wplastic))
        self.logger.debug("The used WTA network {}".format(self.simClass.onlyWta))
        self.logger.debug("The used bias vector is {}".format(
                                self.simClass.getBias()))

        return [output, inputExample['label']]

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

        if self.saveOnlyReward:
            lagrangeRL.tools.visualization.plotMeanReward(
                            self.avgRArray,
                            'Output/meanReward.png',
                            self.avgRArrays)
        else:
            Warray = np.array(self.Warray)
            # Plot the report
            lagrangeRL.tools.visualization.plotLearningReport(
                                    Warray,
                                    self.avgRArray,
                                    self.avgRArrays,
                                    'Output/learningReport.png')

    def saveResults(self):

        if self.saveOnlyReward:
            dictToSave = {'P': {'mean': self.avgRArray}}
        else:
            dictToSave = {'weights': np.array(self.wToOutputArray).tolist(),
                      'P': {'mean': self.avgRArray}}

        for label in self.labels:
            dictToSave['P'][label] = self.avgRArrays[label]

        # Save to the result to output
        with open('Output/results.json', 'w') as outfile:
            json.dump(dictToSave, outfile)

    def enableCheckpointing(self, perIter):
        """
            Turn on the checkpointing

            Args:
                -- perIter: checkpointing every perIter iteration
        """

        self.checkpointing = True
        self.checkPerIter = perIter

    def saveCheckpoint(self, currentIter):

        # create a checkpoint folder if necessary
        if not os.path.exists('Checkpoint'):
            os.makedirs('Checkpoint')

        dictToSave = {'P': {'mean': self.avgRArray}}
        for label in self.params['labels']:
            dictToSave['P'][label] = self.avgRArrays[label]
        dictToSave['wWta'] = self.simClass.getWtaNetwork().tolist()
        dictToSave['wNoWta'] = self.simClass.getNoWtaNetwork().tolist()
        dictToSave['currentIter'] = currentIter

        # Save to the result to output
        fileName = 'Checkpoint/checkpoint_iter{}.json'.format(currentIter)
        with open(fileName, 'w') as outfile:
            json.dump(dictToSave, outfile)

    def loadCheckpoint(self, fileName):

        # Save the specified file
        with open(fileName, 'r') as infile:
            loadDict = json.load(infile)

        # load the arrays
        self.avgRArray = loadDict['P']['mean']
        self.avgRArrays = {}
        self.avgRewards = {}
        for label in self.params['labels']:
            self.avgRArrays[label] = loadDict['P'][str(label)]
            self.avgRewards[label] = self.avgRArrays[label][-1]
        self.meanReward = self.avgRArray[-1]

        # load the current weights
        self.loadWWta = np.array(loadDict['wWta'])
        self.loadNoWta = np.array(loadDict['wNoWta'])

        # Load current iteration
        self.currIter = loadDict['currentIter']

    def continueFromCheckpoint(self):

        ################################
        ## Set up the network again
        self.initLogging()
        self.setUpNetwork()
        self.setUpInput()
        self.setUpActivationFunction()
        self.setUpDataHandler()
        self.setUpRewardScheme()
        self.setUpEmptyTarget()
        loadW = np.ma.masked_array(self.loadWWta + self.loadNoWta,
                                   self.simClass.W.mask)
        self.simClass.W = loadW
        self.simClass.calcWnoWta(self.layers[-1])
        self.simClass.calcOnlyWta(self.layers[-1])
        self.simClass.initCompGraph()
        self.makeOutputFolder(overwriteOutput=True)

        ################################

        # continue
        self.runSimulation(startFrom = self.currIter)

    def runTesting(self, testSet):

        ################################
        ## Set up the network again
        self.initLogging()
        self.setUpNetwork()
        self.setUpInput()
        self.setUpActivationFunction()
        self.setUpDataHandler()
        self.setUpRewardScheme()
        self.setUpEmptyTarget()
        loadW = np.ma.masked_array(self.loadWWta + self.loadNoWta,
                                   self.simClass.W.mask)
        self.simClass.W = loadW
        self.simClass.calcWnoWta(self.layers[-1])
        self.simClass.calcOnlyWta(self.layers[-1])
        self.simClass.initCompGraph()
        self.makeOutputFolder(overwriteOutput=True)

        # Load the test set into the data handler
        self.myData.pathTest = testSet
        self.myData.loadTestSet()

        ################################

        # continue
        confMatrix = self.runSimulation(testing = True)

        return confMatrix