#!/usr/bin/env python

from mlModel.network import mlNetwork
import tensorflow as tf
import json
import logging
import coloredlogs
from mlModel.aux import tfAux
from lagrangeRL import tools
from lagrangeRL.tools import visualization
import numpy as np
import os
from lagrangeRL.tools import activationFunctions


class basicExperiment(object):
    """
        The basic Experiment class

        This experiment works for a predefined number of steps
        contant learning rate and separate trials
    """

    def initLogging(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

    def __init__(self, jsonFile, overwriteOutput=False):
        """
            Load the jsonFile and check if it contains all the necessary parameters

        """

        necessaryParams = ['layers',
                           'learningRate',
                           'labels',
                           'gammaReward',
                           'Niter',
                           'dataSet',
                           'trueReward',
                           'falseReward',
                           'logLevel',
                           'reportFreq',
                           'randomSeed']

        # Load the parameters from the json file
        with open(jsonFile, 'r') as f:
            self.params = json.load(f)

        # check if all the parameters are in the dictionary
        for parameter in necessaryParams:
            if not parameter in self.params:
                raise RuntimeError(
                    'The parameter {} is missing from the parameter file!'.format(parameter))

        # Start the logger
        self.initLogging()

        # Make a folder for the Output file
        # If the Output file exists then stop the simulation
        if not os.path.exists('Output'):
            os.makedirs('Output')
        elif not overwriteOutput:
            raise RuntimeError(
                'Idiot check! An <<Output>> folder exists. Delete it to proceed!')

        # checkpointing is turned off by default
        self.checkpointing = False

        # Set the random seed for numpy and tensorflow
        # for the sake of simplicity we use the same seed
        np.random.seed(self.params['randomSeed'])
        tf.set_random_seed(self.params['randomSeed'])

    def initializeExperiment(self):

        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetwork(self.params['layers'],
                                             self.actFunc.value)
        self.networkTf.initialize()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # Set up arrays and parameters to save the progress
        self.meanR = 0
        self.meanRArray = [0]
        self.meanRArrayClass = {}
        for label in self.params['labels']:
            self.meanRArrayClass[label] = [0]
        self.currentRArray = []

    def singleIteration(self):
        """
            Perform the single iteration:
                --- take an action
                --- obtain a reward
                --- save the reward into the mean reward and the rewards
                    by class
                --- update the parameters
        """

        # Get a random input example
        example = self.dataHandler.getRandomTrainExample()[0]
        currentLabel = self.params['labels'][np.argmax(example['label'])]

        # Take an action
        actionVector = self.networkTf.getActionVector(example['data'])
        actionIndex = np.argmax(actionVector)
        actionLabel = self.params['labels'][actionIndex]

        # Observe reward
        currentReward = self.rewardScheme.obtainReward(example['label'],
                                                       actionVector)

        # Update the parameters
        modulator = currentReward - self.meanRArray[-1]
        self.networkTf.updateParameters(example['data'],
                                        actionVector,
                                        modulator,
                                        self.params['learningRate'])

        # Save the observed rewards into the respective arrays
        self.currentRArray.append(currentReward)
        newMeanR = self.params['gammaReward'] * self.meanRArray[-1] + \
            (1. - self.params['gammaReward']) * currentReward
        self.meanRArray.append(newMeanR)

        for label in self.params['labels']:
            if currentLabel == label:
                # If the label is the same as we have just tested then
                # update the array with the new reward
                newMeanRClass = self.params['gammaReward'] * \
                    self.meanRArrayClass[label][-1] + \
                    (1. - self.params['gammaReward']) * currentReward
                self.meanRArrayClass[label].append(newMeanRClass)
            else:
                # For any other label the mean weight stays
                self.meanRArrayClass[label].append(
                    self.meanRArrayClass[label][-1])

    def runFullExperiment(self, startFrom=0):
        """
            Run the full experiment for the defined number of iterations
        """

        # loop through the experiment
        for index in xrange(startFrom + 1, self.params['Niter'] + 1):
            self.singleIteration()
            self.logger.info('Iteration number {} finished.'.format(index + 1))
            # Report the weights in the last layer for debugging
            self.logger.debug('Weights in last layer {}'.format(
                    self.networkTf._getLastLayerWeights()))

            if index % self.params['reportFreq'] == 0:
                visualization.plotMeanReward(self.meanRArray,
                                             'Output/meanReward.png',
                                             self.meanRArrayClass)
                self.logger.info('Mean reward plotted.')

            if self.checkpointing and (index % self.checkPerIter == 0):
                self.saveCheckpoint(index)

        # plot the results
        visualization.plotMeanReward(self.meanRArray,
                                     'Output/meanReward.png',
                                     self.meanRArrayClass)

        # save the results
        self.saveResults()

    def saveResults(self):

        dictToSave = {'P': {'mean': self.meanRArray}}

        for label in self.params['labels']:
            dictToSave['P'][label] = self.meanRArrayClass[label]

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

        dictToSave = {'P': {'mean': self.meanRArray}}
        for label in self.params['labels']:
            dictToSave['P'][label] = self.meanRArrayClass[label]
        wLists = []
        for w in self.networkTf.getWeights():
            wLists.append(w.tolist())
        dictToSave['weights'] = wLists
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
        self.meanRArray = loadDict['P']['mean']
        self.meanRArrayClass = {}
        for label in self.params['labels']:
            self.meanRArrayClass[label] = loadDict['P'][str(label)]

        # load the current weights
        self.currentWs = []
        for w in loadDict['weights']:
            self.currentWs.append(np.array(w))

        # Load current iteration
        self.currIter = loadDict['currentIter']
        self.currentRArray = []

    def continueFromCheckpoint(self):

        # Set up the experiment again
        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetwork(self.params['layers'],
                                             self.actFunc.value)
        self.networkTf.getInitialWeights(self.currentWs)
        self.networkTf._createComputationalGraph()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # continue experiment
        st = self.currIter + 1
        self.runFullExperiment(startFrom=st)

    def runTesting(self, testSetFile):

        # Set up the experiment again
        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetwork(self.params['layers'],
                                             self.actFunc.value)
        self.networkTf.getInitialWeights(self.currentWs)
        self.networkTf._createComputationalGraph()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            testSetFile,
            self.params['dataSet'])

        self.dataHandler.loadTestSet()

        # Set up variables and arrays
        nLabels =  len(self.params['labels'])
        confMatrix = np.zeros((nLabels, nLabels))
        nTestExamples = self.dataHandler.nTest

        for index in xrange(1, nTestExamples + 1):

            # Get a random input example
            example = self.dataHandler.getNextTestExample()[0]
            currentLabel = self.params['labels'][np.argmax(example['label'])]
            labelIndex = np.where(example['label'] == 1)[0][0]

            # Take an action
            actionVector = self.networkTf.getActionVector(example['data'])
            actionIndex = np.argmax(actionVector)
            actionLabel = self.params['labels'][actionIndex]

            # add instance to confusion matrix
            confMatrix[labelIndex, actionIndex] += 1
            self.logger.info('Test example {} evaluated'.format(index))

        
        return confMatrix


class expMlWna(basicExperiment):

    def __init__(self, jsonFile, overwriteOutput=False):
        """
            Load the jsonFile and check if it contains all the necessary parameters

        """

        necessaryParams = ['layers',
                           'learningRate',
                           'labels',
                           'gammaReward',
                           'Niter',
                           'dataSet',
                           'trueReward',
                           'falseReward',
                           'logLevel',
                           'reportFreq',
                           'randomSeed',
                           'noiseSigma',
                           'learningRateH',
                           'uLow',
                           'uHigh']

        # Load the parameters from the json file
        with open(jsonFile, 'r') as f:
            self.params = json.load(f)

        # check if all the parameters are in the dictionary
        for parameter in necessaryParams:
            if not parameter in self.params:
                raise RuntimeError(
                    'The parameter {} is missing from the parameter file!'.format(parameter))

        # Start the logger
        self.initLogging()

        # Make a folder for the Output file
        # If the Output file exists then stop the simulation
        if not os.path.exists('Output'):
            os.makedirs('Output')
        elif not overwriteOutput:
            raise RuntimeError(
                'Idiot check! An <<Output>> folder exists. Delete it to proceed!')

        # Set the random seed for numpy and tensorflow
        # for the sake of simplicity we use the same seed
        np.random.seed(self.params['randomSeed'])
        tf.set_random_seed(self.params['randomSeed'])

        # checkpointing is turned off by default
        self.checkpointing = False

    def initializeExperiment(self):

        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetworkWta(self.params['layers'],
                                                self.actFunc.value)
        # tf.nn.relu)
        self.networkTf.setNoiseSigma(self.params['noiseSigma'])
        self.networkTf.setHomeostaticParams(self.params['learningRateH'],
                                            self.params['uLow'],
                                            self.params['uHigh'])
        self.networkTf.initialize()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])


        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # Set up arrays and parameters to save the progress
        self.meanR = 0
        self.meanRArray = [0]
        self.meanRArrayClass = {}
        for label in self.params['labels']:
            self.meanRArrayClass[label] = [0]
        self.currentRArray = []

    def singleIteration(self):
        """
            Perform the single iteration:
                --- take an action
                --- obtain a reward
                --- save the reward into the mean reward and the rewards
                    by class
                --- update the parameters
        """

        # Get a random input example
        example = self.dataHandler.getRandomTrainExample()[0]
        currentLabel = self.params['labels'][np.argmax(example['label'])]

        # Make an iteration
        currentReward = self.networkTf.doOneIteration(
                                    example['data'],
                                    example['label'],
                                    self.params['learningRate'],
                                    self.meanRArray[-1])
        self.logger.info('The true label is {}'.format(example['label']))

        # Save the observed rewards into the respective arrays
        self.currentRArray.append(currentReward)
        newMeanR = self.params['gammaReward'] * self.meanRArray[-1] + \
            (1. - self.params['gammaReward']) * currentReward
        self.meanRArray.append(newMeanR)

        for label in self.params['labels']:
            if currentLabel == label:
                # If the label is the same as we have just tested then
                # update the array with the new reward
                newMeanRClass = self.params['gammaReward'] * \
                    self.meanRArrayClass[label][-1] + \
                    (1. - self.params['gammaReward']) * currentReward
                self.meanRArrayClass[label].append(newMeanRClass)
            else:
                # For any other label the mean weight stays
                self.meanRArrayClass[label].append(
                    self.meanRArrayClass[label][-1])

    def continueFromCheckpoint(self):

        # Set up the experiment again
        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetworkWta(self.params['layers'],
                                                self.actFunc.value)
        self.networkTf.setNoiseSigma(self.params['noiseSigma'])
        self.networkTf.setHomeostaticParams(self.params['learningRateH'],
                                            self.params['uLow'],
                                            self.params['uHigh'])
        self.networkTf.getInitialWeights(self.currentWs)
        self.networkTf._createComputationalGraph()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # continue experiment
        st = self.currIter + 1
        self.runFullExperiment(startFrom=st)


    def runTesting(self, testSetFile):

        # Set up the experiment again
        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetworkWta(self.params['layers'],
                                                self.actFunc.value)
        self.networkTf.setNoiseSigma(self.params['noiseSigma'])
        self.networkTf.setHomeostaticParams(self.params['learningRateH'],
                                            self.params['uLow'],
                                            self.params['uHigh'])
        self.networkTf.getInitialWeights(self.currentWs)
        self.networkTf._createComputationalGraph()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            testSetFile,
            self.params['dataSet'])

        self.dataHandler.loadTestSet()

        # Set up variables and arrays
        nLabels =  len(self.params['labels'])
        confMatrix = np.zeros((nLabels, nLabels))
        nTestExamples = self.dataHandler.nTest

        for index in xrange(1, nTestExamples + 1):

            # Get a random input example
            example = self.dataHandler.getNextTestExample()[0]
            currentLabel = self.params['labels'][np.argmax(example['label'])]
            labelIndex = np.where(example['label'] == 1)[0][0]

            # Take an action
            actionVector = self.networkTf.getActionVector(example['data'])
            actionIndex = np.argmax(actionVector)
            actionLabel = self.params['labels'][actionIndex]

            # add instance to confusion matrix
            confMatrix[labelIndex, actionIndex] += 1
            self.logger.info('Test example {} evaluated'.format(index))

        
        return confMatrix


class expMlVarifyBp(basicExperiment):
    """

        Experiment identical to the basic Experiment but the last layer of weights is not updated

    """


    def initializeExperiment(self):

        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetworkVerifyBp(
                                        self.params['layers'],
                                        self.actFunc.value)
        # tf.nn.relu)
        self.networkTf.initialize()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # Set up arrays and parameters to save the progress
        self.meanR = 0
        self.meanRArray = [0]
        self.meanRArrayClass = {}
        for label in self.params['labels']:
            self.meanRArrayClass[label] = [0]
        self.currentRArray = []


class expMlWnaVerifyBp(expMlWna):
    """
        This experiment inherits everything from the one before but it does not update the weight to the output neurons
    """

    def initializeExperiment(self):

        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetworkWtaVerifyBp(
                                        self.params['layers'],
                                        self.actFunc.value)
        # tf.nn.relu)
        self.networkTf.setNoiseSigma(self.params['noiseSigma'])
        self.networkTf.setHomeostaticParams(self.params['learningRateH'],
                                            self.params['uLow'],
                                            self.params['uHigh'])
        self.networkTf.initialize()

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
            self.params['labels'],
            self.params['dataSet'],
            self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardSchemes.maxClassification(
            self.params['trueReward'],
            self.params['falseReward'])

        # Set up arrays and parameters to save the progress
        self.meanR = 0
        self.meanRArray = [0]
        self.meanRArrayClass = {}
        for label in self.params['labels']:
            self.meanRArrayClass[label] = [0]
        self.currentRArray = []