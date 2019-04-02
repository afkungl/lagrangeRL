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

    def __init__(self, jsonFile):
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
        else:
            raise RuntimeError(
                'Idiot check! An <<Output>> folder exists. Delete it to proceed!')

        # Set the random seed for numpy and tensorflow
        # for the sake of simplicity we use the same seed
        np.random.seed(self.params['randomSeed'])
        tf.random.set_random_seed(self.params['randomSeed'])

    def initializeExperiment(self):

        # Set up the network
        self.actFunc = activationFunctions.softReluTf(1., 0., 0.1)
        self.networkTf = mlNetwork.mlNetwork(self.params['layers'],
                                             self.actFunc.value)
                                             #tf.nn.relu)
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
            if actionLabel == label:
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

    def runFullExperiment(self):
        """
            Run the full experiment for the defined number of iterations
        """

        # loop through the experiment
        for index in xrange(self.params['Niter']):
            self.singleIteration()
            self.logger.info('Iteration number {} finished.'.format(index + 1))
            if index % self.params['reportFreq'] == 0:
                visualization.plotMeanReward(self.meanRArray,
                                             'Output/meanReward.png',
                                             self.meanRArrayClass)
                self.logger.info('Mean reward plotted.')

        # plot the results
        visualization.plotMeanReward(self.meanRArray,
                                     'Output/meanReward.png',
                                     self.meanRArrayClass)
