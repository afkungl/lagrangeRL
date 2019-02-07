#!/usr/bin/env python

from mlModel.network import mlNetwork
import json
import logging
import coloredlogs
from mlModel.aux import tfAux
from lagrangeRL import tools

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
                           'logLevel']

        # Load the parameters from the json file
        with open(jsonFile, 'r') as f:
            self.params = json.load(f)

        # check if all the parameters are in the dictionary
        for parameter in necessaryParams:
            if not parameter in self.params:
                raise RuntimeError('The parameter {} is missing from the parameter file!'.format(parameter))

        # Start the logger
        self.initLogging()

    def initializeExperiment(self):

        # Set up the network
        self.testNetwork = mlNetwork.mlNetwork(self.params['layers'],
                                               tfAux.leaky_relu)

        # Set up the data handler
        self.dataHandler = tools.dataHandler.dataHandlerMnist(
                                        self.params['labels'],
                                        self.params['dataSet'],
                                        self.params['dataSet'])

        self.dataHandler.loadTrainSet()

        # Set up the reward scheme
        self.rewardScheme = tools.rewardScheme.maxClassification(
                                        self.params['trueReward'],
                                        self.params['falseReward'])



