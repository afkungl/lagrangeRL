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


class expExactLagrangeScrambled(expExactLagrange):
    """
        Experiment class for the optimized network.
        The reward arrives delayed by one trial
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

        # get the delay parameter
        self.meanDelay = params['meanDelay']

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

        # save only reward as a default parameter
        if 'fixedPatternNoiseSigma' in params:
            self.fixedPatternNoiseSigma = params['fixedPatternNoiseSigma']
        else:
            self.fixedPatternNoiseSigma = 0.0

        # Set the random seed for numpy and tensorflow
        np.random.seed(params['randomSeed'])
        tf.set_random_seed(params['randomSeed'])

        # checkpointing is turned off by default
        self.checkpointing = False

        self.params = params

    def runSimulation(self, startFrom=1):

        # fill up the event array with change input events
        self.initEvents()
        self.createInputEvent(0.0)
        counter = startFrom
        counterReward = 0

        while self.events:

            self.logger.debug('The current events are: {}'.format(self.events))
            # pop next event
            idNext = self.getNextEvent()
            nextEvent = self.events.pop(idNext)
            timeStampNext = nextEvent[0]
            self.logger.debug('The next event with id {0} is: {1}'.format(idNext, nextEvent))

            # propagate the network to the next event
            tGlobal = self.simClass.T
            if timeStampNext > tGlobal:
                deltaTime = np.around(timeStampNext - tGlobal, decimals=1)
                self.logger.debug('The delta time to be propagated is {}'.format(deltaTime))
                self.simClass.run(deltaTime)
            self.logger.debug('The time after propagation is {}'.format(self.simClass.T))

            # apply event
            eventType = nextEvent[1]
            if eventType == 'changeInput':
                self.eventChangeInput(self.simClass.T,
                                      counter < self.Niter)
                counter += 1
                self.logger.debug('I applied changeInput')
                if self.checkpointing and (counter % self.checkPerIter == 0):
                    self.saveCheckpoint(counter)
                    self.logger.info('Checkpoint created at {}'.format(counter))
            elif eventType == 'readOut':
                self.eventReadOut(self.simClass.T,
                                  nextEvent[2])
                self.logger.debug('I applied readOut')
            elif eventType == 'reward':
                self.eventReward(nextEvent[2],
                                 nextEvent[3])
                # make report if applicable
                if counter % self.reportFrequency == 0:
                    self.plotFinalReport()
                    self.saveResults()
                self.logger.debug('I applied reward')
                counterReward += 1
                self.logger.info('After {0} applied reward the mean reward is {1}'.format(counterReward, self.meanReward))
            else:
                self.logger.error(
                    'The received event type is not in [<changeInput>, <readOut>, <reward>]. Received: {}'.format(eventType))

            self.logger.debug('The events after one iteration in the while loop are: {}'.format(self.events))


        self.logger.info('The simulation finished after presenting {} inputs'.format(counter))

    def eventReward(self, reward, trueLabel):

        # apply the parameter updates
        modulatedAvgReward = np.max([self.meanReward, 0.0])
        if self.saveOnlyReward:
            self.simClass.applyWeightUpdates(
                    reward - modulatedAvgReward)
        else:
           self.Wnew = self.simClass.applyWeightUpdates(
                    reward - modulatedAvgReward)

        # update the mean reward
        self.meanReward = self.meanReward + \
            self.gammaReward * (reward - self.meanReward)

        # Update the reward for the single classes
        trueLabelIndex = self.labels[np.argmax(trueLabel)]
        self.avgRewards[trueLabelIndex] = self.avgRewards[trueLabelIndex] + \
            self.gammaReward * (reward - self.avgRewards[trueLabelIndex])

        # save the averaged reward arrays
        self.avgRArray.append(self.meanReward)
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgRewards[key])

        # save the weights in an array if applicable
        if not self.saveOnlyReward:
            self.Warray.append(self.Wnew[~self.simClass.W.mask])
            self.wToOutputArray.append(
                self.simClass.W.data[self.layers[-2]:, :self.layers[-2]].flatten())


    def eventReadOut(self, time, trueLabel):

        # get network answer
        output = self.simClass.getLowPassActivity()[self.N - self.layers[-1]:]

        # obtain reward
        Reward = self.rewardScheme.obtainReward(trueLabel, output)

        # create reward event
        rewardDelay = np.random.gamma(2.0, self.meanDelay/2.0)
        rewardTime = np.around(time + rewardDelay, decimals=1)
        self.createRewardEvent(rewardTime,
                               Reward,
                               trueLabel)

    def createRewardEvent(self, time, reward, label):

        time = np.around(time, decimals=1)
        event = (time, 'reward', reward, label)
        self.events[self.eventIdCounter] = event
        self.eventIdCounter += 1

    def eventChangeInput(self, time, createNewInput):

        # get an example as input
        inputExample = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = inputExample['data']

        # create a readout event
        readOutTime = np.around(time + self.simTime - self.tRamp,
                                decimals=1)
        self.createReadOutEvent(time + self.simTime - self.tRamp,
                                inputExample['label'])

        # if applicable create a new Input
        if createNewInput:
            inputTime = np.around(time + self.simTime, decimals=1)
            self.createInputEvent(time + self.simTime)

    def createReadOutEvent(self, time, trueLabel):

        time = np.around(time, decimals=1)
        event = (time, 'readOut', trueLabel)
        self.events[self.eventIdCounter] = event
        self.eventIdCounter += 1

    def getNextEvent(self):

        # get a starter value
        ID_0 = self.events.keys()[0]
        timeStamp_0 = self.events[ID_0][1]

        for eventId in self.events:
            timeStamp = self.events[eventId][0]
            if timeStamp < timeStamp_0:
                timeStamp_0 = timeStamp
                ID_0 = eventId

        return ID_0


    def createInputEvent(self, time):

        time = np.around(time, decimals=1)
        event = (time, 'changeInput')
        self.events[self.eventIdCounter] = event
        self.eventIdCounter += 1

    def initEvents(self):

        self.events = {}
        self.eventIdCounter = 1