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


class slimExperimentRegVerifyBp(timeContinuousClassificationSmOu):

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
        self.tRamp = params['tRamp']
        self.noiseStd = params['noiseStd']
        self.noiseAutoCorrTime = params['noiseAutoCorrTime']
        # tuple of capping the weights, None for no clipping
        self.cap = params['cap']
        self.lowerValley = params['lowerValley']
        self.upperValley = params['upperValley']
        self.kappaDecay = params['kappaDecay']
        # regularization to keep the WTA in the range
        self.uLow = params['uLow']
        self.uHigh = params['uHigh']
        self.uRegAlpha = params['uRegAlpha']
        self.params = params

    def initLogging(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        if 'logLevel' in self.params:
            coloredlogs.install(level=self.params['logLevel'])

    def setUpSavingArrays(self):

        # Set up arrays to save results from the simulation

        # Average rewards
        self.avgR = {}
        self.avgRArrays = {}
        self.instantRArray = []
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
        self.simClass = lagrangeRL.network.lagrangeEligTfApproxReg()
        self.simClass.setLearningRate(1.)
        self.simClass.setTimeStep(self.timeStep)
        self.simClass.setTau(self.tau)
        self.simClass.setRegTerm(self.uLow, self.uHigh)
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
        eligs = self.simClass.getRegTerm()
        #eligs = self.simClass.getEligibilities().T
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


    def runSimulation(self):

        # Set the first new input
        self.newInput()
        self.nextInputTime = self.simTime
        counter = 0. # counter of the presented images
        self.rewards = {}
        self.nextReadoutTime = self.simTime - self.tRamp
        self.globalTime = 0.

        while counter < self.Niter:
            nextEvent = self.obtainNextEvent()
            deltaT = nextEvent[1] - self.globalTime
            self.self.simClass.run(deltaT,
                                   updateNudging=True)
            self.globalTime += deltaT
            if nextEvent[0] == 'nextInput':
                self.newInput()
                self.simClass.deleteTraces()
                self.newInputTime += self.simTime
            elif nextEvent[0] == 'nextReadout':
                self.doReadout()
                self.nextReadoutTime += self.simTime
            elif nextEvent[0] == 'reward':
                self.applyReward(nextEvent[1])
            else:
                sys.exit('A huge problem occured. nextEvent[0] has to be <nextInput>, <nextReadout> or <reward>. It is {} instead!'.format(nextEvent[0]))



        self.plotFinalReport()

        self.saveResults()




        ### OLD
        for index in range(1, self.Niter + 1):
            self.singleIteration(index)
            if index % 10 == 0:
                self.plotFinalReport()

        self.plotFinalReport()

        self.saveResults()

    def applyReward(self, rewardTime):
        """
            Apply the reward and change the weights accordingly
        """

        R = self.rewards[rewardTime][0]
        trueLabel = self.rewards[rewardTime][0]

        # Update the weights
        modavgR = np.min([np.max([self.avgR[trueLabel], -0.9]), 1.])
        self.logger.debug('The avgR for the label {0} is {1}'.format(
            trueLabel, self.avgR[trueLabel]))
        self.logger.debug('The modavgR for the label {0} is {1}'.format(
            trueLabel, modavgR))
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR,
                                                           self.uRegAlpha)
        self.deltaW += -1. * self.weightDecayRate * \
            (copy.deepcopy(self.simClass.W.data) - self.weightDecayTarget)
        self.simClass.applyWeightUpdates(self.deltaW, self.cap)
        self.simClass.calcWnoWta(self.layers[-1])
        self.logger.debug(
            'The applied weigth changes: {}'.format(self.deltaWBatch))

        # delete the applied reward from the reward queue
        del self.rewards[rewardTime]

    def doReadout(self):
        """
            Obtain the readout. Append the reward to the reward saving arrays. Save the reward Delay and reward and append to the reward queue
        """

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
        self.instantRArray.append({trueLabel: R})
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])

        # get a reward delay
        rDelay = np.random.rayleigh(self.simTime * np.sqrt(2./np.pi))
        self.rewards[self.globalTime + rDelay] = [R, trueLabel]

        self.Warray.append(self.simClass.W.data[~self.simClass.W.mask])
        self.wToOutputArray.append(self.simClass.W.data[self.layers[-2]:,:self.layers[-2]].flatten())

        self.plotReport(index, output, example)
        self.logger.info("Iteration {} is done.".format(index))
        self.logger.debug(
            "The current weights are: {}".format(self.simClass.W))



    def obtainNextEvent(self):
        """
            Get the time and type of the next event
        """

        if bool(self.rewards):
            # If the reward queue is not empty then we also check if the reward is first
            d = {'reward': min(self.rewards.values()),
                 'nextInput': self.newInputTime,
                 'nextReadout': self.nextReadoutTime}
            eventType = min(d)
            eventTime = d[eventType]
            return [eventType, eventTime]
        else:
            # If the reward queue is empty then we dont check the rewards
            d = {'nextInput': self.newInputTime,
                 'nextReadout': self.nextReadoutTime}
            eventType = min(d)
            eventTime = d[eventType]
            return [eventType, eventTime]

    def newInput(self):

        # get and set example
        self.example = self.myData.getRandomTestExample()[0]
        self.Input.value[:self.layers[0]] = self.example['data']

    def saveResults(self):

        dictToSave = {'weights': np.array(self.wToOutputArray).tolist(),
        			  'P': {'mean': self.avgRArray,
                            'instantRewards': self.instantRArray}}

        for label in self.labels:
        	dictToSave['P'][label] = self.avgRArrays[label]

        # Save to the result to output
        with open('Output/results.json', 'w') as outfile:
    		json.dump(dictToSave, outfile)

    
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
        self.instantRArray.append({trueLabel: R})
        for key in self.avgRArrays:
            self.avgRArrays[key].append(self.avgR[key])
            

        # Update the weights
        modavgR = np.min([np.max([self.avgR[trueLabel], -0.9]), 1.])
        self.logger.debug('The avgR for the label {0} is {1}'.format(
            trueLabel, self.avgR[trueLabel]))
        self.logger.debug('The modavgR for the label {0} is {1}'.format(
            trueLabel, modavgR))
        self.deltaW = self.simClass.calculateWeightUpdates(self.learningRate,
                                                           R - modavgR,
                                                           self.uRegAlpha)
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
        self.wToOutputArray.append(self.simClass.W.data[self.layers[-2]:,:self.layers[-2]].flatten())

        self.plotReport(index, output, example)

        self.simClass.run(self.tRamp, updateNudging=True)

        self.simClass.deleteTraces()

        self.logger.info("Iteration {} is done.".format(index))
        self.logger.debug(
            "The current weights are: {}".format(self.simClass.W))
