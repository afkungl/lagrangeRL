import lagrangeRL
import numpy as np
import copy
import time
import sys
import os
import logging
import coloredlogs

# Logger
logger = logging.getLogger('test WTA')
coloredlogs.install(level='DEBUG')

# meta parameters
Nwta = 4
layers = [Nwta, Nwta]
tau = 10.
tauElig = 10.
sigmaLog = 1.
nudging = 0.
simTime = 1000.
wtaExc = 0.9#.99
wtaInh = .3#0.3
inputU = [-.5, -.3, -.7, -.1]
timeStep = 0.1
figName = 'reportWtaTest.png'
Tramp = 10.


# Set up network
W = lagrangeRL.tools.networks.feedForwardWtaReadout(layers,
													wtaExc,
													0.,
													0.,
													wtaInh)
np.fill_diagonal(W[Nwta:, :Nwta], 1.)
simClass = lagrangeRL.network.lagrangeEligTf()
simClass.setLearningRate(1.)
simClass.setTimeStep(timeStep)
simClass.setTau(tau)
simClass.setNudging(nudging)
simClass.addMatrix(W)
simClass.setTauEligibility(tauElig)
simClass.saveTraces(True)

# Attach input
maskI = np.zeros(2*Nwta)
maskI[:Nwta] = 1
valueI = np.zeros(2*Nwta)
valueI[:Nwta] = inputU
logger.info('The input mask is: {}'.format(maskI))
logger.info('The input value is: {}'.format(valueI))
Input = lagrangeRL.tools.inputModels.smoothedConstantInput(
            valueI,
            maskI,
            simTime,
            Tramp)
simClass.connectInput(Input)

# Attach nudging (effectively turned off)
maskN = np.zeros(2*Nwta)
maskN[Nwta:] = 1
valueN = np.zeros(2*Nwta)
valueN[:Nwta] = inputU
nudgingNoise = lagrangeRL.tools.targetModels.constantTarget(
    valueN,
    maskN)
simClass.connectTarget(nudgingNoise)

# activation function
actFunc = lagrangeRL.tools.activationFunctions.sigmoidTf(
            				sigmaLog)
simClass.connectActivationFunction(actFunc)

# Do the simulation
simClass.calcWnoWta(Nwta)
simClass.initCompGraph()
logger.debug('The weight matrix of the network is {}'.format(simClass.W))
logger.debug('The weight matrix of the network without WTA is {}'.format(simClass.WnoWta))
simClass.run(simTime - Tramp)

# get the results
traces = simClass.getTraces()
membpot = simClass.getMembPotentials()
rhos = simClass.getActivities()
eligs = simClass.getEligibilities()

# Plot the results
lagrangeRL.tools.visualization.plotReportWtaTest(traces,
												 timeStep,
												 rhos[:Nwta],
												 rhos[Nwta:],
												 membpot[Nwta:],
												 eligs[Nwta:,:Nwta],
												 #np.sign(eligs),
												 figName)