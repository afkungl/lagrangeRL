import lagrangeRL
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

########################
# Set the metaparamters
layers = [4, 4]
N = np.sum(layers)
tau = 10.
tauElig = 1.
sigmaLog = 1.
learningRate = .1
nudging = .3 
simTime = 75.
timeStep = .1
labels = [0,1]
gammaReward = .2
Niter = 1000 #training iterations
dataSet = 'dataset.txt'
trueReward = 1.
falseReward = -1.

#######################
# Set up the components of the system

# Set up the network structure
W = lagrangeRL.tools.networks.feedForwardWtaReadout(layers, .4)
print(W)

# Lagrange network
simClass = lagrangeRL.network.lagrangeEligTf()
simClass.setLearningRate(learningRate)
simClass.setTimeStep(timeStep)
simClass.setTau(tau)
simClass.setNudging(nudging)
simClass.addMatrix(W)
simClass.setTauEligibility(tauElig)
simClass.saveTraces(True)
wMaxFixed = np.zeros((N,N))
wMaxFixed[-layers[-1]:, -layers[-1]:] = 1
simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

# connect to input
value = np.ones(N)
mask = np.zeros(N)
mask[:4] = 1.
constInput = lagrangeRL.tools.inputModels.constantInput(
        							value,
        							mask)
simClass.connectInput(constInput)

# Connect to target
value = 1. * np.ones(N)
mask = np.zeros(N)
mask[N-1] = 1.
target = lagrangeRL.tools.targetModels.constantTarget(
        							value,
        							mask)
simClass.connectTarget(target)

# Connect to activation function
actFunc = lagrangeRL.tools.activationFunctions.sigmoidTf(sigmaLog)
simClass.connectActivationFunction(actFunc)

# Initialize the stuff
simClass.initCompGraph()


# run the simulation
start = time.time()
simClass.run(100.)
stop = time.time()
print("The elapsed time for the run command is: {0:.2f} s".format(stop-start))

# make figures
traces = simClass.getTraces()
timeArray = np.arange(len(traces['uMem'][:,0])) * timeStep
fig, ax  = plt.subplots(3)

# Plot the membrane potentials
uMems = traces['uMem']
for index in range(len(uMems[0,:])):
	ax[0].plot(timeArray, uMems[:, index])
ax[0].set_xlabel('time')
ax[0].set_ylabel('membrane potentials')

eligs = traces['eligibilities']
for index in range(len(eligs[0,:])):
	ax[1].plot(timeArray, eligs[:, index])
ax[1].set_xlabel('time')
ax[1].set_ylabel('eligibilities')


ax[2].plot(timeArray, uMems[:, 0])
ax[2].plot(timeArray, uMems[:, N-1])
ax[2].set_xlabel('time')
ax[2].set_ylabel('membrane potentials')

fig.savefig('traces.png')