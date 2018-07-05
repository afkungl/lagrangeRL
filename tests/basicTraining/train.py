import lagrangeRL
import numpy as np
import copy
import matplotlib.pyplot as plt

########################
# Set the metaparamters
layers = [4, 2]
N = np.sum(layers)
tau = 10.
tauElig = 5.
sigmaLog = 1.
learningRate = .3
nudging = .0
simTime = 120.
timeStep = .1
labels = [0,1]
gammaReward = .2
Niter = 10 #training iterations
dataSet = 'dataset.txt'
trueReward = 1.
falseReward = -1.

#######################
# Set up the components of the system

# Set up the network structure
W = 1. * lagrangeRL.tools.networks.feedForwardWtaReadout(layers, 1.)

# Lagrange network
simClass = lagrangeRL.network.lagrangeElig()
simClass.setLearningRate(learningRate)
simClass.setTimeStep(timeStep)
simClass.setTau(tau)
simClass.setNudging(nudging)
simClass.addMatrix(W)
simClass.setTauEligibility(tauElig)
simClass.saveTraces(False)

# Set up the input class
value = np.ones(N)
mask = np.zeros(N)
mask[:4] = 1
Input = lagrangeRL.tools.inputModels.constantInput(
        							value,
        							mask)
simClass.connectInput(Input)

# Set up the nudging class
value = np.ones(N)
mask = np.zeros(N)
mask[N-2:] = 1
nudgingNoise = lagrangeRL.tools.targetModels.constantTarget(
        							value,
        							mask)
simClass.connectTarget(nudgingNoise)

# Set up the activation function
actFunc = lagrangeRL.tools.activationFunctions.sigmoid(sigmaLog)
simClass.connectActivationFunction(actFunc)

# set up the dataHandler
myData = lagrangeRL.tools.dataHandler.dataHandlerMnist(labels,
													   dataSet,
													   dataSet)
myData.loadTestSet()
myData.loadTrainSet()

# set up the reward scheme
rewardScheme = lagrangeRL.tools.rewardSchemes.maxClassification(
											trueReward,
                                            falseReward)

# Initialize variables
avgR = 0
deltaW = copy.deepcopy(W.data)
deltaW = 0. * deltaW
Warray = []

# Do the training
for index in range(1, Niter+1):

	# get and set example
	example = myData.getNextTestExample()[0]
	Input.value[:4] = example['data']

	# get and set random noise
	noiseVector = 2*(np.random.randint(2, size=2)-0.5)
	nudgingNoise.value[N-2:] = noiseVector

	# run the simulation
	simClass.setInitialConditions(np.zeros(N))
	simClass.initSimulation()
	simClass.run(simTime)
	output = simClass.u[N-2:]

	# obtain reward
	R = rewardScheme.obtainReward(example['label'], output)
	avgR = avgR + gammaReward * (R - avgR)
	loss = np.linalg.norm(example['label'] - output)

	# Update the weights
	deltaW += simClass.calculateWeightUpdates(learningRate, R-avgR)

	if True:
		simClass.applyWeightUpdates(deltaW)
		deltaW = 0.*deltaW
	#simClass.updateWeights(learningRate, R-avgR)

	Warray.append(simClass.W.data[~simClass.W.mask])

	# print some feedback
	print('Iteration {} is ready!'.format(index))
	print("Current reward: {} \naveraged reward:  {} \ncurrent output: {} \nexpected output: {}".format(R, avgR, output, example['label']))

Warray = np.array(Warray)
fig, ax  = plt.subplots(1)
timeArray = np.arange(len(Warray[:,0])) * timeStep

# Plot the membrane potentials
for index in range(len(Warray[0,:])):
	ax.plot(timeArray, Warray[:, index])
ax.set_xlabel('time')
ax.set_ylabel('weights')
fig.savefig('weights.png')