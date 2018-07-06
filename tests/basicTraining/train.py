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
learningRate = .1
nudging = .1
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
W = lagrangeRL.tools.networks.feedForwardWtaReadout(layers, .2)

# Lagrange network
simClass = lagrangeRL.network.lagrangeElig()
simClass.setLearningRate(learningRate)
simClass.setTimeStep(timeStep)
simClass.setTau(tau)
simClass.setNudging(nudging)
simClass.addMatrix(W)
simClass.setTauEligibility(tauElig)
simClass.saveTraces(False)
wMaxFixed = np.zeros((N,N))
wMaxFixed[-layers[-1]:, -layers[-1]:] = 1
simClass.setFixedSynapseMask(wMaxFixed.astype(bool))

# Set up the input class
value = np.ones(N)
mask = np.zeros(N)
mask[:layers[0]] = 1
Input = lagrangeRL.tools.inputModels.constantInput(
        							value,
        							mask)
simClass.connectInput(Input)

# Set up the nudging class
value = np.ones(N)
mask = np.zeros(N)
mask[N-layers[-1]:] = 1
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
	Input.value[:layers[0]] = example['data']

	# get and set random noise
	noiseVector = 2*(np.random.randint(2, size=layers[-1])-0.5)
	nudgingNoise.value[N-layers[-1]:] = noiseVector

	# run the simulation
	simClass.setInitialConditions(np.zeros(N))
	simClass.initSimulation()
	simClass.run(simTime)
	output = simClass.u[N-layers[-1]:]

	# obtain reward
	R = rewardScheme.obtainReward(example['label'], output)
	avgR = avgR + gammaReward * (R - avgR)
	loss = np.linalg.norm(example['label'] - output)

	# Update the weights
	deltaW += simClass.calculateWeightUpdates(learningRate, R-avgR)

	if index % layers[-1] == 0:
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

# report after the training
print('Weights after training: \n {}'.format(simClass.W.data))
