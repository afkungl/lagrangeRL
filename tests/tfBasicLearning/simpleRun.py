import lagrangeRL
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import sys
import os


########################
# Set the metaparamters
layers = [4, 2]
N = np.sum(layers)
tau = 10.
tauElig = 1.
sigmaLog = 1.
learningRate = 100.
learningRateOuter = 1.
nudging = .0 
simTime = 100.
timeStep = .1
labels = [0,1]
gammaReward = .1
Niter = 20 #training iterations
dataSet = 'dataset.txt'
trueReward = 1.
falseReward = -1.

#######################
# Set up the components of the system

# Set up the network structure
W = lagrangeRL.tools.networks.feedForwardWtaReadout(layers, .4, offset=.5,
                                                    noiseMagnitude=.6)
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
mask[:layers[0]] = 1.
Input = lagrangeRL.tools.inputModels.constantInput(
                                    value,
                                    mask)
simClass.connectInput(Input)

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

# Set up the nudging class
value = np.ones(N)
mask = np.zeros(N)
mask[N-layers[-1]:] = 1
nudgingNoise = lagrangeRL.tools.targetModels.constantTarget(
                                    value,
                                    mask)
simClass.connectTarget(nudgingNoise)

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
simClass.calcWnoWta(layers[-1])

# Set up arrays to save results from the simulation
avgR = 0
avgRArray = []
deltaW = copy.deepcopy(W.data)
deltaW = 0. * deltaW
deltaWBatch = 0. * copy.deepcopy(deltaW)
Warray = []
if os.path.exists('Output'):
    sys.exit('There is a folder named Output. Delete it to run the simularion! I will not do it for you!')
else:
    os.makedirs('Output')

# Do the training
for index in range(1, Niter+1):

    # get and set example
    example = myData.getNextTestExample()[0]
    Input.value[:layers[0]] = example['data']

    # get and set random noise
    noiseVector = 2*(np.random.randint(2, size=layers[-1])-0.5)
    nudgingNoise.value[N-layers[-1]:] = noiseVector

    # run the simulation
    simClass.resetCompGraph()
    simClass.run(simTime)
    output = simClass.getMembPotentials()[N-layers[-1]:]

    # obtain reward
    R = rewardScheme.obtainReward(example['label'], output)
    avgR = avgR + gammaReward * (R - avgR)
    avgRArray.append(avgR)
    loss = np.linalg.norm(example['label'] - output)

    # Update the weights
    deltaW = simClass.calculateWeightUpdates(learningRateOuter, R-avgR)
    deltaWBatch += deltaW

    if index % layers[-1] == 0:
        simClass.applyWeightUpdates(deltaWBatch)
        simClass.calcWnoWta(layers[-1])
        print('The applied weigth changes:')
        print(deltaWBatch)
        deltaWBatch = 0.*deltaWBatch

    Warray.append(simClass.W.data[~simClass.W.mask])

    # print some feedback
    print('Iteration {} is ready!'.format(index))
    print("Current reward: {} \naveraged reward:  {} \ncurrent output: {} \nexpected output: {}".format(R, avgR, output, example['label']))

    # Plot the report
    figName = 'Output/reportIteration{}.png'.format(index)
    # timeStep is given above
    traces = simClass.getTraces()
    outputU = output
    outputRho = simClass.getActivities()[N-layers[-1]:]
    target = np.argmax(example['label']) + 1
    data = example['data']
    figSize = (2,2)
    print(simClass.W.data)
    wCurrent = simClass.W.data[layers[0]:,:N-layers[1]].T
    eligs = simClass.getEligibilities()[layers[0]:,:N-layers[1]].T
    signDeltaW = np.sign(deltaW[layers[0]:,:N-layers[1]].T)
    lagrangeRL.tools.visualization.plotReport(
               figName,
               timeStep,
               traces,
               outputU,
               outputRho,
               target,
               data,
               figSize,
               wCurrent,
               eligs,
               signDeltaW)

Warray = np.array(Warray)
fig, ax  = plt.subplots(1)
timeArray = np.arange(len(Warray[:,0])) * timeStep

# Plot the report
lagrangeRL.tools.visualization.plotLearningReport(Warray,
                                                  avgRArray,
                                                  'learningReport.png')

# report after the training
print('Weights after training: \n {}'.format(simClass.W.data))

# run the simulation
start = time.time()
simClass.run(100.)
stop = time.time()
print("The elapsed time for the run command is: {0:.2f} s".format(stop-start))

