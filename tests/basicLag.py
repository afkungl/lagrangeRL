import lagrangeRL
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt

# Metaparameters
layers = [3, 4]  # type: List[int]
N = np.sum(layers)
tau = 10
tauElig = 1.
sigmaLog = 1.
learningRate = 1.
nudging = .1
simTime = 200
timeStep = .1

# Set up the network
W = lagrangeRL.tools.networks.feedForwardWtaReadout(layers, .5)
W.data[2,0] = 0.5
W.data[2,1] = 0.5


# Set up the simulation class
simClass = lagrangeRL.network.lagrangeElig()
simClass.setLearningRate(learningRate)
simClass.setTimeStep(timeStep)
simClass.setTau(tau)
simClass.setNudging(nudging)
simClass.addMatrix(W)
simClass.setTauEligibility(tauElig)

# connect to input and output
value = np.ones(N)
mask = np.zeros(N)
mask[:3] = 1
constInput = lagrangeRL.tools.inputModels.constantInput(
        							value,
        							mask)
simClass.connectInput(constInput)
value = np.ones(N)
mask = np.zeros(N)
mask[N-1] = 1
target = lagrangeRL.tools.targetModels.constantTarget(
        							value,
        							mask)
simClass.connectTarget(target)
actFunc = lagrangeRL.tools.activationFunctions.sigmoid(sigmaLog)
simClass.connectActivationFunction(actFunc)

# Initialize the simulation
simClass.setInitialConditions(np.zeros(N))
simClass.initSimulation()

# save the traces
simClass.saveTraces(True)

# Run the simulation for 100 units
simClass.run(simTime)
print("Simulation run successfully")
print("Output in the last layer: {}".format(simClass.u[-4:]))

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
