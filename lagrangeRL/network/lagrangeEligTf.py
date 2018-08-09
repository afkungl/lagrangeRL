from . import networkBase
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
import logging
import coloredlogs

class lagrangeEligTf(networkBase.networkBase):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.dtype = tf.float32
        self.T = 0.

        # set up a logger
        self.logger = logging.getLogger('lagrangeEligTf')

    def setTfType(self, tfType):
        """
            Set the datatype of the calculations, i.e. float, int etc.

            Keywords:
                --- tfType: datatype has to be a tensorflow datatype, e.g. tf.float32
            
        """

        self.dtyep = tfType

    def setTimeStep(self, timeStep):
        """
        Set the learning rate
        :param eta: the learning rate
        """

        self.timeStep = timeStep

    def setLearningRate(self, eta):
        """
        Set the learning rate
        :param eta: the learning rate
        """

        self.learningRate = eta

    def setTau(self, tau):
        """
        set the membrane time constant
        :param tau: postulated membrane time constant
        """

        self.tau = tau

    def setTauEligibility(self, tau):
        """
        set the time constant of the eligibility trace
        :param tau: postulated membrane time constant
        """

        self.tauEligibility = tau

    def setNudging(self, beta):
        """
        set the value of the nudging strength
        :param beta: strength of the nudging
        """

        self.beta = beta

    def initSimulation(self):
        """
        Set the initial rBar and uDiff values to 0
        """

        # Initialize
        self.rBar = 0. * self.u
        self.uDiff = 0. * self.u
        self.rDiff = 0. * self.u
        self.T = 0.
        # add eligibility trace
        self.eligibility = 0. * copy.deepcopy(self.W)

    def saveTraces(self, booleanVar=True):
        """
            Save the time trace of the voltages and of the eligibility trace
        """

        self.saveTraces = booleanVar
        self.uTraces = []
        self.eligibilityTraces = []

    def connectActivationFunction(self, actFuncObject):
        """

        :param actFunc: function with one argument and one return
        """

        self.actFunc = actFuncObject.value
        self.actFuncPrime = actFuncObject.valuePrime
        self.actFuncPrimePrime = actFuncObject.valuePrimePrime

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """
        # track the dependencies
        dependencies = []

        # Set up the variables which will be then tracked
        self.u = tf.Variable(np.zeros(self.N), dtype = self.dtype)
        self.eligibility = tf.Variable(np.zeros((self.N, self.N)), dtype = self.dtype)
        self.rho = tf.Variable(np.zeros(self.N),
                               dtype = self.dtype)
        self.rhoPrime = tf.Variable(np.zeros(self.N),
                                    dtype = self.dtype)
        self.rhoPrimePrime = tf.Variable(np.zeros(self.N),
                                         dtype = self.dtype)

        # We need an identity tensor with float values
        identity = tf.Variable(np.eye(self.N),
                               dtype=self.dtype)


        # Set up the placeholdeers which can be then modified
        self.tfW = tf.placeholder(shape = self.W.shape, dtype = self.dtype)
        self.tfWnoWta =  tf.placeholder(shape = self.W.shape, dtype = self.dtype)
        self.inputTf = tf.placeholder(dtype = self.dtype,
                                    shape = (self.N))
        self.inputPrimeTf = tf.placeholder(dtype = self.dtype,
                                         shape = (self.N))
        self.inputMaskTf = tf.placeholder(dtype = self.dtype,
                                        shape = (self.N))
        self.targetTf = tf.placeholder(dtype = self.dtype,
                                     shape = (self.N))
        self.targetPrimeTf = tf.placeholder(dtype = self.dtype,
                                          shape = (self.N))
        self.targetMaskTf = tf.placeholder(dtype = self.dtype,
                                         shape = (self.N))

        # an example input mask is needed to build the comp graph
        inputMask = self.input.getInput(0.)[2] 
        nInput = len(np.where(inputMask == 1)[0])
        nFull = len(inputMask)

        ## Start the calculations
        # Calculate the activation functions
        dependencies.append(tf.assign(self.rho, self.actFunc(self.u)))
        dependencies.append(tf.assign(self.rhoPrime, self.actFuncPrime(self.u)))
        dependencies.append(tf.assign(self.rhoPrimePrime, self.actFuncPrimePrime(self.u)))

        # set the membrane potential on the input neurons
        dependencies.append(tf.scatter_update(self.u,
        									  np.arange(nInput),
        									  tf.slice(self.inputTf,
        									  		  [0],
        									  		  [nInput]
        									  		  )
        									  )
        					)

        with tf.control_dependencies(dependencies):
            # Intermediate nodes for the vector
            y1 = tfTools.tf_mat_vec_dot(self.tfW, self.rho)
            y2 = -1. * self.u
            y3 = tfTools.tf_mat_vec_dot(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(self.tfW),self.u - tfTools.tf_mat_vec_dot(self.tfW,self.rho))), self.rho)
            y4 = self.beta*tfTools.tf_mat_vec_dot(tf.diag(self.targetMaskTf), self.targetTf + self.tau * self.targetPrimeTf - self.u)
            y = y1 + y2 + y3 + y4

            # Intermediate nodes for the matrix part
            A1 = tf.matmul(self.tfW, tf.diag(self.rhoPrime))
            A2 = tf.matmul(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(self.tfW), self.u - tfTools.tf_mat_vec_dot(self.tfW, self.rho))), tf.diag(self.rhoPrimePrime))
            AZ = tf.matmul(self.tfW, tf.diag(self.rhoPrime))
            AY = identity - AZ
            AX = tf.matmul(tf.transpose(self.tfW), AY)
            A3 = tf.matmul(tf.diag(self.rhoPrime), AX)
            A4 = self.beta * tf.diag(self.targetMaskTf)
            A = self.tau * (identity - A1 - A2 - A3 + A4)

        # Look at the dynamics of the free neurons only
        inputVector = self.inputPrimeTf * self.inputMaskTf
        yNew = y - tfTools.tf_mat_vec_dot(A,inputVector)
        yRed = tf.slice(y, [nInput], [-1])
        Ared = tf.slice(A, [nInput, nInput], [-1, -1])


        # Calculate the update step for the membrane potentials
        self.uDiff =  tf.cholesky_solve(tf.cholesky(Ared), tf.expand_dims(yRed, 1))[:,0]
        dependencies.append(self.uDiff)

        with tf.control_dependencies(dependencies):
            # Calculate the step in the eligibility trace
            self.eligibilityDiff = self.learningRate * tfTools.tf_outer_product(self.u - tfTools.tf_mat_vec_dot(self.tfWnoWta, self.rho), self.rho)

            # Apply membrane potentials
            self.applyMembranePot = tf.scatter_update(self.u, np.arange(nInput, nFull), tf.slice(self.u, [nInput], [-1]) + self.timeStep * self.uDiff)

            # Apply eligibility trace
            dependencies.append(self.eligibility.assign(self.eligibility + self.timeStep * self.eligibilityDiff))

        with tf.control_dependencies(dependencies):
            # Apply decay to the elifibility trace
            self.applyEligibility = self.eligibility.assign(
                                        self.eligibility*tf.exp(-self.timeStep/self.tauEligibility))


    def initCompGraph(self):

        self.createComputationalGraph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def resetCompGraph(self):

        self.sess.run(tf.global_variables_initializer())
        self.T = 0.
        self.uTraces = []
        self.eligibilityTraces = []

    def applyInput(self):
        """
        Set the membrane potential of the neurons according to the input
        """

        # set the value of the input neurons form the input
        [self.inputValue, self.inputPrime, self.inputMask]  = self.input.getInput(self.T)


    def Update(self):
        """
        Make one update step
        """

        # Get the current value of the input and the target
        self.applyInput()
        self.applyTarget()

        # perform the comp graph with the correct placeholders
        inputs = self.input.getInput(self.T)
        targets = self.target.getTarget(self.T) 
        placeholderDict = {self.tfW : self.W.data,
                           self.tfWnoWta : self.WnoWta,
                           self.inputTf : inputs[0],
                           self.inputPrimeTf : inputs[1],
                           self.inputMaskTf : inputs[2],
                           self.targetTf : targets[0],
                           self.targetPrimeTf : targets[1],
                           self.targetMaskTf : targets[2],
                            }

        #for value in placeholderDict.values():
        #    print value

        # run the updates
        self.sess.run(self.applyMembranePot, placeholderDict)
        self.sess.run(self.applyEligibility, placeholderDict)
        self.T = self.T + self.timeStep

        # Save the traces if applicable
        if self.saveTraces:
            self.uTraces.append(self.sess.run(self.u))
            self.eligibilityTraces.append(self.sess.run(self.eligibility)[~self.W.mask])

    def calcWnoWta(self, nOutputNeurons):
        """
            Calculate the weigth matrix without the WTA. This is necessary as the plasticity has to be calcualted without the WTA network. The WTA is on the soma.
        """

        self.WnoWta = copy.deepcopy(self.W.data)
        self.WnoWta[self.N - nOutputNeurons:, self.N - nOutputNeurons:] = 0.
        #print('The weigth matrix without the WTA:')
        #print(self.WnoWta)

    def run(self, timeDifference):
        """
            run the simulation for the given time Difference
        """

        self.logger.debug('The global time before the run command is: {}'.format(self.T))
        simSteps = int(timeDifference/self.timeStep)
        if abs(simSteps * self.timeStep - timeDifference) > 1e-4:
            self.logger.warning("The simulated time is not an integer multiple of the timestep. This can lead to timing offsets!")
        for i in range(simSteps):
            self.Update()
        self.logger.debug('The global time after the run command is: {}'.format(self.T))

    def getMembPotentials(self):
    	"""
			Run the session and return the mebrane potentials
    	"""

    	return self.sess.run(self.u)

    def getActivities(self):
        """
            Returns the rho of the outpur. Attention this is not the instantaneous spiking probability
        """

        return self.sess.run(self.rho)

    def getEligibilities(self):
        """
            return the eligibilities as they are right now
        """

        return self.sess.run(self.eligibility)

    def getTraces(self):
        """
            Return the saved traces
        """

        elig = np.array(self.eligibilityTraces)
        uMem = np.array(self.uTraces)

        return {'uMem': uMem,
                'eligibilities': elig}

    def updateWeights(self, learningRate, modulator):
        """
            Update the weights according to a three factor learning rule.
            Update = (eligibility trace) * (learning rate) * (modulator)

            Keywords:
                --- learningRate: the learning rate of the rule
                --- modulator: the neuromodulatin signal
        """
        wDummy = copy.deepcopy(self.W)
        self.W = self.W + learningRate * modulator * self.eligibility
        self.W[self.maskIndex] = 0
        self.W[self.wMaxFixed] = wDummy[self.wMaxFixed]

    def calculateWeightUpdates(self, learningRate, modulator):
        """
            calcuate the suggested weight updates

            Keywords:
                --- learningRate: the learning rate of the rule
                --- modulator: the neuromodulatin signal
        """
        return learningRate * modulator * self.sess.run(self.eligibility)

    def applyWeightUpdates(self, deltaW):

        wDummy = copy.deepcopy(self.W)
        self.W = self.W + deltaW
        self.W[self.maskIndex] = 0
        self.W[self.wMaxFixed] = wDummy[self.wMaxFixed]

    def setFixedSynapseMask(self, fixedSynapses):
        """ set a mask for the synapses which stay fixed during training """

        self.wMaxFixed = fixedSynapses


if __name__ == "__main__":
    testClass = lagrangeEligTf()
    testClass.checkSetup()
