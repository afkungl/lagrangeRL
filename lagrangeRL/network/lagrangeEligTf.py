from . import networkBase
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRl.tools.tfTools as tfTools

class lagrangeElig(networkBase.networkBase):

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

    def createCompuationalGraph(self):
        """
            Create the computational graph in tensorflow
        """
        # Set up the variables which will be then tracked
        self.u = tf.Variable(np.zeros(self.N), dtype = self.dtype)
        self.eligibility = tf.Variable(np.zeros(self.N), dtype = self.dtype)
        self.rho = tf.Variable(np.zeros(self.N),
                               dtype = self.dtype)
        self.rhoPrime = tf.Variable(np.zeros(self.N),
                                    dtype = self.dtype)
        self.rhoPrimePrime = tf.Variable(np.zeros(self.N),
                                         dtype = self.dtype)
        self.tfW = self.Variable(self.W, dtype = self.dtype)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Set up the placeholdeers which can be then modified    
        self.input = tf.placeholder(dtype = self.dtype,
                                    shape = (self.N))
        self.inputPrime = tf.placeholder(dtype = self.dtype,
                                         shape = (self.N))
        self.inputMask = tf.placeholder(dtype = self.dtype,
                                        shape = (self.N))
        self.target = tf.placeholder(dtype = self.dtype,
                                     shape = (self.N))
        self.targetPrime = tf.placeholder(dtype = self.dtype,
                                          shape = (self.N))
        self.targetMask = tf.placeholder(dtype = self.dtype,
                                         shape = (self.N))

        ## Start the calculations
        # Calculate the activation functions
        tf.assign(self.rho, self.actFunc(self.u))
        tf.assign(self.rhoPrime, self.actFuncPrime(self.u))
        tf.assign(self.rhoPrimePrime, self.actFuncPrimePrime(self.u))

        # Intermediate nodes for the vector
        y1 = tfTools.tf_mat_vec_dot(self.tfW, self.rho)
        y2 = -1. * self.u
        y3 = tfTools.tf_mat_vec_dot(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(self.tfW),self.u - tfTools.tf_mat_vec_dot(self.tfW,self.rho))), tf.diag(self.rhoPrimePrime))
        y4 = tfTools.tf_mat_vec_dot(tf.diag(self.rhoPrime), tfTools.tf_mat_vec_dot(tf.transpose(self.tfW), tf.identity(self.N)-tfTools.tf_mat_vec_dot(self.tfW, tf.diag(self.rhoPrime))))
        y = y1 + y2 + y3 + y4

        # Intermediate nodes for the matrix part
        A1 = tfTools.tf_mat_vec_dot(self.tfW, tf.diag(self.rhoPrime))
        A2 = tfTools.tf_mat_vec_dot(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(self.tfW), self.u - tfTools.tf_mat_vec_dot(self.tfW, self.rho))), tf.diag(self.rhoPrimePrime))
        A3 = tfTools.tf_mat_vec_dot(tf.diag(self.rhoPrime), tfTools.tf_mat_vec_dot(tf.transpose(self.tfW), tf.identity(self.N)-tfTools.tf_mat_vec_dot(self.tfW, tf.diag(self.rhoPrime))))
        A4 = self.beta * tf.diag(self.targetMask)
        A = self.tau * (tf.identity(self.N) - A1 - A2 - A3 + A4)

        # Calculate the update step for the membrane potentials
        self.uDiff =  tf.cholesky_solve(tf.cholesky(A), tf.expand_dims(y, 1))[:,0]

        # Calculate the step in the eligibility trace
        self.eligibilityDiff = self.learningRate * tfTools.tf_outer_product(self.u - tfTools.tf_mat_vec_dot(self.tfW, self.rho), self.rho)

        # Apply membrane potentials
        self.applyMembranePot = self.u.assign(self.u + self.timestep * self.uDiff)

        # Apply eligibility trace
        self.applyEligibility = self.eligibility.assign(self.eligibility + self.timestep * self.eligibilityDiff)

    def initVaribales(self):

        self.sess.run(tf.global_variables_initializer())


    def getUDiff(self):
        """
        Get the value of the derivative of u from the model description
        :return: uDiffNew the derivative of u
        """

        # Calculate the dynamics of the network
        # the dynamics is calculated by solving a linear equation of the form A*uDiff=y
        # calculate the right hand side y
        y1 = np.dot(self.W, self.rho)
        y2 = -1.*self.u
        y3 = np.dot(np.diag(np.dot(self.W.T, self.u - np.dot(self.W, self.rho))), self.rho)
        y4 = self.beta*np.dot(np.diag(self.targetMask), self.targetValue + self.tau*self.targetPrime - self.u)
        y = y1 + y2 + y3 + y4

        # Calculate the right hand side A
        A1 = np.dot(self.W, np.diag(self.rhoPrime))
        A2 = np.dot(np.diag(np.dot(self.W.T, self.u - np.dot(self.W, self.rho))), np.diag(self.rhoPrimePrime))
        A3 = np.dot(np.diag(self.rhoPrime), np.dot(self.W.T, np.identity(self.N)-np.dot(self.W, np.diag(self.rhoPrime))))
        A4 = self.beta * np.diag(self.targetMask)
        A = self.tau * (np.identity(self.N) - A1 - A2 - A3 + A4)

        # Now we look at the dynamics only of the free neurons
        indexFree = np.where(self.inputMask == 0)[0]
        inputPrime = self.inputPrime * self.inputMask
        y = (y - np.dot(A,inputPrime))[indexFree]
        A = A[np.ix_(indexFree, indexFree)]

        # Solve for uDiff
        res = linalg.cgs(A, y, x0=self.uDiff[indexFree])
        uDiff = np.zeros(self.N)
        uDiff[indexFree] = res[0]

        return uDiff


    def getWDiff(self):
        """
        Calculate the plasticity update for the synapses
        :return:
        """


        WDiff = self.learningRate * np.outer(self.u - np.dot(self.W, self.rho), self.rho)

        return WDiff

    def Update(self):
        """
        Make one update step
        """

        # Get the current value of the input and the target
        

    def UpdateOld(self, timeStep):
        """
        make one update in time
        :param timeStep: timestep for the update
        """

        # apply input and constraints
        self.applyInput()
        self.applyActivation()
        self.applyTarget()

        # calculate the derivatives
        uDiffNew = self.getUDiff()
        WDiffNew = self.getWDiff()

        # apply update to the membrane trace
        self.u = self.u + timeStep * uDiffNew
        self.applyInput()

        # apply updates to the eligibility trace
        self.eligibility = self.eligibility + timeStep * WDiffNew
        self.eligibility *= np.exp(-timeStep/self.tauEligibility)
        self.eligibility[self.maskIndex] = 0
        self.uDiff = uDiffNew
        self.T = self.T + timeStep

        # Save the traces if applicable
        if self.saveTraces:
            self.uTraces.append(self.u)
            self.eligibilityTraces.append(self.eligibility[~self.W.mask])

    def run(self, timeDifference):
        """
            run the simulation for the given time Difference
        """

        endSim = self.T + timeDifference
        while self.T < endSim:
            self.Update(self.timeStep)

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
        return learningRate * modulator * self.eligibility

    def applyWeightUpdates(self, deltaW):

        wDummy = copy.deepcopy(self.W)
        self.W = self.W + deltaW
        self.W[self.maskIndex] = 0
        self.W[self.wMaxFixed] = wDummy[self.wMaxFixed]

    def setFixedSynapseMask(self, fixedSynapses):
        """ set a mask for the synapses which stay fixed during training """

        self.wMaxFixed = fixedSynapses


if __name__ == "__main__":
    testClass = lagrangeElig()
    testClass.checkSetup()
