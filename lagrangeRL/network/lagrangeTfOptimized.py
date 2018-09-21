from . import networkBase
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
from lagrangeRL.tools.misc import timer
import logging
import coloredlogs


class lagrangeTfOptimized(networkBase.networkBase):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.dtype = tf.float32
        self.T = 0.

        # set up a logger
        self.logger = logging.getLogger('lagrangeTfOptimized')

        # Set up an own timer
        self.timer = timer()
        self.timerSmall = timer()

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

    def deleteTraces(self):

        del self.uTraces
        del self.eligibilityTraces
        self.uTraces = []
        self.eligibilityTraces = []

    def connectActivationFunction(self, actFuncObject):
        """

        :param actFunc: function with one argument and one return
        """

        self.actFuncObject = actFuncObject
        self.actFunc = actFuncObject.value
        self.actFuncPrime = actFuncObject.valuePrime
        self.actFuncPrimePrime = actFuncObject.valuePrimePrime

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """

        ######################################
        # Variables that are needed
        self.u = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uNoise = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uDotOld = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.eligibility = tf.Variable(
            np.zeros((self.N, self.N)), dtype=self.dtype)
        self.regEligibility = tf.Variable(
            np.zeros((self.N, self.N)), dtype=self.dtype)
        self.wTfNoWta = tf.Variable(self.WnoWta, dtype=self.dtype)
        self.wTfOnlyWta = tf.Variable(self.onlyWta, dtype=self.dtype)
        inputMask = self.input.getInput(0.)[2]
        self.inputMaskTf = tf.Variable(inputMask,
                                       dtype=self.dtype)
        outputMask = self.target.getTarget(self.T)[2]
        self.outputMaskTf = tf.Variable(outputMask,
                                        dtype=self.dtype)
        # set up a mask for the learned weights in self.wTfNoWta
        # note that W.mask must omit the WTA network
        self.wNoWtaMask = tf.Variable(np.logical_not(self.W.mask).astype(int),
                                      dtype=self.dtype)

        #####################################
        # Placeholders
        # The only datatransfer between the GPU and the CPU should be the
        # input to the input layer and the modulatory signal
        self.inputTf = tf.placeholder(dtype=self.dtype,
                                      shape=(self.N))
        self.inputPrimeTf = tf.placeholder(dtype=self.dtype,
                                           shape=(self.N))
        self.modulator = tf.placeholder(dtype=self.dtype,
                                        shape=())

        ####################################
        # Aux variables for the calculations
        nInput = len(np.where(inputMask == 1)[0])
        nOutput = len(np.where(outputMask == 1)[0])
        nFull = len(inputMask)

        #####################################################
        # Start the actual calculations for the comp graph  #
        #####################################################

        ####################################
        # Update the values of u and uDotOld according to the input
        applyInputU = tf.scatter_update(self.u,
                                        np.arange(nInput),
                                        tf.slice(self.inputTf,
                                                 [0],
                                                 [nInput]
                                                 )
                                        )
        applyInputUDot = tf.scatter_update(self.uDotOld,
                                           np.arange(nInput),
                                           tf.slice(self.inputPrimeTf,
                                                    [0],
                                                    [nInput]
                                                    )
                                           )

        ####################################
        # Calculate the activations functions using the updated values
        with tf.control_dependencies([applyInputU, applyInputUDot]):
            rho = self.actFunc(self.u)
            rhoPrime = self.actFuncPrime(self.u)
            rhoPrimePrime = self.actFuncPrimePrime(self.u)

        ###################################
        # Update the exploration noise on the output neurons
        uNoiseOut = tf.slice(self.uNoise, [nFull - nOutput], [-1])
        duOutNoise = self.noiseTheta * (self.noiseMean - uNoiseOut) * self.timeStep + self.noiseSigma * \
            np.sqrt(self.timeStep) * \
            tf.random_normal([nOutput], mean=0., stddev=1.0, dtype=self.dtype)
        updateNoise = tf.scatter_update(uNoise,
                                        np.arange(nFull - nOutput, nOutput),
                                        uNoiseOut + duOutNoise)

        ####################################
        # Calculate the updates for the membrane potential and for the
        # eligibility trace

        with tf.control_dependencies([updateNoise,
                                      applyInputU,
                                      applyInputUDot]):

            # frequently used tensors are claculated early on
            wNoWtaT = tf.transpose(self.wTfNoWta)
            wNoWtaRho = tfTools.tf_mat_vec_dot(self.wTfNoWta, rho)
            c = tfTools.tf_mat_vec_dot(wNoWtaT, self.u - wNoWtaRho)
            uOut = self.u * self.outputMaskTf
            uDotOut = self.uDotOld * self.outputMaskTf
            wOnlyWtaT = tf.transpose(self.wTfOnlyWta)
            wOnlyWtaRho = tfTools.tf_mat_vec_dot(self.wTfOnlyWta, rho)
            cOnlyWta = tfTools.tf_mat_vec_dot(wOnlyWtaT, uOut - wOnlyWtaRho)

            # The regular component with lookahead
            reg = tfTools.tf_mat_vec_dot(
                self.wTfNoWta, rho + rhoPrime * self.uDotOld) - self.u

            # Error term from the vanilla lagrange
            eVfirst = rhoPrime * c
            eVsecond = (rhoPrimePrime * self.uDotOld) * c
            eVthird = rhoPrime * \
                tfTools.tf_mat_vec_dot(
                    wNoWtaT,
                    self.uDotOld - tfTools.tf_mat_vec_dot(
                        self.wTfNoWta,
                        rhoPrime * self.uDotOld)
                )
            eV = eVfirst + self.tau * (eVsecond + eVthird)

            # terms from the winner nudges all circuit
            eWnaFirst = tfTools.tf_mat_vec_dot(
                self.wTfOnlyWta, rho + rhoPrime * uDotOut) - (uOut + uDotOut * self.tau)
            eWnaSecond = self.outputMaskTf * rhoPrime * cOnlyWta
            eWnaThird = (self.outputMaskTf *
                         rhoPrimePrime * uDotOut) * cOnlyWta
            eWnaFourth = rhoPrime * \
                tfTools.tf_mat_vec_dot(
                    wOnlyWtaT,
                    uDotOut - tfTools.tf_mat_vec_dot(
                        self.wTfOnlyWta,
                        rhoPrime * uDotOut)
                )
            eWna = self.alphaWna * self.beta * \
                (eWnaFirst + eWnaSecond + self.tau * (eWnaThird + eWnaFourth))

            # Terms from the exploration noise term
            eNoise = self.alphaNoise * self.beta * \
                ((uNoise) - (uOut + self.tau * uDotOut))

        uDiff = (1. / self.tau) * (reg + eV + eWna + eNoise)
        saveOldUDot = self.uDotOld.assign(uDiff)

        with tf.control_dependencies(saveOldUDot):

            self.applyMembranePot = tf.scatter_update(self.u, np.arange(
                nInput, nFull), tf.slice(self.u, [nInput], [-1]) + self.timeStep * tf.slice(uDiff, [nInput], [-1]))

        self.updateEligiblity = self.eligibility.assign(
            (self.eligibility + self.timeStep * tfTools.tf_outer_product(
                self.u - tfTools.tf_mat_vec_dot(self.tfWnoWta, rho), rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
        )
        self.updateRegEligibility = self.regEligibility.assign(
            (self.regEligibility + self.timeStep * tfTools.tf_outer_product(
                tf.nn.relu(self.uLow - self.u) -
                tf.nn.relu(self.u - self.uHigh),
                rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
        )

        ###############################################
        ## Node to update the weights of the network ##
        ###############################################

        self.updateW = self.wNoWtaT.assign(self.wNoWtaT + (self.learningRate / self.tauEligibility) * (
            self.modulator * self.eligibility + self.regEligibility) * self.wNoWtaMask)

    def setNoiseParameter(self, mean, std, corrTime):
        """
            Set the parameters of the OU process on the read-out

            Keywords:
                --- mean: the mean of the OU process
                --- std: standard deviation of the OU process
                --- corrTime: correlation-time of the OU process
        """

        self.noiseTheta = 1. / corrTime
        self.noiseMean = mean
        self.noiseSigma = np.sqrt(2. * self.noiseTheta) * std

    def initCompGraph(self):

        self.createComputationalGraph()
        self.sess = tf.Session(
            config=tf.ConfigProto(log_device_placement=True))
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
        [self.inputValue, self.inputPrime,
            self.inputMask] = self.input.getInput(self.T)

    def Update(self):
        """
        Make one update step
        """

        # perform the comp graph with the correct placeholders
        inputs = self.input.getInput(self.T)
        placeholderDict = {self.inputTf: inputs[0],
                           self.inputPrimeTf: inputs[1],
                           }

        # run the updates
        self.sess.run(self.applyMembranePot, placeholderDict)
        self.sess.run(self.updateEligiblity, placeholderDict)
        self.sess.run(self.updateRegEligibility, placeholderDict)
        self.T = self.T + self.timeStep

        # Save the traces if applicable
        if self.saveTraces:
            self.uTraces.append(self.sess.run(self.u))
            self.eligibilityTraces.append(
                self.sess.run(self.eligibility)[~self.W.mask])

    def calcWnoWta(self, nOutputNeurons):
        """
            Calculate the weigth matrix without the WTA. This is necessary as the plasticity has to be calcualted without the WTA network. The WTA is on the soma.
        """

        self.WnoWta = copy.deepcopy(self.W.data)
        self.WnoWta[self.N - nOutputNeurons:, self.N - nOutputNeurons:] = 0.

    def calcOnlyWta(self, nOutputNeurons):

        self.onlyWta = np.zeros(self.W.data.shape)
        self.onlyWta[self.N - nOutputNeurons:, self.N - nOutputNeurons:] = \
            self.W.data[self.N - nOutputNeurons:, self.N - nOutputNeurons:]

    def run(self, timeDifference, updateNudging=False):
        """
            run the simulation for the given time Difference
        """

        self.logger.debug(
            'The global time before the run command is: {}'.format(self.T))

        simSteps = int(timeDifference / self.timeStep)
        if abs(simSteps * self.timeStep - timeDifference) > 1e-4:
            self.logger.warning(
                "The simulated time is not an integer multiple of the timestep. This can lead to timing offsets!")

        self.timer.start()
        for i in range(simSteps):
            self.Update()
        self.logger.info(
            "Simulating {0} time in the model took {1} wall-clock time.".format(timeDifference, self.timer.stop()))

        self.logger.debug(
            'The global time after the run command is: {}'.format(self.T))

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


    def applyWeightUpdates(self, modulator):

        sess.run(self.updateW, self.modulator: modulator)



if __name__ == "__main__":
    testClass = lagrangeEligTf()
    testClass.checkSetup()
