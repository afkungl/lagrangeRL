from . import networkBase
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
from lagrangeRL.tools.misc import timer
import logging
import coloredlogs
import sys


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

    def setRegParameters(self, uLow, uHigh, kappaDecay):
        """
        Set the regularization parameters
        """

        self.uLow = uLow
        self.uHigh = uHigh
        self.kappaDecay = kappaDecay

    def setCostWeightings(self, alphaWna, alphaNoise, beta):

        self.alphaWna = alphaWna
        self.alphaNoise = alphaNoise
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
        Connects the methods of the activation function object.
        This makes sure that the activation functions can be changed easily and independently from this network code.
        
        """

        self.actFuncObject = actFuncObject
        self.actFunc = actFuncObject.value
        self.actFuncPrime = actFuncObject.valuePrime
        self.actFuncPrimePrime = actFuncObject.valuePrimePrime

    def setPlasticSynapses(self, Wplastic):
        """
            set as boolean mask the synapses which can change during training
            The plastic synapses are marked with True or 1
        """

        self.Wplastic = Wplastic

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """

        ######################################
        # Variables that are needed
        self.u = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.rLowPass = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        uNoise = tf.Variable(np.zeros(self.N), dtype=self.dtype)
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
        self.wNoWtaMask = tf.Variable(self.Wplastic.astype(float), dtype=self.dtype)

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
            self.rho = self.actFunc(self.u)
            rhoPrime = self.actFuncPrime(self.u)
            rhoPrimePrime = self.actFuncPrimePrime(self.u)
        self.rhoOutput = self.actFunc(self.u)

        ###################################
        # Update the exploration noise on the output neurons
        uNoiseOut = tf.slice(uNoise, [nFull - nOutput], [-1])
        duOutNoise = self.noiseTheta * (self.noiseMean - uNoiseOut) * self.timeStep + self.noiseSigma * \
            np.sqrt(self.timeStep) * \
            tf.random_normal([nOutput], mean=0., stddev=1.0, dtype=self.dtype)
        updateNoise = tf.scatter_update(uNoise,
                                        np.arange(nFull - nOutput, nFull),
                                        uNoiseOut + duOutNoise)

        ####################################
        # Calculate the updates for the membrane potential and for the
        # eligibility trace

        with tf.control_dependencies([updateNoise,
                                      applyInputU,
                                      applyInputUDot]):

            # frequently used tensors are claculated early on
            wNoWtaT = tf.transpose(self.wTfNoWta)
            wNoWtaRho = tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho)
            c = tfTools.tf_mat_vec_dot(wNoWtaT, self.u - wNoWtaRho)
            uOut = self.u * self.outputMaskTf
            uDotOut = self.uDotOld * self.outputMaskTf
            wOnlyWtaT = tf.transpose(self.wTfOnlyWta)
            wOnlyWtaRho = tfTools.tf_mat_vec_dot(self.wTfOnlyWta, self.rho)
            cOnlyWta = tfTools.tf_mat_vec_dot(wOnlyWtaT, uOut - wOnlyWtaRho)

            # The regular component with lookahead
            reg = tfTools.tf_mat_vec_dot(
                self.wTfNoWta, self.rho + rhoPrime * self.uDotOld * self.tau) - self.u

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
                self.wTfOnlyWta, self.rho + rhoPrime * uDotOut) - (uOut + uDotOut * self.tau)
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
        updateLowPassActivity = self.rLowPass.assign((self.rLowPass + self.timeStep / self.tauEligibility * self.rho) * tf.exp(-1. * self.timeStep / self.tauEligibility))

        with tf.control_dependencies([saveOldUDot, updateLowPassActivity]):

            self.updateEligiblity = self.eligibility.assign(
                (self.eligibility + self.timeStep * tfTools.tf_outer_product(
                    self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho), self.rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
            )
            
            self.updateRegEligibility = self.regEligibility.assign(
                (self.regEligibility + self.timeStep * tfTools.tf_outer_product(
                    tf.nn.relu(self.uLow - self.u) -
                    tf.nn.relu(self.u - self.uHigh),
                    self.rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
            )

            #self.applyMembranePot = tf.scatter_update(self.u, np.arange(
            #    nInput, nFull), tf.slice(self.u, [nInput], [-1]) + self.timeStep * tf.slice(uDiff, [nInput], [-1]))

            self.applyMembranePot = self.u.assign(self.u + self.timeStep * uDiff)

        ###############################################
        ## Node to update the weights of the network ##
        ###############################################

        self.updateW = self.wTfNoWta.assign(self.wTfNoWta + (self.learningRate / self.tauEligibility) * (
            self.modulator * self.eligibility + self.kappaDecay * self.regEligibility) * self.Wplastic)

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

    def run(self, timeDifference):
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

    def getLowPassActivity(self):
        """
                        Run the session and return the mebrane potentials
        """

        return self.sess.run(self.rLowPass)

    def getActivities(self):
        """
            Returns the rho of the output. Attention this is not the instantaneous spiking probability
        """

        return self.sess.run(self.rhoOutput)

    def getEligibilities(self):
        """
            return the eligibilities as they are right now
        """

        return self.sess.run(self.eligibility) * self.Wplastic

    def getTraces(self):
        """
            Return the saved traces
        """

        elig = np.array(self.eligibilityTraces)
        uMem = np.array(self.uTraces)

        return {'uMem': uMem,
                'eligibilities': elig}

    def applyWeightUpdates(self, modulator):

        return self.sess.run(self.updateW, {self.modulator: modulator})

    def setFixedSynapseMask(self, fixedSynapses):
        """ set a mask for the synapses which stay fixed during training """

        self.wMaxFixed = fixedSynapses


if __name__ == "__main__":
    testClass = lagrangeEligTf()
    testClass.checkSetup()
