from . import lagrangeEligTf
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
from lagrangeRL.tools.misc import timer
import logging
import coloredlogs


class lagrangeFromFirst(lagrangeEligTf):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.dtype = tf.float64
        self.T = 0.

        # set up a logger
        self.logger = logging.getLogger('lagrangeEligTfApproxOpt')

        # Set up an own timer
        self.timer = timer()
        self.timerSmall = timer()

    def setAlphaNoise(self, alphaNoise):

        self.alphaNoise = alphaNoise

    def setAlphaWna(self, alphaWna):

        self.alphaWna = alphaWna

    def calcOnlyWta(self, nOutputNeurons):

        self.onlyWta = np.zeros(self.W.data.shape)
        self.onlyWta[self.N - nOutputNeurons:, self.N - nOutputNeurons:] = \
            self.W.data[self.N - nOutputNeurons:, self.N - nOutputNeurons:]

    def calculateWeightUpdates(self, learningRate, modulator):
        """
            calcuate the suggested weight updates

            Keywords:
                --- learningRate: the learning rate of the rule
                --- modulator: the neuromodulating signal
        """
        return learningRate * modulator * (1./self.tauEligibility) * self.sess.run(self.eligibility)

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """
        # track the dependencies
        dependencies = []

        # Set up the variables which will be then tracked
        self.u = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uDotOld = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.tfOnlyWta = tf.Variable(self.onlyWta, dtype=self.dtype)
        self.eligibility = tf.Variable(
            np.zeros((self.N, self.N)), dtype=self.dtype)
        self.eligibilityDiff = tf.Variable(
            np.zeros((self.N, self.N)), dtype=self.dtype)
        self.rho = tf.Variable(np.zeros(self.N),
                               dtype=self.dtype)
        self.rhoPrime = tf.Variable(np.zeros(self.N),
                                    dtype=self.dtype)
        self.rhoPrimePrime = tf.Variable(np.zeros(self.N),
                                         dtype=self.dtype)
        self.rho3Prime = tf.Variable(np.zeros(self.N),
                                     dtype=self.dtype)

        # We need an identity tensor with float values
        identity = tf.Variable(np.eye(self.N),
                               dtype=self.dtype)
        one = tf.Variable(np.eye(1),
                          dtype=self.dtype)

        # Set up the placeholdeers which can be then modified
        self.tfW = tf.placeholder(shape=self.W.shape, dtype=self.dtype)
        self.tfWnoWta = tf.placeholder(shape=self.W.shape, dtype=self.dtype)
        self.inputTf = tf.placeholder(dtype=self.dtype,
                                      shape=(self.N))
        self.inputPrimeTf = tf.placeholder(dtype=self.dtype,
                                           shape=(self.N))
        self.inputMaskTf = tf.placeholder(dtype=self.dtype,
                                          shape=(self.N))
        self.targetTf = tf.placeholder(dtype=self.dtype,
                                       shape=(self.N))
        self.targetPrimeTf = tf.placeholder(dtype=self.dtype,
                                            shape=(self.N))
        self.targetMaskTf = tf.placeholder(dtype=self.dtype,
                                           shape=(self.N))

        # an example input mask is needed to build the comp graph
        inputMask = self.input.getInput(0.)[2]
        nInput = len(np.where(inputMask == 1)[0])
        nFull = len(inputMask)

        # Start the calculations
        # Calculate the activation functions
        dependencies.append(tf.assign(self.rho, self.actFunc(self.u)))
        dependencies.append(
            tf.assign(self.rhoPrime, self.actFuncPrime(self.u)))
        dependencies.append(tf.assign(self.rhoPrimePrime,
                                      self.actFuncPrimePrime(self.u)))
        dependencies.append(tf.assign(self.rho3Prime,
                                      self.actFuncObject.value3Prime(self.u)))

        # set the membrane potential on the input neurons
        dependencies.append(tf.scatter_update(self.u,
                                              np.arange(nInput),
                                              tf.slice(self.inputTf,
                                                       [0],
                                                       [nInput]
                                                       )
                                              )
                            )
        dependencies.append(tf.scatter_update(self.uDotOld,
                                              np.arange(nInput),
                                              tf.slice(self.inputPrimeTf,
                                                       [0],
                                                       [nInput]
                                                       )
                                              )
                            )

        with tf.control_dependencies(dependencies):

            # frequently used tensors are claculated early on
            wNoWtaT = tf.transpose(self.tfWnoWta)
            wNoWtaRho = tfTools.tf_mat_vec_dot(self.tfWnoWta, self.rho)
            c = tfTools.tf_mat_vec_dot(wNoWtaT, self.u - wNoWtaRho)
            uOut = self.u * self.targetMaskTf
            uDotOut = self.uDotOld * self.targetMaskTf
            wOnlyWtaT = tf.transpose(self.tfOnlyWta)
            wOnlyWtaRho = tfTools.tf_mat_vec_dot(self.tfOnlyWta, self.rho)
            cOnlyWta = tfTools.tf_mat_vec_dot(wOnlyWtaT, uOut - wOnlyWtaRho)

            # The regular component with lookahead
            reg = tfTools.tf_mat_vec_dot(
                self.tfWnoWta, self.rho + self.rhoPrime * self.uDotOld) - self.u

            # Error term from the vanilla lagrange
            eVfirst = (-1.) * self.rhoPrime * c
            eVsecond = (-1.) * (self.rhoPrimePrime * self.uDotOld) * c
            eVthird = (-1.) * self.rhoPrime * \
                tfTools.tf_mat_vec_dot(
                wNoWtaT,
                self.uDotOld - tfTools.tf_mat_vec_dot(
                    self.tfWnoWta,
                    self.rhoPrime * self.uDotOld)
            )
            eV = eVfirst + self.tau * (eVsecond + eVthird)

            # terms from the winner nudges all circuit
            eWnaFirst = tfTools.tf_mat_vec_dot(
                self.tfOnlyWta, self.rho + self.rhoPrime * uDotOut) - (uOut + self.tau * uDotOut)
            eWnaSecond = (-1.) * self.rhoPrime * cOnlyWta
            eWnaThird = (-1.) * (self.rhoPrimePrime * uDotOut) * cOnlyWta
            eWnaFourth = (-1.) * self.rhoPrime * \
                tfTools.tf_mat_vec_dot(
                wOnlyWtaT,
                uDotOut - tfTools.tf_mat_vec_dot(
                    self.tfOnlyWta,
                    self.rhoPrime * uDotOut)
            )
            eWna = self.alphaWna * self.beta * \
                (eWnaFirst + eWnaSecond + self.tau * (eWnaThird + eWnaFourth))

            # Terms from the exploration noise term
            noise = self.targetTf * self.targetMaskTf
            noiseDot = self.targetPrimeTf * self.targetMaskTf
            eNoise1 = self.rhoPrimePrime / \
                (2. * self.rhoPrime * self.rhoPrime) * tf.pow(uOut - noise, 2.)
            eNoise2 = (-1.) / self.rhoPrime * (uOut - noise)
            eNoise3 = self.rhoPrimePrime / \
                tf.pow(self.rhoPrime, 2) * (uOut - noise)
            eNoise4 = (-1.) / self.rhoPrime * (uDotOut - noiseDot)
            eNoise5 = (self.rho3Prime * uDotOut * tf.pow(self.rhoPrime, 2) - self.rhoPrimePrime * 2. *
                       self.rhoPrime * self.rhoPrimePrime * uDotOut) / (2. * tf.pow(self.rhoPrime, 4)) * (uOut - noise)
            eNoise6 = self.rhoPrimePrime/tf.pow(self.rhoPrime, 2) * (uOut - noise) * (uDotOut - noiseDot)
            eNoise = self.alphaNoise * self.beta * (eNoise1 + eNoise2 + self.tau * (eNoise3 + eNoise4 + eNoise5 + eNoise6))


        self.uDiff = (1. / self.tau) * (reg + eV + eWna + eNoise)
        dependencies.append(self.uDiff)
        dependencies.append(self.uDotOld.assign(self.uDiff))

        # Calculate the step in the eligibility trace
        dependencies.append(self.eligibilityDiff.assign(tfTools.tf_outer_product(
            self.u - tfTools.tf_mat_vec_dot(self.tfWnoWta, self.rho), self.rho)))

        self.logger.warn('You are using the directly derived version of the WNA and the noise term')

        with tf.control_dependencies(dependencies):

            # Apply membrane potentials
            self.applyMembranePot = tf.scatter_update(self.u, np.arange(
                nInput, nFull), tf.slice(self.u, [nInput], [-1]) + self.timeStep * tf.slice(self.uDiff, [nInput], [-1]))

            # Apply eligibility trace
            dependencies.append(self.eligibility.assign(
                self.eligibility + self.timeStep * self.eligibilityDiff))

        with tf.control_dependencies(dependencies):
            # Apply decay to the elifibility trace
            self.applyEligibility = self.eligibility.assign(
                self.eligibility * tf.exp(-1. * one * self.timeStep / self.tauEligibility))

    def applyWeightUpdates(self, deltaW):

        wDummy = copy.deepcopy(self.W)
        self.W = self.W + deltaW
        self.W[self.maskIndex] = 0
        self.W[self.wMaxFixed] = wDummy[self.wMaxFixed]
