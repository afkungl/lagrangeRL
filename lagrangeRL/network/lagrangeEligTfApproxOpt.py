from . import lagrangeEligTf
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
from lagrangeRL.tools.misc import timer
import logging
import coloredlogs


class lagrangeEligTfApproxOpt(lagrangeEligTf):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.dtype = tf.float32
        self.T = 0.

        # set up a logger
        self.logger = logging.getLogger('lagrangeEligTfApproxOpt')

        # Set up an own timer
        self.timer = timer()
        self.timerSmall = timer()

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """
        # track the dependencies
        dependencies = []

        # Set up the variables which will be then tracked
        self.u = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uDotOld = tf.Variable(np.zeros(self.N), dtype=self.dtype)
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
            
            # frequently used ones
            Wt = tf.transpose(self.tfW) 
            Wrho = tfTools.tf_mat_vec_dot(self.tfW, self.rho)
            c = tfTools.tf_mat_vec_dot(Wt, self.u - Wrho)

            # calculate the update
            term1 = Wrho + self.tau * tfTools.tf_mat_vec_dot(self.tfW, self.rhoPrime * self.uDotOld) - self.u
            term2 = c * (self.rhoPrime + self.tau * self.rhoPrimePrime * self.uDotOld)
            term3 = self.tau * self.rhoPrime * (tfTools.tf_mat_vec_dot(Wt, self.uDotOld) - tfTools.tf_mat_vec_dot(tf.matmul(Wt, self.tfW), self.rhoPrime * self.uDotOld))
            term4 = self.beta * self.targetMaskTf * ( self.targetTf - self.u + self.tau * (self.targetPrimeTf - self.uDotOld))

            uDiffDummy = term1 + term2 + term3 + term4


        self.uDiff = (1. / self.tau) * uDiffDummy
        dependencies.append(self.uDiff)
        dependencies.append(self.uDotOld.assign(self.uDiff))

        # Calculate the step in the eligibility trace
        dependencies.append(self.eligibilityDiff.assign(self.learningRate * tfTools.tf_outer_product(
            self.u - tfTools.tf_mat_vec_dot(self.tfWnoWta, self.rho), self.rho)))

        self.logger.warn('It is the approximativ lagrange')

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