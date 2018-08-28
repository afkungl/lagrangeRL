from . import lagrangeEligTf
import numpy as np
from scipy.sparse import linalg
import copy
import tensorflow as tf
import lagrangeRL.tools.tfTools as tfTools
import logging
import coloredlogs


class lagrangeEligTfApprox(lagrangeEligTf):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.dtype = tf.float32
        self.T = 0.

        # set up a logger
        self.logger = logging.getLogger('lagrangeEligTfApprox')

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
            # Intermediate nodes for the vector
            y1 = tfTools.tf_mat_vec_dot(self.tfW, self.rho)
            y2 = -1. * self.u
            y3 = tfTools.tf_mat_vec_dot(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(
                self.tfW), self.u - tfTools.tf_mat_vec_dot(self.tfW, self.rho))), self.rho)
            y4 = self.beta * tfTools.tf_mat_vec_dot(tf.diag(
                self.targetMaskTf), self.targetTf + self.tau * self.targetPrimeTf - self.u)
            y = y1 + y2 + y3 + y4

            # Intermediate nodes for the matrix part
            A1 = tf.matmul(self.tfW, tf.diag(self.rhoPrime))
            A2 = tf.matmul(tf.diag(tfTools.tf_mat_vec_dot(tf.transpose(
                self.tfW), self.u - tfTools.tf_mat_vec_dot(self.tfW, self.rho))), tf.diag(self.rhoPrimePrime))
            AZ = tf.matmul(self.tfW, tf.diag(self.rhoPrime))
            AY = identity - AZ
            AX = tf.matmul(tf.transpose(self.tfW), AY)
            A3 = tf.matmul(tf.diag(self.rhoPrime), AX)
            A4 = self.beta * tf.diag(self.targetMaskTf)
            A = self.tau * ((-1.) * A1 - A2 - A3 + A4)

        self.uDiff = (1. / self.tau) * \
            (y - tfTools.tf_mat_vec_dot(A, self.uDotOld))
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
