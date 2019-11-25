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
from .lagrangeTfOptimized import lagrangeTfOptimized


class lagrangeTfDirect(lagrangeTfOptimized):
    """
        Experiment class for the optimized network
    """

    def createComputationalGraph(self):
        """
            Create the computational graph in tensorflow
        """

        ######################################
        # Variables that are needed
        self.u = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.rLowPass = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        uNoise = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uNoiseLowPass = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.uDotOld = tf.Variable(np.zeros(self.N), dtype=self.dtype)
        self.eligNow = tf.Variable(
            np.zeros((self.N, self.N)), dtype=self.dtype)
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
        self.biasTf = tf.Variable(self.biasVector, dtype=self.dtype)
        # set up a mask for the learned weights in self.wTfNoWta
        # note that W.mask must omit the WTA network
        self.wNoWtaMask = tf.Variable(self.Wplastic.astype(float), dtype=self.dtype)

        #####################################
        # Variables for debugging
        self.error = tf.Variable(np.zeros(self.N), dtype=self.dtype)


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
        # Calculate the activations functions using the updated values
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
        # Update the low-pass noise
        with tf.control_dependencies([updateNoise]):
            self.uNoiseLowPass = self.uNoiseLowPass + (self.timeStep/self.tau) * (uNoise - self.uNoiseLowPass)

        ####################################
        # Calculate the updates for the membrane potential and for the
        # eligibility trace

        with tf.control_dependencies([self.uNoiseLowPass, updateNoise, rhoPrime, rhoPrimePrime]):

            # frequently used tensors are claculated early on
            wNoWtaT = tf.transpose(self.wTfNoWta)
            wNoWtaRho = tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho)
            c = tfTools.tf_mat_vec_dot(wNoWtaT, self.u - wNoWtaRho - self.biasTf - self.inputTf - self.uNoiseLowPass)

            # get the matrix side of the equation
            A1 = tf.matmul(self.wTfNoWta, tf.diag(rhoPrime))
            A2 = tf.matmul(tf.diag(c), tf.diag(rhoPrimePrime))
            A3 = tf.matmul(tf.diag(rhoPrime), wNoWtaT)
            A4 = tf.matmul(tf.matmul(tf.diag(rhoPrime),wNoWtaT),
                           tf.matmul(self.wTfNoWta, tf.diag(rhoPrime)))
            A5 = self.beta * self.alphaWna * tf.matmul(self.wTfOnlyWta, tf.diag(rhoPrime))
            A = self.tau*(tf.eye(self.N) - A1 - A2 - A3 + A4 - A5)

            # get the vector side of the equation
            y1 = wNoWtaRho + self.biasTf + self.inputTf + self.alphaNoise * self.beta * uNoise - self.u
            y2 = self.tau * self.inputPrimeTf
            y3 = rhoPrime * c
            y4 = self.tau * rhoPrime * tfTools.tf_mat_vec_dot(
                                                        wNoWtaT, 
                                                        self.inputPrimeTf)
            y5 = self.beta * self.alphaWna * tfTools.tf_mat_vec_dot(
                                        self.wTfOnlyWta, self.rho)
            y6 = rhoPrime * tfTools.tf_mat_vec_dot(wNoWtaT, uNoise - self.uNoiseLowPass)
            y = y1 + y2 + y3 - y4 + y5 - y6

            # Solve the equation for uDot
            #self.uDiff = (1. / self.tau) * tf.linalg.solve(A, y)
            uDiff = tf.linalg.solve(A, tf.expand_dims(y, 1))[:, 0]
            #chol = tf.cholesky(A)
            #uDiff = tf.cholesky_solve(tf.cholesky(A), tf.expand_dims(y, 1))[:, 0]

            """
            # The regular component with lookahead
            reg = tfTools.tf_mat_vec_dot(
                self.wTfNoWta, self.rho + rhoPrime * self.uDotOld * self.tau) - self.u + self.biasTf

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
            regWna = self.beta * self.alphaWna * tfTools.tf_mat_vec_dot(
                self.wTfOnlyWta, self.rho + self.tau * rhoPrime * self.uDotOld)

            # Terms from the exploration noise term
            #eNoise = self.alphaNoise * self.beta * \
            #    ((uNoise) - (uOut + self.tau * uDotOut))
            eNoise = self.alphaNoise * self.beta * uNoise
            """

        #uDiff = (1. / self.tau) * (reg + eV + regWna + eNoise)
        saveOldUDot = self.uDotOld.assign(uDiff)
        updateLowPassActivity = self.rLowPass.assign((self.rLowPass + self.timeStep / self.tauEligibility * self.rho) * tf.exp(-1. * self.timeStep / self.tauEligibility))

        self.eligNowUpdate = self.eligNow.assign(tfTools.tf_outer_product(self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho) - self.biasTf, self.rho))
        errorUpdate = self.error.assign(self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho) - self.biasTf - self.inputTf)

        with tf.control_dependencies([saveOldUDot, updateLowPassActivity, self.eligNowUpdate, errorUpdate]):

            self.updateEligiblity = self.eligibility.assign(
                (self.eligibility + self.timeStep * tfTools.tf_outer_product(
                    self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho) - self.biasTf - self.inputTf - self.uNoiseLowPass, self.rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
            )
            
            self.updateRegEligibility = self.regEligibility.assign(
                (self.regEligibility + self.timeStep * tfTools.tf_outer_product(
                    tf.nn.relu(self.uTarget - self.u),
                    self.rho)) * tf.exp(-1. * self.timeStep / self.tauEligibility)
            )

            #self.applyMembranePot = tf.scatter_update(self.u, np.arange(
            #    nInput, nFull), tf.slice(self.u, [nInput], [-1]) + self.timeStep * tf.slice(uDiff, [nInput], [-1]))

        with tf.control_dependencies([saveOldUDot, updateLowPassActivity, self.eligNowUpdate, errorUpdate, self.updateEligiblity, self.updateRegEligibility]):
            self.applyMembranePot = self.u.assign(self.u + self.timeStep * uDiff)

        ###############################################
        ## Node to update the weights of the network ##
        ###############################################

        self.updateW = self.wTfNoWta.assign(self.wTfNoWta + ( 1. / self.tauEligibility) * (
            self.modulator * self.learningRate * self.eligibility * self.Wplastic + tf.math.abs(self.modulator) * self.learningRateH * self.regEligibility * self.noWnaMask))

        ############################################
        ## Outputs for debugging                  ##
        ############################################

        
    def setBias(self, bias):
        """
            Set a bias vector which is then applied to the network
        """

        self.biasVector = bias

    def getBias(self):
        """
            Print the bias vector fro debugging
        """

        return self.sess.run(self.biasTf)

    def getPlastNow(self):
        """
            Get the plasticity as it is right now
        """

        return self.sess.run(self.eligNow) * self.Wplastic

    def getErrorNow(self):
        """
            Get the plasticity as it is right now
        """

        return self.sess.run(self.error)

    def getWtaNetwork(self):
        """
            Get the Wta network
        """

        return self.sess.run(self.wTfOnlyWta)

    def getNoWtaNetwork(self):
        """
            Get the Wta network without the wta network
        """

        return self.sess.run(self.wTfNoWta)

    def setRegParameters(self, uTarget, learningRateH):
        """
        Set the regularization parameters
        """

        self.uTarget = uTarget
        self.learningRateH = learningRateH