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


class lagrangeTfOptimized2(lagrangeTfOptimized):
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
            regWna = self.beta * self.alphaWna * tfTools.tf_mat_vec_dot(
                self.wTfOnlyWta, self.rho + self.tau * rhoPrime * self.uDotOld)

            # Terms from the exploration noise term
            #eNoise = self.alphaNoise * self.beta * \
            #    ((uNoise) - (uOut + self.tau * uDotOut))
            eNoise = self.alphaNoise * self.beta * uNoise

        uDiff = (1. / self.tau) * (reg + eV + regWna + eNoise)
        saveOldUDot = self.uDotOld.assign(uDiff)
        updateLowPassActivity = self.rLowPass.assign((self.rLowPass + self.timeStep / self.tauEligibility * self.rho) * tf.exp(-1. * self.timeStep / self.tauEligibility))

        self.eligNowUpdate = self.eligNow.assign(tfTools.tf_outer_product(self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho), self.rho))
        errorUpdate = self.error.assign(self.u - tfTools.tf_mat_vec_dot(self.wTfNoWta, self.rho))

        with tf.control_dependencies([saveOldUDot, updateLowPassActivity, self.eligNowUpdate, errorUpdate]):

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

        ############################################
        ## Outputs for debugging                  ##
        ############################################

        

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