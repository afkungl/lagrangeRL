import tensorflow as tf
import numpy as np
import logging
import coloredlogs
from mlModel.aux import tfAux


class mlNetwork(object):


    def __init__(self, layers, actFunc, dtype=tf.float32):
        """ Constructor

        Args:
            layers: list withn the number of layers from the input to the
                output layer.
            actFunc: an activation function file
            dtype: type to be used by tensorflow
        """

        self.layers = layers
        self.actFunc = actFunc
        self.dtype = tf.float32


    def _createInitialWeights(self):
        """
            Create the initial weights
            We use the He et al (2015) initialization

        """

        self.wInits = []

        for index in range(len(self.layers) - 1):
            norm = np.sqrt(2./self.layers[index])
            W = np.random.randn(self.layers[index + 1],
                                self.layers[index]) * norm
            self.wInits.append(W)

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input
        self.inputPh = tf.placeholder(dtype=self.dtype, shape=(self.layers[0]))

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        for index, w in enumerate(self.wArrayTf):
            if index == 0:
                self.activities.append(self.actFunc(tfAux.tf_mat_vec_dot(
                                                            w,
                                                            self.inputPh)))
            else:
                self.activities.append(self.actFunc(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    self.activities[index - 1])))

        # get probabilities and action
        self.probs = tf.nn.softmax(self.activities[-1])
        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                                 dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                          dtype=tf.int64)
        with tf.control_dependencies(self.activities + [self.probs]):
            self.getAction = self.actionIndex.assign(tf.multinomial(
                                        tf.log([self.probs]), 1)[0][0])
        self.cleanActionVector = self.actionVector.assign(tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities+ [self.probs,
                                                       self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)


        # Set up the parameter updates
        # Modulator
        self.modulatorTf = tf.placeholder(dtype=self.dtype,
                                     shape=())
        self.actionVectorPh = tf.placeholder(dtype=self.dtype,
                                             shape=(self.layers[-1]))
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                     shape=())

        # get the gradients
        self.wgradArr = []
        for wTf in self.wArrayTf:
            self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1], wTf)))

        # tensors to update the parameters
        self.updParArray = []
        for index, wTf in enumerate(self.wArrayTf):
            self.updParArray.append(tf.assign(wTf,
                                         wTf + self.learningRateTf * self.modulatorTf * tf.einsum('kij,k->ij',
                                                     self.wgradArr[index],
                                                     self.actionVectorPh - self.probs)
                                         )
                               )

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    ###################################
    ##  Methods to interact with an  ##
    ##  experiment                   ##
    ###################################

    def getActionVector(self, input):
        """
            Get an action vector as a reaction to an input
        """

        # First cleant the action vector
        self.sess.run(self.cleanActionVector)

        # run the session and respond with an action
        return self.sess.run(self.getActionVectorTf,
                             {self.inputPh: input})