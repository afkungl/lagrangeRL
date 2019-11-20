import tensorflow as tf
import numpy as np
import logging
import coloredlogs
from mlModel.aux import tfAux
from lagrangeRL.tools import tfTools


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
        self.logger = logging.getLogger(self.__class__.__name__)


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

    def getInitialWeights(self, wInits):

        self.wInits = wInits

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
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]
            if index == len(self.wArrayTf) - 1:
                self.activities.append(self.actFunc(tfAux.tf_mat_vec_dot(
                                                            w,
                                                            prevAct) + 5.0))
            else:
                self.activities.append(self.actFunc(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct)))

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
            self.updParArray.append(self.getUpdateParameters(index, wTf))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def getUpdateParameters(self, index, wTf):
        '''
            Create a tensor to update the parameters in the connection matrices
        '''

        return tf.assign(wTf,
                         wTf + self.learningRateTf * self.modulatorTf * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                    self.actionVectorPh - self.probs)
                        )


    ###################################
    ##  Methods to interact with an  ##
    ##  experiment                   ##
    ###################################

    def initialize(self):
        """
            Initialize the experiment or reset to an initial state
        """

        self._createInitialWeights()
        self._createComputationalGraph()

    def getActionVector(self, input):
        """
            Get an action vector as a reaction to an input
        """

        # First cleant the action vector
        self.sess.run(self.cleanActionVector)

        # run the session and respond with an action
        return self.sess.run(self.getActionVectorTf,
                             {self.inputPh: input})

    def updateParameters(self,
                         inputVector,
                         actionVector,
                         modulator,
                         learningRate):
        """
            Update the parameters based on the formula
        """

        for upd in self.updParArray:
            self.sess.run(upd, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})

    def _getLastLayerWeights(self):
        """
            Return the weights between the last hidden layer and the output layer.

            For debugging
        """

        return self.sess.run(self.wArrayTf[-1])

    def getWeights(self):

        weights = []
        for w in self.wArrayTf:
            weights.append(self.sess.run(w))

        return weights


class mlNetworkWta(mlNetwork):
    """
    
        This network implementation inherits everything from the machine learning model except for the calcualtion of the gradient. This one approximates the gradient via the winner-nudges-all approximation

    """

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input placeholders
        self.inputPh = tf.placeholder(dtype=self.dtype,
                                      shape=(self.layers[0]))
        self.trueLabel = tf.placeholder(dtype=self.dtype,
                                        shape=(self.layers[-1]))
        self.meanReward = tf.placeholder(dtype=self.dtype,
                                         shape=())
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                             shape=())

        # winner nudges all circuit
        nNeurons = self.layers[-1]
        wnaW = np.ones((nNeurons, nNeurons)) * (-1.)/(nNeurons - 1.)
        np.fill_diagonal(wnaW, 1.) 
        self.wnaTf = tf.Variable(wnaW, dtype=self.dtype)

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        self.memPots = []
        for index, w in enumerate(self.wArrayTf):

            # get the appropriate input
            if index == 0:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]

            # calculate the next layer activity (firing rate)
            # based on the previous activity level
            if index == len(self.wArrayTf) - 1:
                #randomCont = tf.random.normal([self.layers[-1]],
                #                              0.,
                #                              self.noiseSigma)
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                w,
                                prevAct) + 3.0 + tf.random.normal(
                                                    [self.layers[-1]],
                                                    0.,
                                                    self.noiseSigma))
                self.activities.append(
                        self.actFunc(self.memPots[-1]))
            else:
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct))
                self.activities.append(self.actFunc(self.memPots[-1]))

        # get action vector
        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                        dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                                       dtype=tf.int64)
        self.cleanActionVector = self.actionVector.assign(
                                        tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities + [self.cleanActionVector]):
            self.getAction = self.actionIndex.assign(
                                tf.math.argmax(self.activities[-1]))

        with tf.control_dependencies(self.activities+ [self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)

        # obtain reward
        with tf.control_dependencies([self.getActionVectorTf]):
            self.R = tf.cond(tf.reduce_all(tf.equal(self.actionVector,
                                                    self.trueLabel)),
                             lambda: 1.0,
                             lambda: -1.0)

        # Set up the parameter updates
        # Modulator
        self.modulatorTf = self.R - tf.math.maximum(0., self.meanReward)

        # get the gradients
        with tf.control_dependencies([self.getActionVectorTf, self.R, self.modulatorTf] + self.memPots + self.activities):
            self.wgradArr = []
            for wTf in self.wArrayTf:
                self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1],
                                                         wTf)))

            # tensors to update the parameters
            self.updParArray = []
            for index, wTf in enumerate(self.wArrayTf):
                self.updParArray.append(self.getUpdateParameters(index, wTf))

            # do the homoestatic update
            #self.homUpdate = (tf.nn.relu(self.uLow - self.memPots[-1]) - tf.nn.relu(self.memPots[-1] - self.uHigh))
            if len(self.layers) == 2:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[-2]
            with tf.control_dependencies(self.updParArray):
                self.updateH = []
                self.homRule = []
                self.homTarget = []
                for (index, wTf) in enumerate(self.wArrayTf):
                    if index == 0:
                        prevAct = self.inputPh
                    else:
                        prevAct = self.activities[index - 1]

                    # Homeostatis at the border
                    self.homRule.append(tf.nn.relu(self.uLow - self.memPots[index]) - tf.nn.relu(self.memPots[index] - self.uHigh))

                    # Homeostatis towards the middle
                    self.homTarget.append(self.uTarget - self.memPots[index])
                    self.updateH.append(wTf.assign(wTf + 
                        self.learningRateH * tfTools.tf_outer_product(
                                                self.homRule[-1],
                                                prevAct) +
                        self.learningRateHt * tf.math.abs(self.modulatorTf) * 
                        tfTools.tf_outer_product(self.homTarget[-1],
                                                 prevAct)
                                                ))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def doOneIteration(self, inputData, trueLabel, learningRate, meanR):

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: trueLabel,
                     self.meanReward: meanR,
                     self.learningRateTf: learningRate}

        res = self.sess.run(tensors, inputDict)

        self.logger.debug('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))
        self.logger.debug('The correct action vector: {}'.format(trueLabel))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[0]

    def getActionVector(self, inputData):
        '''
            The computation graph is evaluated once and the current 
            action vector is reported

            This should only be used for testing
        '''

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        # for the input dictionary only the input data is real
        # the rest is just placeholder to get an action
        # the learning rate has to be 0.0 in order not to change the weights
        # during testing
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: np.zeros(self.layers[-1]),
                     self.meanReward: 0.0,
                     self.learningRateTf: 0.0}

        res = self.sess.run(tensors, inputDict)

        self.logger.debug('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[1]

    def getUpdateParameters(self, index, wTf):
        '''
            Create a tensor to update the parameters in the connection matrices
        '''

        mat = tfAux.tf_mat_vec_dot(self.wnaTf, self.activities[-1]) * \
                tfAux.homFunc(self.memPots[-1], self.uLow, self.uHigh, 0.5)
 
        return tf.assign(wTf,
                         wTf + self.learningRateTf * self.modulatorTf * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                     mat)
                        )

    def setNoiseSigma(self, noiseSigma):

        self.noiseSigma = noiseSigma

    def setHomeostaticParams(self, learningRateH, uLow, uHigh, learningRateHt, uTarget):

        self.learningRateH = learningRateH
        self.uLow = uLow
        self.uHigh = uHigh
        self.learningRateHt = learningRateHt
        self.uTarget = uTarget

    def updateParameters(self,
                         inputVector,
                         actionVector,
                         modulator,
                         learningRate):
        """
            Update the parameters based on the formula
        """

        for upd in self.updParArray:
            self.sess.run(upd, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})

        self.sess.run(self.updateH, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})




class mlNetworkVerifyBp(mlNetwork):
    """

        This network implementation inherits everything from the machine learning model, but the weights from the last hidden layer to the output layer are not updated.

    """

    def updateParameters(self,
                         inputVector,
                         actionVector,
                         modulator,
                         learningRate):
        """
            Update the parameters based on the formula
        """

        for counter, upd in enumerate(self.updParArray):
            if counter == len(self.updParArray) - 1:
                # break before updating the last layer
                break
            self.sess.run(upd, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})


class mlNetworkWtaVerifyBp(mlNetworkWta):
    """

        This network implementation inherits everything from the wna machine learning model, but the weights from the last hidden layer to the output layer are not updated.

    """

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input placeholders
        self.inputPh = tf.placeholder(dtype=self.dtype,
                                      shape=(self.layers[0]))
        self.trueLabel = tf.placeholder(dtype=self.dtype,
                                        shape=(self.layers[-1]))
        self.meanReward = tf.placeholder(dtype=self.dtype,
                                         shape=())
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                             shape=())

        # winner nudges all circuit
        nNeurons = self.layers[-1]
        wnaW = np.ones((nNeurons, nNeurons)) * (-1.)/(nNeurons - 1.)
        np.fill_diagonal(wnaW, 1.) 
        self.wnaTf = tf.Variable(wnaW, dtype=self.dtype)

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        for index, w in enumerate(self.wArrayTf):
            if index == 0:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]
            if index == len(self.wArrayTf) - 1:
                #randomCont = tf.random.normal([self.layers[-1]],
                #                              0.,
                #                              self.noiseSigma)
                self.lastLayerU = tfAux.tf_mat_vec_dot(
                                w,
                                prevAct) + 5.0 + tf.random.normal(
                                                    [self.layers[-1]],
                                                    0.,
                                                    self.noiseSigma)
                self.activities.append(
                        self.actFunc(self.lastLayerU))
            else:
                self.activities.append(self.actFunc(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct)))

        # get action vector
        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                        dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                                       dtype=tf.int64)
        self.cleanActionVector = self.actionVector.assign(
                                        tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities + [self.cleanActionVector]):
            self.getAction = self.actionIndex.assign(
                                tf.math.argmax(self.activities[-1]))

        with tf.control_dependencies(self.activities+ [self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)

        # obtain reward
        with tf.control_dependencies([self.getActionVectorTf]):
            self.R = tf.cond(tf.reduce_all(tf.equal(self.actionVector,
                                                    self.trueLabel)),
                             lambda: 1.0,
                             lambda: -1.0)

        # Set up the parameter updates
        # Modulator
        self.modulatorTf = self.R - tf.math.maximum(0., self.meanReward)

        # get the gradients
        with tf.control_dependencies([self.getActionVectorTf, self.R, self.modulatorTf]):
            self.wgradArr = []
            for wTf in self.wArrayTf:
                self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1],
                                                         wTf)))

            # do the homoestatic update
            self.homUpdate = (tf.nn.relu(self.uLow - self.lastLayerU) - tf.nn.relu(self.lastLayerU - self.uHigh))
            if len(self.layers) == 2:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[-2]
            self.updateH = self.wArrayTf[-1].assign(self.wArrayTf[-1] + 
                            self.learningRateH * tfTools.tf_outer_product(
                                                    self.homUpdate,
                                                    prevAct))

            # tensors to update the parameters
            self.updParArray = []
            for index, wTf in enumerate(self.wArrayTf):
                if index == len(self.wArrayTf) - 1:
                    continue
                self.updParArray.append(self.getUpdateParameters(index, wTf))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

class mlNetworkSelfPred(mlNetworkWta):
    """
    
        This network implementation inherits everything from the machine learning model except for the calcualtion of the gradient. 
        The gradient is calculated from the self predicting nudging

    """

    def __init__(self, layers, actFunc, actFuncPrime, dtype=tf.float32):
        """ Constructor

        Args:
            layers: list withn the number of layers from the input to the
                output layer.
            actFunc: an activation function file
            dtype: type to be used by tensorflow
        """

        self.layers = layers
        self.actFunc = actFunc
        self.actFuncPrime = actFuncPrime
        self.dtype = tf.float32
        self.logger = logging.getLogger(self.__class__.__name__)

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input placeholders
        self.inputPh = tf.placeholder(dtype=self.dtype,
                                      shape=(self.layers[0]))
        self.trueLabel = tf.placeholder(dtype=self.dtype,
                                        shape=(self.layers[-1]))
        self.meanReward = tf.placeholder(dtype=self.dtype,
                                         shape=())
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                             shape=())
        self.actionBias = tf.Variable(np.zeros(self.layers[-1]),
                                      dtype=self.dtype)

        # winner nudges all circuit
        nNeurons = self.layers[-1]
        wnaW = np.ones((nNeurons, nNeurons)) * (-1.)/(nNeurons - 1.)
        np.fill_diagonal(wnaW, 1.) 
        self.wnaTf = tf.Variable(wnaW, dtype=self.dtype)

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        self.memPots = []
        for index, w in enumerate(self.wArrayTf):

            # get the appropriate input
            if index == 0:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]

            # calculate the next layer activity (firing rate)
            # based on the previous activity level
            if index == len(self.wArrayTf) - 1:
                #randomCont = tf.random.normal([self.layers[-1]],
                #                              0.,
                #                              self.noiseSigma)
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                w,
                                prevAct) + self.actionBias + tf.random.normal(
                                                    [self.layers[-1]],
                                                    0.,
                                                    self.noiseSigma))
                self.activities.append(
                        self.actFunc(self.memPots[-1]))
            else:
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct))
                self.activities.append(self.actFunc(self.memPots[-1]))

        # get action vector
        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                        dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                                       dtype=tf.int64)
        self.cleanActionVector = self.actionVector.assign(
                                        tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities + [self.cleanActionVector]):
            self.getAction = self.actionIndex.assign(
                                tf.math.argmax(self.activities[-1]))

        with tf.control_dependencies(self.activities+ [self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)

        # obtain reward
        with tf.control_dependencies([self.getActionVectorTf]):
            self.R = tf.cond(tf.reduce_all(tf.equal(self.actionVector,
                                                    self.trueLabel)),
                             lambda: 1.0,
                             lambda: -1.0)

        # Set up the parameter updates
        # Modulator
        self.modulatorTf = self.R - tf.math.maximum(0., self.meanReward)

        # get the gradients
        with tf.control_dependencies([self.getActionVectorTf, self.R, self.modulatorTf] + self.memPots + self.activities):
            self.wgradArr = []
            for wTf in self.wArrayTf:
                self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1],
                                                         wTf)))

            # tensors to update the parameters
            self.updParArray = []
            for index, wTf in enumerate(self.wArrayTf):
                self.updParArray.append(self.getUpdateParameters(index, wTf))
            self.updParArray.append(self.getHomParameters())

            # do the homoestatic update
            #self.homUpdate = (tf.nn.relu(self.uLow - self.memPots[-1]) - tf.nn.relu(self.memPots[-1] - self.uHigh))
            if len(self.layers) == 2:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[-2]
            with tf.control_dependencies(self.updParArray):
                self.updateH = []
                self.homRule = []
                for (index, wTf) in enumerate(self.wArrayTf):
                    if index == 0:
                        prevAct = self.inputPh
                    else:
                        prevAct = self.activities[index - 1]
                    if True:
                        self.homRule.append(tf.nn.relu(self.uLow - self.memPots[index]) - tf.nn.relu(self.memPots[index] - self.uHigh))
                        self.updateH.append(wTf.assign(wTf + 
                            self.learningRateH * tfTools.tf_outer_product(
                                                    self.homRule[-1],
                                                    prevAct)))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def getUpdateParameters(self, index, wTf):
        '''
            Create a tensor to update the parameters in the connection matrices
        '''

        eRl = tfAux.tf_mat_vec_dot(self.wnaTf, self.activities[-1])
        eHom = (-1.) * self.memPots[-1]
        eSelfPred = self.actFuncPrime(self.memPots[-1]) * \
                    tfAux.tf_mat_vec_dot(self.wnaTf, self.memPots[-1] - \
                    tfAux.tf_mat_vec_dot(self.wnaTf, self.activities[-1]))

        errorvec = eRl +  0.0 * eHom + eSelfPred
 
        return tf.assign(wTf,
                         wTf + self.learningRateTf * (self.modulatorTf * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                    eRl + eSelfPred) + \
                                    0.0 * tf.abs(self.modulatorTf) * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                    eHom))
                        )

    def getHomParameters(self):
        ''' Add the homeostatic paramters '''

        eHom = (-1.) * self.memPots[-1]
        return tf.assign(self.actionBias, self.actionBias + 0.6 * self.learningRateTf *  tf.abs(self.modulatorTf) * eHom)


class mlNetworkDirectNodePert(mlNetwork):
    """
    
        This network implementation inherits everything from the machine learning model except for the calcualtion of the gradient.

        We use direct node perturbation, that is at each iteration we add
        an extra error vector to the output of the network. This extra error
        serves both as an error vector and as exploratio noise.

    """

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input placeholders
        self.inputPh = tf.placeholder(dtype=self.dtype,
                                      shape=(self.layers[0]))
        self.trueLabel = tf.placeholder(dtype=self.dtype,
                                        shape=(self.layers[-1]))
        self.meanReward = tf.placeholder(dtype=self.dtype,
                                         shape=())
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                             shape=())

        # winner nudges all circuit
        nNeurons = self.layers[-1]
        wnaW = np.ones((nNeurons, nNeurons)) * (-1.)/(nNeurons - 1.)
        np.fill_diagonal(wnaW, 1.) 
        self.wnaTf = tf.Variable(wnaW, dtype=self.dtype)

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        self.memPots = []
        for index, w in enumerate(self.wArrayTf):

            # get the appropriate input
            if index == 0:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]

            # calculate the next layer activity (firing rate)
            # based on the previous activity level
            if index == len(self.wArrayTf) - 1:
                #randomCont = tf.random.normal([self.layers[-1]],
                #                              0.,
                #                              self.noiseSigma)
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                w,
                                prevAct) + 5.0)
                self.activities.append(
                        self.actFunc(self.memPots[-1]))
            else:
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct))
                self.activities.append(self.actFunc(self.memPots[-1]))

        # get action vector
        self.perturbation = tf.random.normal(
            [self.layers[-1]],
            0.,
            self.noiseSigma)

        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                        dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                                       dtype=tf.int64)
        self.cleanActionVector = self.actionVector.assign(
                                        tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities + [self.cleanActionVector, self.perturbation]):
            self.getAction = self.actionIndex.assign(
                                tf.math.argmax(self.actFunc(self.memPots[-1] + self.perturbation)))

        with tf.control_dependencies(self.activities+ [self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)

        # obtain reward
        with tf.control_dependencies([self.getActionVectorTf]):
            self.R = tf.cond(tf.reduce_all(tf.equal(self.actionVector,
                                                    self.trueLabel)),
                             lambda: 1.0,
                             lambda: -1.0)

        # Set up the parameter updates
        # Modulator
        self.modulatorTf = self.R - tf.math.maximum(0., self.meanReward)

        # get the gradients
        with tf.control_dependencies([self.getActionVectorTf, self.R, self.modulatorTf] + self.memPots + self.activities):
            self.wgradArr = []
            for wTf in self.wArrayTf:
                self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1],
                                                         wTf)))

            # tensors to update the parameters
            self.updParArray = []
            for index, wTf in enumerate(self.wArrayTf):
                self.updParArray.append(self.getUpdateParameters(index, wTf))

            # do the homoestatic update
            #self.homUpdate = (tf.nn.relu(self.uLow - self.memPots[-1]) - tf.nn.relu(self.memPots[-1] - self.uHigh))
            if len(self.layers) == 2:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[-2]
            with tf.control_dependencies(self.updParArray):
                self.updateH = []
                self.homRule = []
                for (index, wTf) in enumerate(self.wArrayTf):
                    if index == 0:
                        prevAct = self.inputPh
                    else:
                        prevAct = self.activities[index - 1]
                    if True:
                        self.homRule.append(tf.nn.relu(self.uLow - self.memPots[index]) - tf.nn.relu(self.memPots[index] - self.uHigh))
                        self.updateH.append(wTf.assign(wTf + 
                            self.learningRateH * tfTools.tf_outer_product(
                                                    self.homRule[-1],
                                                    prevAct)))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def doOneIteration(self, inputData, trueLabel, learningRate, meanR):

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: trueLabel,
                     self.meanReward: meanR,
                     self.learningRateTf: learningRate}

        res = self.sess.run(tensors, inputDict)

        self.logger.info('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[0]

    def getActionVector(self, inputData):
        '''
            The computation graph is evaluated once and the current 
            action vector is reported

            This should only be used for testing
        '''

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        # for the input dictionary only the input data is real
        # the rest is just placeholder to get an action
        # the learning rate has to be 0.0 in order not to change the weights
        # during testing
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: np.zeros(self.layers[-1]),
                     self.meanReward: 0.0,
                     self.learningRateTf: 0.0}

        res = self.sess.run(tensors, inputDict)

        self.logger.debug('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[1]

    def getUpdateParameters(self, index, wTf):
        '''
            Create a tensor to update the parameters in the connection matrices
        '''

        with tf.control_dependencies(self.activities + [self.cleanActionVector, self.perturbation]):
            return tf.assign(wTf,
                         wTf + self.learningRateTf * self.modulatorTf * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                    self.perturbation)
                        )

    def setNoiseSigma(self, noiseSigma):

        self.noiseSigma = noiseSigma

    def setHomeostaticParams(self, learningRateH, uLow, uHigh):

        self.learningRateH = learningRateH
        self.uLow = uLow
        self.uHigh = uHigh

    def updateParameters(self,
                         inputVector,
                         actionVector,
                         modulator,
                         learningRate):
        """
            Update the parameters based on the formula
        """

        for upd in self.updParArray:
            self.sess.run(upd, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})

        self.sess.run(self.updateH, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})


class mlNetworkCombinedNodePert(mlNetwork):
    """
    
        This network implementation inherits everything from the machine learning model except for the calcualtion of the gradient.

        We use node perturbation combined with the lateral connections,
        that is at each iteration we add
        an extra error vector to the output of the network. This extra error
        serves both as an error vector and as exploratio noise.

    """

    def __init__(self, layers, actFunc, actFuncPrime, dtype=tf.float32):
        """ Constructor

        Args:
            layers: list withn the number of layers from the input to the
                output layer.
            actFunc: an activation function file
            dtype: type to be used by tensorflow
        """

        self.layers = layers
        self.actFunc = actFunc
        self.actFuncPrime = actFuncPrime
        self.dtype = tf.float32
        self.logger = logging.getLogger(self.__class__.__name__)

    def _createComputationalGraph(self):
        """
            Create the computational graph ready to start
        """

        # Check if the weights are initialized
        if not 'wInits' in dir(self):
            raise RuntimeError("The weights have to be inititalized before creating the computational graph!")

        # Input placeholders
        self.inputPh = tf.placeholder(dtype=self.dtype,
                                      shape=(self.layers[0]))
        self.trueLabel = tf.placeholder(dtype=self.dtype,
                                        shape=(self.layers[-1]))
        self.meanReward = tf.placeholder(dtype=self.dtype,
                                         shape=())
        self.learningRateTf = tf.placeholder(dtype=self.dtype,
                                             shape=())

        # winner nudges all circuit
        nNeurons = self.layers[-1]
        wnaW = np.ones((nNeurons, nNeurons)) * (-1.)/(nNeurons - 1.)
        np.fill_diagonal(wnaW, 1.) 
        self.wnaTf = tf.Variable(wnaW, dtype=self.dtype)

        # network array
        self.wArrayTf = []
        for w in self.wInits:
            self.wArrayTf.append(tf.Variable(w, dtype=self.dtype))

        # forward pass of action selection
        self.activities = []
        self.memPots = []
        for index, w in enumerate(self.wArrayTf):

            # get the appropriate input
            if index == 0:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[index - 1]

            # calculate the next layer activity (firing rate)
            # based on the previous activity level
            if index == len(self.wArrayTf) - 1:
                #randomCont = tf.random.normal([self.layers[-1]],
                #                              0.,
                #                              self.noiseSigma)
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                w,
                                prevAct) + 2.0)
                self.activities.append(
                        self.actFunc(self.memPots[-1]))
            else:
                self.memPots.append(tfAux.tf_mat_vec_dot(
                                                    w,
                                                    prevAct))
                self.activities.append(self.actFunc(self.memPots[-1]))

        # get action vector
        self.perturbation = tf.random.normal(
            [self.layers[-1]],
            0.,
            self.noiseSigma)

        self.actionVector = tf.Variable(np.zeros(self.layers[-1]),
                                        dtype=self.dtype)
        self.actionIndex = tf.Variable(0,
                                       dtype=tf.int64)
        self.cleanActionVector = self.actionVector.assign(
                                        tf.zeros([self.layers[-1]]))
        with tf.control_dependencies(self.activities + [self.cleanActionVector, self.perturbation]):
            self.getAction = self.actionIndex.assign(
                                tf.math.argmax(self.actFunc(self.memPots[-1] + self.perturbation)))

        with tf.control_dependencies(self.activities+ [self.getAction]):
            self.getActionVectorTf = tf.scatter_update(self.actionVector,
                                                     self.actionIndex,
                                                     1.)

        # obtain reward
        with tf.control_dependencies([self.getActionVectorTf]):
            self.R = tf.cond(tf.reduce_all(tf.equal(self.actionVector,
                                                    self.trueLabel)),
                             lambda: 1.0,
                             lambda: -1.0)

        # Set up the parameter updates
        # Modulator
        self.modulatorTf = self.R - tf.math.maximum(0., self.meanReward)

        # get the gradients
        with tf.control_dependencies([self.getActionVectorTf, self.R, self.modulatorTf] + self.memPots + self.activities):
            self.wgradArr = []
            for wTf in self.wArrayTf:
                self.wgradArr.append(tf.stack(tfAux.jacobian(self.activities[-1],
                                                         wTf)))

            # tensors to update the parameters
            self.updParArray = []
            for index, wTf in enumerate(self.wArrayTf):
                self.updParArray.append(self.getUpdateParameters(index, wTf))

            # do the homoestatic update
            #self.homUpdate = (tf.nn.relu(self.uLow - self.memPots[-1]) - tf.nn.relu(self.memPots[-1] - self.uHigh))
            if len(self.layers) == 2:
                prevAct = self.inputPh
            else:
                prevAct = self.activities[-2]
            with tf.control_dependencies(self.updParArray):
                self.updateH = []
                self.homRule = []
                for (index, wTf) in enumerate(self.wArrayTf):
                    if index == 0:
                        prevAct = self.inputPh
                    else:
                        prevAct = self.activities[index - 1]
                    if True:
                        self.homRule.append(tf.nn.relu(self.uLow - self.memPots[index]) - tf.nn.relu(self.memPots[index] - self.uHigh))
                        self.updateH.append(wTf.assign(wTf + 
                            self.learningRateH * tfTools.tf_outer_product(
                                                    self.homRule[-1],
                                                    prevAct)))

        # start the session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def doOneIteration(self, inputData, trueLabel, learningRate, meanR):

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: trueLabel,
                     self.meanReward: meanR,
                     self.learningRateTf: learningRate}

        res = self.sess.run(tensors, inputDict)

        self.logger.info('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[0]

    def getActionVector(self, inputData):
        '''
            The computation graph is evaluated once and the current 
            action vector is reported

            This should only be used for testing
        '''

        tensors = [self.R, self.getActionVectorTf, self.updateH] + self.updParArray + self.activities
        # for the input dictionary only the input data is real
        # the rest is just placeholder to get an action
        # the learning rate has to be 0.0 in order not to change the weights
        # during testing
        inputDict = {self.inputPh: inputData,
                     self.trueLabel: np.zeros(self.layers[-1]),
                     self.meanReward: 0.0,
                     self.learningRateTf: 0.0}

        res = self.sess.run(tensors, inputDict)

        self.logger.debug('The activity in the output layer is: {}'.format(res[-1]))
        self.logger.debug('The action vector is: {}'.format(res[1]))

        if np.isnan(res[-1]).any():
            self.logger.error('There is nan in the activities of the last layer')
            raise RuntimeError('The calculations led to nan values in the activities')

        return res[1]

    def setBeta(self, beta):

        self.beta = beta

    def getUpdateParameters(self, index, wTf):
        '''
            Create a tensor to update the parameters in the connection matrices
        '''

        eRl = tfAux.tf_mat_vec_dot(self.wnaTf, self.activities[-1])
        eHom = (-1.) * self.memPots[-1]
        eSelfPred = self.actFuncPrime(self.memPots[-1]) * \
                    tfAux.tf_mat_vec_dot(self.wnaTf, self.memPots[-1] - \
                    tfAux.tf_mat_vec_dot(self.wnaTf, self.activities[-1]))

        with tf.control_dependencies(self.activities + [self.cleanActionVector, self.perturbation]):
            return tf.assign(wTf,
                         wTf + self.learningRateTf * self.modulatorTf * tf.einsum('kij,k->ij',
                                    self.wgradArr[index],
                                    self.perturbation + 
                                    self.beta * (eRl + eHom + eSelfPred))
                        )

    def setNoiseSigma(self, noiseSigma):

        self.noiseSigma = noiseSigma

    def setHomeostaticParams(self, learningRateH, uLow, uHigh):

        self.learningRateH = learningRateH
        self.uLow = uLow
        self.uHigh = uHigh

    def updateParameters(self,
                         inputVector,
                         actionVector,
                         modulator,
                         learningRate):
        """
            Update the parameters based on the formula
        """

        for upd in self.updParArray:
            self.sess.run(upd, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})

        self.sess.run(self.updateH, {self.modulatorTf: modulator,
                                self.actionVectorPh: actionVector,
                                self.inputPh: inputVector,
                                self.learningRateTf: learningRate})