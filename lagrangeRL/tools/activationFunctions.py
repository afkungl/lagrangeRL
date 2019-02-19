from scipy.special import expit
import tensorflow as tf
import sys
import logging
import coloredlogs


class sigmoid(object):

    def __init__(self, width):
        """
                Sigmoid activation function centered around zero.

                Keyrwords:
                        --- width: the width of the activation function
        """
        self.width = width

    def __call__(self, x):

        value = expit(x / self.width)
        prime = value * (1. - value) * (1. / self.width)
        primePrime = prime * (1. - 2. * value) * (1. / self.width)

        return [value, prime, primePrime]


class sigmoidTf(object):

    def __init__(self, width):
        """
            Sigmoid activation function centered around zero.
            This implementation is compatible with a tensorflow framework

            Keyrwords:
                --- width: the width of the activation function
        """

        self.width = width

    def value(self, x):
        """ Calculate the value of the sigmoid """

        return tf.sigmoid(x / self.width)

    def valuePrime(self, x):

        return self.value(x) * (1. - self.value(x))

    def valuePrimePrime(self, x):

        return self.valuePrime(x) * (1. - 2. * self.value(x))

    def value3Prime(self, x):

        return self.valuePrimePrime(x) * (1. - 2. * self.valuePrime(x)) - 2. * tf.pow(self.valuePrime(x), 2)

class reluTf(object):

    def __init__(self, slope, offset):
        """
            Simple relu actiovation function

            Keyrwords:
                --- slope: slope of the increasing part
                --- offset: point of the nonlinearity
        """

        self.slope = slope
        self.offset = offset

    def value(self, x):
        """ Calculate the value of the activation function """

        return self.slope * tf.nn.relu(x - self.offset)

    def valuePrime(self, x):

        return tf.gradients(self.value(x), x)[0]

    def valuePrimePrime(self, x):

        return tf.gradients(self.valuePrime(x), x)[0]

    def value3Prime(self, x):

        return tf.gradients(self.valuePrimePrime(x), x)[0]

class softReluTf(reluTf):

    def __init__(self, slope, offset, width):
        """
            Simple relu actiovation function

            Keyrwords:
                --- slope: slope of the increasing part
                --- offset: point of the nonlinearity
                --- width: the width of the transotion between the zero and the linear regime
        """

        self.slope = slope
        self.offset = offset
        self.width = width

    def value(self, x):
        """ Calculate the value of the activation function """

        return self.slope * self.width * tf.nn.softplus((x - self.offset)/self.width)

class hardSigmoidTf(object):

    def getLogger(self):
        logger = logging.getLogger(
            'Activation functions: {}'.format(type(self).__name__))

    def __init__(self, width):
        """
            A hard edgy version of the sigmoid activation function or capped Relu. This is excpected to work better for backprop as it is less effected by vanishing gradients problem. The activation function is centered around zero such that the activity at zero x value is the half (standard: 0.5) of the maximum (standard: 1.0). 

            Keyrwords:
                --- width: the width of the activation function
        """
        self.getLogger()
        if width <= 0.:
            self.logger.error(
                'The width of the activation function has to be a positive number. We received: width == {}'.format(width))
            sys.exit()

        self.width = width

    def value(self, x):
        """ Calculate the value of the sigmoid """

        return tf.minimum(1., tf.nn.relu(x + self.width / 2.) / self.width)

    def valuePrime(self, x):

        return tf.gradients(self.value(x), x)[0]

    def valuePrimePrime(self, x):

        return x * 0.

    def value3Prime(self, x):

        return x * 0.
