from scipy.special import expit
import tensorflow as tf


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
