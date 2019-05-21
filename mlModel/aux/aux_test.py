import numpy as np
import tensorflow as tf
import unittest
import tfAux
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class homFuncCase(unittest.TestCase):

    def setUp(self):
        """ Simple setup """
        self.x = tf.placeholder(tf.float32, shape=())
        self.hom = tfAux.homFunc(self.x, -2.0, 2.0, 0.5)
        self.sess = tf.Session()

    def test_homFunc(self):
        """ Check the saved values """

        yArr = np.linspace(-3.0, 3.0, 301)
        value = [self.sess.run(self.hom, {self.x: y}) for y in yArr]
        f, ax = plt.subplots(1)
        ax.plot(yArr, value)
        ax.set_title('Hom-function')
        ax.set_xlabel('Value')
        ax.set_ylabel('Function value')
        plt.savefig('homFunc.pdf')


if __name__ == '__main__':

    unittest.main(verbosity=2)