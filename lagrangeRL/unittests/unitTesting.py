import unittest
import numpy as np
import scipy.stats
import lagrangeRL


class constantInputCase(unittest.TestCase):

    def setUp(self):
        """ Simple setup """
        self.value = np.array([0.1, 0.2, 0.3, 0.4])
        self.mask = np.array([0., 0., 1., 1.])
        self.constantInput = lagrangeRL.tools.inputModels.constantInput(
        							self.value,
        							self.mask)

    def test_syntax(self):
        """ Check the saved values """

        [value, prime, mask] = self.constantInput.getInput(10.)
        self.assertTrue((self.value == value).all())
        self.assertTrue((0.*self.value == prime).all())
        self.assertTrue((self.mask == mask).all())

class constantTargetCase(unittest.TestCase):

    def setUp(self):
        """ Simple setup """
        self.value = np.array([-0.1, 0.1, 0.33, 0.41])
        self.mask = np.array([0., 0., 1., 1.])
        self.constantTarget = lagrangeRL.tools.targetModels.constantTarget(
        							self.value,
        							self.mask)

    def test_syntax(self):
        """ Check the saved values """

        [value, prime, mask] = self.constantTarget.getTarget(10.)
        self.assertTrue((self.value == value).all())
        self.assertTrue((0.*self.value == prime).all())
        self.assertTrue((self.mask == mask).all())

if __name__ == '__main__':

    unittest.main(verbosity=2)