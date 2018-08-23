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

class rewardSchemeCase(unittest.TestCase):

    def setUp(self):
        """ Simple setup """
        self.true = np.array([0,0,1,0])
        self.falsePred = np.array([1., 0.2, 0.3, .9])
        self.truePred = np.array([0.2, 0.3, 1., .9])
        self.trueReward = 1.
        self.falseReward = -1.
        self.maxScheme = \
            lagrangeRL.tools.rewardSchemes.maxClassification(self.trueReward,
                                                         self.falseReward)


    def test_maxScheme(self):
        """ Check the saved values """

        self.assertEqual(self.trueReward, self.maxScheme.obtainReward(self.truePred, self.true))
        self.assertEqual(self.falseReward, self.maxScheme.obtainReward(self.falsePred, self.true))

class weigthDecays(unittest.TestCase):

    def test_flatValleyL2Decay(self):

        W = np.array([[0.0, 0.2, 0.4],
                        [1.0, 0.8, 0.6],
                        [0.5, 0.1, 0.9]])
        dWExpected = np.array([[2., 0.0, 0.0],
                        [-2., 0.0, 0.0],
                        [0.0, 1., -1.]])

        decayModel = lagrangeRL.tools.weightDecayModels.flatValleyL2Decay(
                                                                0.2,
                                                                0.8,
                                                                10.)
        dW = decayModel.getDeltaW(W)

        self.assertTrue((np.around(dW,3) == np.around(dWExpected, 3)).all())

if __name__ == '__main__':

    unittest.main(verbosity=2)