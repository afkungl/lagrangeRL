import numpy as np


class l2weightDecay(object):

    def __init__(self, target, kappa):
        """
                Weight decay of the form:
                        deltaW = - kappa (W - target)

        """

        self.target = target
        self.kappa = kappa

    def getDeltaW(self, currentW):
        """
                Calculate the change in weight based on the current weights
        """

        dW = - self.kappa * (currentW - self.target)

        return dW


class flatValleyL2Decay(object):

    def __init__(self, valleyLower, valleyUpper, kappa):
        """
                Weight decay model with no weight decay in the middle of the valley and l2 like decay other wise:
                        if W < valleyLower:
                                dW = - kappa * (W - valleyLower)
                        if valleyLower < W < valleyUpper:
                                dW = 0
                        if valleyUpper < W:
                                dW = - kappa * (W - valleyUpper)
        """

        self.valleyLower = valleyLower
        self.valleyUpper = valleyUpper
        self.kappa = kappa

    def getDeltaW(self, currentW):
        """
                Calculate the change in weight based on the current weights
        """
        dW = np.zeros(currentW.shape)
        indexBelow = np.where(currentW < self.valleyLower)
        indexAbove = np.where(currentW > self.valleyUpper)
        dW[indexBelow] = - self.kappa * \
            (currentW[indexBelow] - self.valleyLower)
        dW[indexAbove] = - self.kappa * \
            (currentW[indexAbove] - self.valleyUpper)

        return dW
