import numpy as np
import sys
import numpy.ma as ma


class networkBase(object):

    def __init__(self):
        """
        Initialize an empty class instance
        """

        self.W = np.nan
        self.u = np.nan
        self.r = np.nan
        self.rDiff = np.nan
        self.input = np.nan
        self.target = np.nan
        self.actFunc = np.nan
        self.weightDecay = None


    def addMatrix(self, W):
        """

        :param W: masked array, the connection matrix of the
        """

        self.W = W
        self.N = len(W[:,0])

        # make sure that the data below the mask is zero
        self.maskIndex = np.where(W.mask == 1)
        self.W[self.maskIndex] = 0

    def connectWeightDecay(self, decayModel):

        self.weightDecay = decayModel

    def connectInput(self, input):
        """

        :param input: callable class defining the input
        """

        self.input = input

    def connectTarget(self, target):
        """

        :param target: callable class defining the target
        """

        self.target = target

    def connectActivationFunction(self, actFunc):
        """

        :param actFunc: function with one argument and one return
        """

        self.actFunc = actFunc

    def setInitialConditions(self, uInit):
        """

        :param uInit: define the initial conditions for the u parameters
        """

        self.u = uInit

    def checkSetup(self):
        """
        Check if the necessary components are set and/or connected
        """
        attrToCheck = ['W', 'u', 'input', 'target', 'actFunc']

        missingAttr = []  # type: List[str]
        for attr in attrToCheck:
            gAttr = np.isnan(getattr(self, attr))
            if gAttr.size == 1:
                if gAttr:
                    missingAttr.append(attr)

        if missingAttr:
            sys.exit('The following attributes of the model were not specified: {}'.format(missingAttr))

        print('The model class is initialized.')

    def applyInput(self):
        """
        Set the membrane potential of the neurons according to the input
        """

        # set the value of the input neurons form the input
        [self.inputValue, self.inputPrime, self.inputMask]  = self.input.getInput(self.T)
        indices = np.where(self.inputMask == 1)[0]
        self.u[indices] = self.inputValue[indices]

    def applyActivation(self):
        """
        calculate and save the activation of the neurons
        """

        [self.rho, self.rhoPrime, self.rhoPrimePrime] = self.actFunc(self.u)

    def applyTarget(self):

        [self.targetValue, self.targetPrime, self.targetMask] = self.target.getTarget(self.T)

if __name__ == "__main__":

    testClass = networkBase()
    testClass.addMatrix(ma.masked_equal(np.ones((5,6)),2))
    testClass.checkSetup()

