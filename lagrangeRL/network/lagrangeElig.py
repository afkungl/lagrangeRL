from . import networkBase
import numpy as np
from scipy.sparse import linalg
import copy

class lagrangeElig(networkBase.networkBase):

    def setTimeStep(self, timeStep):
        """
        Set the learning rate
        :param eta: the learning rate
        """

        self.timeStep = timeStep

    def setLearningRate(self, eta):
        """
        Set the learning rate
        :param eta: the learning rate
        """

        self.learningRate = eta

    def setTau(self, tau):
        """
        set the membrane time constant
        :param tau: postulated membrane time constant
        """

        self.tau = tau

    def setTauEligibility(self, tau):
        """
        set the time constant of the eligibility trace
        :param tau: postulated membrane time constant
        """

        self.tauEligibility = tau

    def setNudging(self, beta):
        """
        set the value of the nudging strength
        :param beta: strength of the nudging
        """

        self.beta = beta

    def initSimulation(self):
        """
        Set the initial rBar and uDiff values to 0
        """

        # Initialize
        self.rBar = 0. * self.u
        self.uDiff = 0. * self.u
        self.rDiff = 0. * self.u
        self.T = 0.
        # add eligibility trace
        self.eligibility = 0. * copy.deepcopy(self.W)

    def saveTraces(self, booleanVar=True):
        """
            Save the time trace of the voltages and of the eligibility trace
        """

        self.saveTraces = booleanVar
        self.uTraces = []
        self.eligibilityTraces = []



    def getUDiff(self):
        """
        Get the value of the derivative of u from the model description
        :return: uDiffNew the derivative of u
        """

        # Calculate the dynamics of the network
        # the dynamics is calculated by solving a linear equation of the form A*uDiff=y
        # calculate the right hand side y
        y1 = np.dot(self.W, self.rho)
        y2 = -1.*self.u
        y3 = np.dot(np.diag(np.dot(self.W.T, self.u - np.dot(self.W, self.rho))), self.rho)
        y4 = self.beta*np.dot(np.diag(self.targetMask), self.targetValue + self.tau*self.targetPrime - self.u)
        y = y1 + y2 + y3 + y4

        # Calculate the right hand side A
        A1 = np.dot(self.W, np.diag(self.rhoPrime))
        A2 = np.dot(np.diag(np.dot(self.W.T, self.u - np.dot(self.W, self.rho))), np.diag(self.rhoPrimePrime))
        A3 = np.dot(np.diag(self.rhoPrime), np.dot(self.W.T, np.identity(self.N)-np.dot(self.W, np.diag(self.rhoPrime))))
        A4 = self.beta * np.diag(self.targetMask)
        A = self.tau * (np.identity(self.N) - A1 - A2 - A3 + A4)

        # Now we look at the dynamics only of the free neurons
        indexFree = np.where(self.inputMask == 0)[0]
        inputPrime = self.inputPrime * self.inputMask
        y = (y - np.dot(A,inputPrime))[indexFree]
        A = A[np.ix_(indexFree, indexFree)]

        # Solve for uDiff
        res = linalg.cgs(A, y, x0=self.uDiff[indexFree])
        uDiff = np.zeros(self.N)
        uDiff[indexFree] = res[0]

        return uDiff


    def getWDiff(self):
        """
        Calculate the plasticity update for the synapses
        :return:
        """


        WDiff = self.learningRate * np.outer(self.u - np.dot(self.W, self.rho), self.rho)

        return WDiff

    def Update(self, timeStep):
        """
        make one update in time
        :param timeStep: timestep for the update
        """

        # apply input and constraints
        self.applyInput()
        self.applyActivation()
        self.applyTarget()

        # calculate the derivatives
        uDiffNew = self.getUDiff()
        WDiffNew = self.getWDiff()

        # apply update to the membrane trace
        self.u = self.u + timeStep * uDiffNew
        self.applyInput()

        # apply updates to the eligibility trace
        self.eligibility = self.eligibility + timeStep * WDiffNew
        self.eligibility *= np.exp(-timeStep/self.tauEligibility)
        self.eligibility[self.maskIndex] = 0
        self.uDiff = uDiffNew
        self.T = self.T + timeStep

        # Save the traces if applicable
        if self.saveTraces:
            self.uTraces.append(self.u)
            self.eligibilityTraces.append(self.eligibility[~self.W.mask])

    def run(self, timeDifference):
        """
            run the simulation for the given time Difference
        """

        endSim = self.T + timeDifference
        while self.T < endSim:
            self.Update(self.timeStep)

    def getTraces(self):
        """
            Return the saved traces
        """

        elig = np.array(self.eligibilityTraces)
        uMem = np.array(self.uTraces)

        return {'uMem': uMem,
                'eligibilities': elig}

    def updateWeights(self, learningRate, modulator):
        """
            Update the weights according to a three factor learning rule.
            Update = (eligibility trace) * (learning rate) * (modulator)

            Keywords:
                --- learningRate: the learning rate of the rule
                --- modulator: the neuromodulatin signal
        """
        self.W = self.W + learningRate * modulator * self.eligibility
        self.W[self.maskIndex] = 0

    def calculateWeightUpdates(self, learningRate, modulator):
        """
            calcuate the suggested weight updates

            Keywords:
                --- learningRate: the learning rate of the rule
                --- modulator: the neuromodulatin signal
        """
        return learningRate * modulator * self.eligibility

    def applyWeightUpdates(self, deltaW):

        self.W = self.W + deltaW
        self.W[self.maskIndex] = 0



if __name__ == "__main__":
    testClass = lagrangeElig()
    testClass.checkSetup()
