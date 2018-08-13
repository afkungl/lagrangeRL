import numpy as np


class constantTarget(object):

    def __init__(self, value, mask):
        """
            Set up a time-constant target according to the given mask

            Keywords:
                --- mask: list of booleans with length N, where to apply the inout
                --- value: array of length N, the values to be set
        """
        self.mask = np.array(mask)
        self.value = np.array(value)

    def getTarget(self, T):
        """
        Provide the input at global time T
            Keywords:
                --- T: global time

            Returns: [vals, primes, mask]
                --- vals: list of lenght N with the values
                --- primes: time derivative of the values
                --- mask: input mask, where the input applies
        """
        vals = self.value
        primes = 0. * vals
        mask = self.mask

        return [vals, primes, mask]


class ornsteinUhlenbeckTarget(object):

    def __init__(self, mask, mean, tau, standardDiv):
        """
            Set up a target where the target value evolves according to an OU process. The values are evolving according to their own process, but the parameters of the processes are the same.

            The values are initialized individually with a gaussian N(mean, stadardDiv)

            Keywords:
                --- mask: list of booleans with length N, where to apply the inout
                --- mean: mean of the OU process
                --- tau: autocorrelation time of the OU process
                --- standardDiv: standard deviation of the OU process 
        """

        self.mask = np.array(mask)
        self.nTotal = len(self.mask)
        self.nudgedInd = np.where(self.mask == 1)[0]
        self.nNudged = len(self.nudgedInd)
        self.mean = mean
        self.tau = tau
        self.theta = 1. / self.tau
        self.standardDiv = standardDiv
        self.sigma = np.sqrt(2. * self.theta) * self.standardDiv

        self.value = np.random.normal(self.mean, self.standardDiv, self.nTotal)

    def updateNudging(self, timeStep):
        """
            Update the OU processes with the given timeStep
        """

        x = self.values[self.nudgedInd]
        dx = self.theta * (self.mean - x) + self.sigma * \
            np.sqrt(timeStep) * np.random.normal(1., 0., self.nNudged)
        self.values[self.nudgedInd] = x + dx

    def getTarget(self, T):
        """
        Provide the input at global time T
            Keywords:
                --- T: global time

            Returns: [vals, primes, mask]
                --- vals: list of lenght N with the values
                --- primes: time derivative of the values
                --- mask: input mask, where the input applies
        """
        vals = self.value
        primes = 0. * vals
        mask = self.mask

        return [vals, primes, mask]
