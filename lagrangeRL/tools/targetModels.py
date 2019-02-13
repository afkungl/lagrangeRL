import numpy as np
import logging
import coloredlogs
import copy

# Set up the stimulus
class constantTarget(object):

    def __init__(self, value, mask):
        """
            Set up a time-constant input according to the given mask

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

    logger = logging.getLogger('ornsteinUhlenbeckTarget')
    coloredlogs.install()

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
        self.standardDiv = standardDiv

        self.theta = 1. / self.tau
        self.sigma = np.sqrt(2. * self.theta) * self.standardDiv
        
        self.value = np.random.normal(self.mean, self.standardDiv, self.nTotal)

    def updateNudging(self, timeStep):
        """
            Update the OU processes with the given timeStep
        """

        x = self.value[self.nudgedInd]
        dx = self.theta * (self.mean - x) * timeStep + self.sigma * \
            np.sqrt(timeStep) * np.random.normal(0., 1., self.nNudged)
        self.value[self.nudgedInd] += dx

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
        vals = copy.deepcopy(self.value)
        primes = 0. * vals
        mask = self.mask

        return [vals, primes, mask]