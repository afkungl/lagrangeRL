import numpy as np


# Set up the stimulus
class constantInput(object):

    def __init__(self, value, mask):
        """
            Set up a time-constant input according to the given mask

            Keywords:
                --- mask: list of booleans with length N, where to apply the inout
                --- value: array of length N, the values to be set
        """
        self.mask = np.array(mask)
        self.value = np.array(value)

    def getInput(self, T):
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


class smoothedConstantInput(object):
    """
        Constant input with initial sinusoidal ramp up and ramp down at the end
    """

    def __init__(self, value, mask, Tfull, Tramp):
        """
            Setting up the smoothed constant input

            Keywords:
                --- mask: list of booleans with length N, where to apply the
                input
                --- value: array of length N, the values to be set
                --- T: length of the stimulus period
                --- Tr: length of the ramp up and ramp down phase at the beginning and of the end of the stimulus
        """

        self.mask = np.array(mask)
        self.value = np.array(value)
        self.Tfull = Tfull
        self.Tramp = Tramp

    def getInput(self, t):
        """
        Provide the input at global time T
            Keywords:
                --- T: global time

            Returns: [vals, primes, mask]
                --- vals: list of lenght N with the values
                --- primes: time derivative of the values
                --- mask: input mask, where the input applies
        """

        # Transfer the time into the relevant period
        tmod = (t - int(t / self.Tfull) * self.Tfull) * np.sign(t)

        # Calculate the value and the prime multiplier in the three cases
        if tmod < self.Tramp:
            x = (tmod * np.pi) / self.Tramp
            valueMultiplier = 0.5 * (1. - np.cos(x))
            primeMultiplier = np.pi / \
                (2. * self.Tramp) * np.sin(x)
        elif ((tmod >= self.Tramp) and (tmod <= (self.Tfull - self.Tramp))):
            valueMultiplier = 1.
            primeMultiplier = 0.
        elif tmod > (self.Tfull - self.Tramp):
            x = ((tmod - (self.Tfull - self.Tramp)) / self.Tramp ) * np.pi 
            valueMultiplier = 0.5 * (1. + np.cos(x))
            primeMultiplier =  -1. * np.sin(x) * np.pi / (2. * self.Tramp)

        vals = self.value * valueMultiplier
        primes = self.value * primeMultiplier
        mask = self.mask

        return [vals, primes, mask]

