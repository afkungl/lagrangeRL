import numpy as np



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