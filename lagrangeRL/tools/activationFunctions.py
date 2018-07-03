from scipy.special import expit

class sigmoid(object):

    def __init__(self, width):
    	"""
    		Sigmoid activation function centered around zero.

    		Keyrwords:
    			--- width: the width of the activation function
    	"""
        self.width = width

    def __call__(self, x):

        value = expit(x / self.width)
        prime = value * (1. - value) * (1./self.width)
        primePrime = prime * (1. - 2.*value) * (1./self.width)

        return [value, prime, primePrime]