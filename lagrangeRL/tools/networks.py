import numpy as np
import numpy.ma as ma


def feedForward(layers):
    """
        Create a the connection matrix as a masked matrix of a feedforward network

        Keywords:
            --- layers: list of number of neurons in the consecutive layers
    """

    # get the number of neurons
    N = np.sum(layers)
    W = 2.*(np.random.random((N, N)) - .5)
    WMask = np.ones((N, N))

    low = 0;
    mid = layers[0];
    upper = mid + layers[1]
    for i in range(len(layers) - 1):
        WMask[mid:upper, low:mid] = 0
        if not i == len(layers) - 2:
            low = mid
            mid = upper
            upper += layers[i + 2]

    WX = ma.masked_array(W, mask=WMask.astype(int))
    index = np.where(WX.mask == 1)
    WX.data[index] = 0

    return WX

def feedForwardWtaReadout(layers, wtaStrength=1., offset=0.,
                          noiseMagnitude=1., inhStrength=None,
                          noWtaMask=False):
    """
        Create a the connection matrix as a masked matrix of a feedforward network with a winner-take-all network in the last layer

        Keywords:
            --- layers: list of number of neurons in the consecutive layers
            --- wtaStrength: wieght of the excitatory connections
            --- offset: offset in the feedforward connections
            --- noiseMagnitued: magnitude of the uniform noise in the feedforward connections
            --- inhStrength: the strength of the inhibitory connections if specified
    """

    # get the number of neurons
    N = np.sum(layers)
    W = 2.*noiseMagnitude*(np.random.random((N, N)) - .5) + offset
    WMask = np.ones((N, N))

    low = 0;
    mid = layers[0];
    upper = mid + layers[1]
    for i in range(len(layers) - 1):
        WMask[mid:upper, low:mid] = 0
        if not i == len(layers) - 2:
            low = mid
            mid = upper
            upper += layers[i + 2]

    # create WTA matrix
    if not (inhStrength is None):
        inhW = inhStrength
    elif layers[-1] != 1:
        inhW = wtaStrength / (layers[-1] - 1.)
    
    WX = ma.masked_array(W, mask=WMask.astype(int))
    index = np.where(WX.mask == 1)
    WX.data[index] = 0

    Nlast = layers[-1]
    wta = -1.*np.ones((Nlast, Nlast))*inhW
    np.fill_diagonal(wta, wtaStrength)
    if not noWtaMask:
        WMask[-Nlast:,-Nlast:] = 0
    W[-Nlast:, -Nlast:] = wta

    return WX


def layeredRecurrent(layers):
    """
        Create a the connection matrix as a masked matrix of a feedforward network where each connection has a recurrent conterpart

        Keywords:
            --- layers: list of number of neurons in the consecutive layers
    """

    # get the number of neurons
    N = np.sum(layers)
    W = 2.*(np.random.random((N, N)) - .5)
    WMask = np.ones((N, N))

    low = 0;
    mid = layers[0];
    upper = mid + layers[1]
    for i in range(len(layers) - 1):
        WMask[mid:upper, low:mid] = 0
        if not i == len(layers) - 2:
            low = mid
            mid = upper
            upper += layers[i + 2]
    WMask = WMask * WMask.T # allow the recurrent connections

    WX = ma.masked_array(W, mask=WMask.astype(int))
    index = np.where(WX.mask == 1)
    WX.data[index] = 0

    return WX


def recurrent(N, p):
    """
        Create a the connection matrix as a masked matrix of a random recurrent network

        Keywords:
            --- N: number of neurons in the network
            --- p: creation probability for each possible synapse
    """

    # get the number of neurons
    W = 2. * (np.random.random((N, N)) - .5)
    WMask = np.random.binomial(1, p, (N, N))
    np.fill_diagonal(WMask, 1)

    WX = ma.masked_array(W, mask=WMask.astype(int))
    index = np.where(WX.mask == 1)
    WX.data[index] = 0

    return WX



if __name__ == '__main__':


    W = recurrent(5, 0.6)
    print(W.data)
    print(W)

