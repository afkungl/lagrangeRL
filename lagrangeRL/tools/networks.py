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
                          noWtaMask=False, fixedPatternNoiseSigma=0.0):
    """
        Create a the connection matrix as a masked matrix of a feedforward network with a winner-take-all network in the last layer.
        The network uses He initialization (He et al IEEE Comp Vision 2015).

        Keywords:
            --- layers: list of number of neurons in the consecutive layers
            --- wtaStrength: wieght of the excitatory connections
            --- offset: offset in the feedforward connections
            --- noiseMagnitued: magnitude of the uniform noise in the feedforward connections
            --- inhStrength: the strength of the inhibitory connections if specified
    """

    # get the number of neurons
    N = np.sum(layers)
    W = np.zeros((N, N)) # Placeholder
    WMask = np.ones((N, N))

    low = 0;
    mid = layers[0];
    upper = mid + layers[1]
    for i in range(len(layers) - 1):
        WMask[mid:upper, low:mid] = 0
        # Initialize the weights
        numbBefore = mid - low
        numbAfter = upper - mid
        norm = np.sqrt(2./float(numbBefore))
        W[mid:upper, low:mid] = np.random.randn(numbAfter,
                                                numbBefore) * norm
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
    relNoise = 1. + np.random.normal(0.0,
                                     fixedPatternNoiseSigma,
                                     size=(Nlast,Nlast))
    relNoise = np.maximum(relNoise, 0.0)
    wta = wta * relNoise
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


def actorCritic(layers,
                layersCritic,
                wtaStrength):
    """
        Create the connectivity matrix of the actor critic architecture
        The mask does not contain the wta network, but the wta network is created

        Args:
            -- layers: layers in the actor network
            -- layersCritic: layers in the critic network. has to end with one
                            and the first layer of the actor is shared with the critic
            -- wtaStrength: strength of the self-excitation in the winner-nudges-all circuit
    """
    
    # get metaparameters
    nAll = np.sum(layers) + np.sum(layersCritic)
    nActor = np.sum(layers)
    W = np.zeros((nAll, nAll))
    wMask = np.ones((nAll, nAll))

    # use existing function of the actor path
    wActor = feedForwardWtaReadout(layers,
                                   wtaStrength,
                                   noWtaMask=True)
    W[:nActor, :nActor] = wActor.data
    wMask[:nActor, :nActor] = wActor.mask

    # set the eight from the input layer to the first hidden layer in the 
    # critic
    numbBefore = layers[0]
    numbAfter = layersCritic[0]
    norm = np.sqrt(2./float(numbBefore))
    W[nActor:nActor + layersCritic[0], :layers[0]] = np.random.randn(numbAfter,
                                            numbBefore) * norm
    wMask[nActor:nActor + layersCritic[0], :layers[0]] = False

    # set the connectivity in the critic network
    low = nActor;
    mid = nActor + layersCritic[0];
    upper = mid + layersCritic[1]
    for i in range(len(layersCritic) - 1):
        wMask[mid:upper, low:mid] = False
        # Initialize the weights
        numbBefore = mid - low
        numbAfter = upper - mid
        norm = np.sqrt(2./float(numbBefore))
        W[mid:upper, low:mid] = np.random.randn(numbAfter,
                                                numbBefore) * norm
        print(i)
        if not i == len(layersCritic) - 2:
            low = mid
            mid = upper
            upper += layersCritic[i + 2]

    return ma.masked_array(W, wMask)



if __name__ == '__main__':


    W = recurrent(5, 0.6)
    print(W.data)
    print(W)

