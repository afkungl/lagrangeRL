#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec



def rewardFunc(true, predicted):
    """
        The reward function R(i,j)
    """

    if true == predicted:
        return 1.0
    elif true != predicted:
        return -1.0


def deltaTensor(i, j):
    """
        Convenience function for the Kornecker delta symbol
    """

    if i == j:
        return 1.0
    else:
        return 0.0


def getRealizations(alphas, sigma):
    """
        Draw a sample for all the x random variables
    """

    return np.random.normal(alphas, sigma)


def createMatrix(mDiagonal, N):
    """
        Create the matrix M from the theorem
    """

    matrix = np.ones((N, N)) * (-1.0) * mDiagonal / (N - 1.)
    np.fill_diagonal(matrix, mDiagonal)

    return matrix


def getProbs(alphas, sigma):
    """
        get the approximate probabilities fro the actions
    """

    probs = np.exp(alphas/sigma)
    probs = probs/np.sum(probs)
    return probs


def getV(x, matrix, true):
    """
        Calculate the vector v
    """

    predicted = np.argmax(x)

    return np.matmul(matrix, x) * rewardFunc(true, predicted)


def getW(x, alphas, true, probs):
    """
        calculate the vector w
    """

    # get the winner
    predicted = np.argmax(x)
    deltaFunc = [deltaTensor(i, predicted) for i in range(len(x))]

    return (deltaFunc - probs) * rewardFunc(true, predicted)


def getCosineAngle(vec1, vec2):
    """
        get cos(\phi), where \phi is the angle between the vectors vec1 and vec2
    """

    dotProduct = np.dot(vec1, vec2)
    norming = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dotProduct/norming


def main(params):

    xWinners = np.zeros(params['N'])
    cosineAngles = []
    matrix = createMatrix(params['diagM'], params['N'])
    print(matrix)
    probs = getProbs(params['alphas'], params['sigma'])
    print(params['alphas'])
    print(probs)
    print(np.sum(probs))

    # generate the given number of samples and accumulate the results
    for index in xrange(params['numberSamples']):

        xReal =  getRealizations(params['alphas'], params['sigma'])
        vModel = getV(xReal, matrix, params['trueAction'])
        vPolicyGradient = getW(xReal,
                               params['alphas'],
                               params['trueAction'],
                               probs)
        cosineAngle = getCosineAngle(vModel, vPolicyGradient)
        predicted = np.argmax(xReal)

        print('######### New Iteration ##########')
        print('The winner action is: {}'.format(predicted))
        print('The true action is: {}'.format(params['trueAction']))
        print('The obtained reward is: {}'.format(
                        rewardFunc(predicted, params['trueAction'])))
        print('The realizations are: {}'.format(xReal))
        print('The model error is: {}'.format(vModel))
        print('The policy gradient error is: {}'.format(vPolicyGradient))
        print('The cos(angle) is: {}'.format(cosineAngle))

        # gather the results
        cosineAngles.append(cosineAngle)
        winnerArg = np.argmax(xReal)
        xWinners[winnerArg] += 1

    # Make the plots
    f, axs = plt.subplots(2)
    f.set_size_inches(9, 5, forward=True)
    gs = gridspec.GridSpec(1, 2, hspace=0.4)
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    
    # Make a histogram of the obtained cos(angle)
    axs[0].hist(cosineAngles)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_xlabel('cosine of angle', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Occurrences', fontsize=12, fontweight='bold')
    xPos = np.array(range(0, len(xWinners)))
    xWinnersRel = xWinners / np.sum(xWinners)

    # Plot the approximated and the simulated action selection probabilities
    axs[1].bar(xPos, xWinnersRel, width=0.35, label='simulated',
           bottom=1E-3, color='tab:blue')
    axs[1].bar(xPos + 0.35, probs, width=0.35,
           label='approx', bottom=1E-3, color='tab:orange')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xlabel('Actions', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Probabilities', fontsize=12, fontweight='bold')
    axs[1].legend()
    plt.savefig('report.pdf')


if __name__ == '__main__':

    params = {'N': 10, # number of neurons (possible actions)
              'numberSamples': 10000, # samples drawn
              'diagM': 0.8, # value of the matrix diagonal
              'alphas': np.random.normal(1.0, 0.2, size=10), # vector of the alpha valuers
              'sigma': 0.5,
              'trueAction': 3, # the fixed number l from the theorem
              }

    main(params)
