import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
import numpy as np
from scipy.special import expit
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def hide_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def show_axis(ax):
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

def make_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plotReport(figName,
               timeStep,
               traces,
               outputU,
               outputRho,
               target,
               data,
               figSize,
               wCurrent,
               eligs,
               signDeltaW,
               simTime = None):
    """
        Function to plot the report of one iteration
    """

    # make the figure and the axes grid
    plt.rcParams["font.family"] = "serif"
    width = 12.
    ratio = 0.75
    fig = plt.figure( figsize=(width,ratio*width))
    gs_main = gs.GridSpec(2, 1, hspace=0.2, height_ratios=[1,2],
                          left=0.07, right=0.93, top=.97, bottom=0.07)
    gs_upper = gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0], wspace=0.2,
                                           width_ratios=[1, 1])
    gs_lower = gs.GridSpecFromSubplotSpec(2, 3, gs_main[1, 0], wspace=0.2,
    									  hspace=.35,
                                           width_ratios=[1, 1, 1])
    axMemb = plt.Subplot(fig, gs_upper[0])
    fig.add_subplot(axMemb)
    axElig = plt.Subplot(fig, gs_upper[1])
    fig.add_subplot(axElig)
    axOutputRaw = plt.Subplot(fig, gs_lower[0,0])
    fig.add_subplot(axOutputRaw)
    axData = plt.Subplot(fig, gs_lower[0,1])
    fig.add_subplot(axData)
    axCurrentW = plt.Subplot(fig, gs_lower[0,2])
    fig.add_subplot(axCurrentW)
    axOutputRho = plt.Subplot(fig, gs_lower[1,0])
    fig.add_subplot(axOutputRho)
    axDeltaW = plt.Subplot(fig, gs_lower[1,1])
    fig.add_subplot(axDeltaW)
    axDeltaWSign = plt.Subplot(fig, gs_lower[1,2])
    fig.add_subplot(axDeltaWSign)

    if not(simTime is None):
        lastN = int(simTime/timeStep)
        traces['uMem'] = traces['uMem'][-lastN:,:]
        traces['eligibilities'] = traces['eligibilities'][-lastN:,:]


    # make a timearray
    timeArray = np.arange(len(traces['uMem'][:,0])) * timeStep

    # Plot the membrane potentials
    uMems = traces['uMem']
    make_spines(axMemb)
    for index in range(len(uMems[0,:])):
        axMemb.plot(timeArray, uMems[:, index])
    axMemb.set_xlabel(r'time $[ms]$')
    axMemb.set_ylabel('memb. pot. [a.u.]')

    # Plot the eligibility traces
    uElig = traces['eligibilities']
    make_spines(axElig)
    for index in range(len(uElig[0,:])):
        axElig.plot(timeArray, uElig[:, index])
    axElig.set_xlabel(r'time $[ms]$')
    axElig.set_ylabel('elig. traces [a.u.]')

    # bar plots of the membrane potentials
    nOutput = len(outputU)
    width = 0.7
    xPos = np.arange(1, nOutput + 1)
    make_spines(axOutputRaw)
    axOutputRaw.bar(xPos, outputU, width=width)
    axOutputRaw.set_xlabel('output neurons')
    axOutputRaw.set_ylabel('memb. pot. [a.u.]')
    axOutputRaw.set_xticks(np.arange(1, nOutput + 1))

    # bar plots of the membrane potentials
    nOutput = len(outputRho)
    width = 0.7
    xPos = np.arange(1, nOutput + 1)
    make_spines(axOutputRho)
    axOutputRho.bar(xPos, outputRho, width=width, zorder=1)
    axOutputRho.set_xlabel('output neurons')
    axOutputRho.set_ylabel('activity')
    axOutputRho.set_xticks(np.arange(1, nOutput + 1))
    # Add target marker to the plot
    h = np.max(outputRho)/2.
    axOutputRho.scatter(target, h, marker='x', s=100, zorder=2)
    hwinner = np.max(outputRho) * .75
    axOutputRho.scatter(np.argmax(outputRho) + 1, hwinner, marker='D', s=100, zorder=2)

    # print the data
    im = np.reshape(data, figSize)
    show_axis(axData)
    axData.imshow(im, cmap='bwr',
                  aspect=1., interpolation='nearest')
    axData.set_xticks([], [])
    axData.set_yticks([], [])
    axData.set_title('presented data')

    # plot the current weights
    show_axis(axCurrentW)
    maxAbs = np.max(np.abs(wCurrent))
    imAx = axCurrentW.imshow(wCurrent, cmap='bwr',
                  aspect=1, interpolation='nearest',
                  vmin=-1.*maxAbs, vmax=maxAbs)
    cax = inset_axes(axCurrentW,
                 width="5%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc=3,
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axCurrentW.transAxes,
                 borderpad=0,
                 )
    cbar = colorbar(imAx, cax=cax)
    axCurrentW.set_ylabel('input')
    axCurrentW.set_xlabel('neurons')
    axCurrentW.set_title('current weights')
    axCurrentW.set_xticks(np.arange(0, nOutput))

    # plot the final eligibility traces
    show_axis(axDeltaW)
    maxAbs = np.max(np.abs(eligs))
    imAx = axDeltaW.imshow(eligs, cmap='bwr',
                  aspect=1, interpolation='nearest',
                  vmin=-1.*maxAbs, vmax=maxAbs)
    cax = inset_axes(axDeltaW,
                 width="5%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc=3,
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axDeltaW.transAxes,
                 borderpad=0,
                 )
    cbar = colorbar(imAx, cax=cax)
    axDeltaW.set_ylabel('input')
    axDeltaW.set_title('final elig.')
    axDeltaW.set_xlabel('neurons')
    axDeltaW.set_xticks(np.arange(0, nOutput))

    # plot the sign of the appplied weight chages
    show_axis(axDeltaWSign)
    imAxSign = axDeltaWSign.imshow(signDeltaW, cmap='bwr',
                  aspect=1, interpolation='nearest',
                  vmin=-1., vmax=1.)
    caxSign = inset_axes(axDeltaWSign,
                 width="5%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc=3,
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axDeltaWSign.transAxes,
                 borderpad=0,
                 )
    cbar = colorbar(imAxSign, cax=caxSign)
    axDeltaWSign.set_ylabel('input')
    axDeltaWSign.set_title(r'$\mathrm{sign} (\Delta W)$')
    axDeltaWSign.set_xlabel('neurons')
    axDeltaWSign.set_xticks(np.arange(0, nOutput))

    fig.savefig(figName, dpi=200)
    plt.close(fig)

def plotLearningReport(Warray,
                       rewardArray,
                       rewardArrays,
                       figName):

    # make the figure and the axes grid
    plt.rcParams["font.family"] = "serif"
    width = 8.
    ratio = 1.
    fig = plt.figure( figsize=(width,ratio*width))
    gs_main = gs.GridSpec(2, 1, hspace=0.2, height_ratios=[1,1],
                          left=0.12, right=0.95, top=.97, bottom=0.07)

    axWeights = plt.Subplot(fig, gs_main[0])
    fig.add_subplot(axWeights)
    axReward = plt.Subplot(fig, gs_main[1])
    fig.add_subplot(axReward)

    # Plot the evolution of the weights
    make_spines(axWeights)
    iterationArray = np.arange(1, len(Warray[:,1]) + 1)
    for index in range(len(Warray[0,:])):
        axWeights.plot(iterationArray, Warray[:, index])
    axWeights.grid(True, linestyle='--')
    axWeights.set_xlabel('# iterations')
    axWeights.set_ylabel('weights')

    # Plot the moving average of the reward
    make_spines(axReward)
    axReward.plot(iterationArray, rewardArray, label='mean reward')
    for key in rewardArrays:
        label = 'reward class {}'.format(key)
        axReward.plot(iterationArray, rewardArrays[key], label=label)
    axReward.grid(True, linestyle='--')
    axReward.legend(fontsize=8)

    axReward.set_xlabel('# iterations')
    axReward.set_ylabel('mean reward')

    fig.savefig(figName, dpi=200)
    plt.close(fig)

def main():
    """
        Function for local testing
    """
    figureName = 'report.png'
    timestep = 0.1
    traces = {}
    traces['uMem'] = np.random.rand(100, 5)
    traces['eligibilities'] = np.random.rand(100, 5)
    outputU = np.random.rand(3)
    outputRho = expit(outputU)
    target = 2
    data = np.array([1,1,-1,-1])
    figSize=(2,2)
    wCurrent = np.random.rand(4, 3)
    elig = np.random.rand(4,3)
    signDeltaW = np.sign(np.random.rand(4,3) - .5)
    print(signDeltaW)


    plotReport(figureName,
               timestep,
               traces,
               outputU,
               outputRho,
               target,
               data,
               figSize,
               wCurrent,
               elig,
               signDeltaW)

if __name__ == '__main__':
    main()