import tensorflow as tf
import numpy as np
import lagrangeRL
import logging
import coloredlogs
import matplotlib.pyplot as plt
import os

# Metaparameters
layers = [5, 20, 4]
labels = [0, 1, 2, 3]
wInits = []
for i in range(len(layers) - 1):
    high = np.sum(layers[:i + 2])
    mid = np.sum(layers[:i + 1])
    low = int(np.sum(layers[:i]))
    print high, mid, low
    magnitude = 1. / np.sqrt((mid - low) / 2.)
    wInits.append(
        (np.random.rand(high - mid, mid - low) - 0.5) * magnitude)
dataSet = 'simpleData.txt'
learningRate = 0.1
trueReward = 1.
falseReward = -1.
dtype = tf.float32
gammaDecay = 0.9
iterations = 10000

######################
# Load necessary helpers
myData = lagrangeRL.tools.dataHandler.dataHandlerMnist(
    labels,
    dataSet,
    dataSet)
myData.loadTrainSet()
rewardScheme = lagrangeRL.tools.rewardSchemes.maxClassification(
    trueReward,
    falseReward)

# helper functions
def tf_mat_vec_dot(matrix, vector):
    '''
    Matrix product between matrix and vector.
    '''
    return tf.matmul(matrix, tf.expand_dims(vector, 1))[:, 0]

def leaky_relu(x):
    '''
    Leaky relus does not exist in tensorflow 1.0
    '''    

    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)

# Further helper functions to calculate the Jacobian

def map(f, x, dtype=None, parallel_iterations=10):
    '''
    Apply f to each of the elements in x using the specified number of parallel iterations.

    Important points:
    1. By "elements in x", we mean that we will be applying f to x[0],...x[tf.shape(x)[0]-1].
    2. The output size of f(x[i]) can be arbitrary. However, if the dtype of that output
       is different than the dtype of x, then you need to specify that as an additional argument.
    '''
    if dtype is None:
        dtype = x.dtype

    n = tf.shape(x)[0]
    loop_vars = [
        tf.constant(0, n.dtype),
        tf.TensorArray(dtype, size=n),
    ]
    _, fx = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, f(x[j]))),
        loop_vars,
        parallel_iterations=parallel_iterations
    )
    return fx.stack()

def jacobian(fx, x, parallel_iterations=10):
    '''
    Given a tensor fx, which is a function of x, vectorize fx (via tf.reshape(fx, [-1])),
    and then compute the jacobian of each entry of fx with respect to x.
    Specifically, if x has shape (m,n,...,p), and fx has L entries (tf.size(fx)=L), then
    the output will be (L,m,n,...,p), where output[i] will be (m,n,...,p), with each entry denoting the
    gradient of output[i] wrt the corresponding element of x.
    '''
    return map(lambda fxi: tf.gradients(fxi, x)[0],
               tf.reshape(fx, [-1]),
               dtype=x.dtype,
               parallel_iterations=parallel_iterations)


#######################
# Set up the tensorflow model
# Input
x = tf.placeholder(dtype=dtype, shape=(layers[0]))

# network params
wArrayTf = []
for w in wInits:
    wArrayTf.append(tf.Variable(w, dtype=dtype))

# activities
#activities = []
#for index, w in enumerate(wInits):
#    activities.append(tf.Variable(np.zeros(layers[index + 1]), dtype=dtype))

# action probability
#probs = tf.Variable(np.zeros(layers[-1]), dtype=dtype)

# Forward pass action selection
activities = []
for index, w in enumerate(wArrayTf):
    if index == 0:
        activities.append(leaky_relu(tf_mat_vec_dot(w, x)))
    else:
        activities.append(leaky_relu(tf_mat_vec_dot(w, activities[index - 1])))

# Update probs
probs = tf.nn.softmax(activities[-1])
#probs = activities[-1] / tf.reduce_sum(activities[-1])
actionVector = tf.Variable(np.zeros(layers[-1]), dtype=dtype)
actionIndex = tf.Variable(0,
                          dtype=tf.int64)
with tf.control_dependencies(activities + [probs]):
    getAction = actionIndex.assign(tf.multinomial(tf.log([probs]), 1)[0][0])
cleanActionVector = actionVector.assign(tf.zeros([layers[-1]]))
with tf.control_dependencies(activities+ [probs, getAction]):
    getActionVector = tf.scatter_update(actionVector, actionIndex, 1.)


######################################################
#### Update the network parameters with gradients ####
# Modulator
modulatorTf = tf.placeholder(dtype=dtype,
                             shape=())
actionVectorPh = tf.placeholder(dtype=dtype, shape=(layers[-1]))

# get the gradients
wgradArr = []
print('====== set up getting gradients ========')
for wTf in wArrayTf:
    print(activities[-1])
    print(wTf)
    wgradArr.append(tf.stack(jacobian(activities[-1], wTf)))
    print(wgradArr[-1])
    #wgradArr.append((-1.)*tf.gradients(tf.norm(actionVectorPh - probs), wTf)[0])

# create the update arrays
updParArray = []
print('======= update W arrays =======')
for index, wTf in enumerate(wArrayTf):
    print(actionVector - probs)
    print(wgradArr[index])
    updParArray.append(tf.assign(wTf,
                                 wTf + learningRate * modulatorTf * tf.einsum('kij,k->ij',
                                                               wgradArr[index],
                                                               actionVectorPh - probs)))
    #with tf.control_dependencies(wgradArr):
    #   updParArray.append(tf.assign(wTf,
    #                                wTf + learningRate * modulatorTf * wgradArr[index]))


'''
# The gradients
grads = []
for w in wInits:
    grads.append(tf.Variable(w, dtype=dtype))
'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Update the parameters
def updateParams(modulator, actionVec, inputData):
    for upd in updParArray:
        sess.run(upd, {modulatorTf: modulator,
                       actionVectorPh: actionVec,
                       x: inputData})

def makePlot():

    # Make a directory for the output
    outputDir = 'output'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # make a plot to save
    f, ax = plt.subplots(1)
    ax.plot(meanRArr)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('mean reward')    
    ax.set_ylim([-1.05, 1.05])
    f.savefig(os.path.join(outputDir, 'meanReward.png'))
    plt.close(f)

###########################
# for the time being: tests
meanR = 0
meanRArr = [meanR]
for i in range(iterations):

    print('=============== New Iteration =================')
    print('Iteration number: {}'.format(i))
    # clean action
    sess.run(cleanActionVector)

    # obtain input
    inputExample = myData.getRandomTrainExample()[0]
    inputValue = inputExample['data']
    print('Label of the input: {}'.format(inputExample['label']))
    print('The presented input is: {}'.format(inputValue))

    # get activation in the last layer
    activation = sess.run(activities[-1], {x:inputValue})
    print('The activation in the last layer is {}'.format(activation))

    # get action vector
    aVec = sess.run(getActionVector, {x:inputValue})
    print('The obtained action vector in the last layer is {}'.format(aVec))
    print('The desired action vector in the last layer is {}'.format(inputExample['label']))

    # get probabilities
    probValue = sess.run(probs, {x:inputValue})
    print('The probability value {}'.format(probValue))

    # observe reward
    Reward = rewardScheme.obtainReward(inputExample['label'], aVec)
    print('The observed reward is {}'.format(Reward))
    
    
    # The current weights are
    wCurr = sess.run(wArrayTf[-1], {x:inputValue})
    print('The current W is:\n{}'.format(wCurr))

    # The current acticity is
    aCurr = sess.run(activities[-1], {x:inputValue})
    print('The current activity is: {}'.format(aCurr))

    # The gradient of the weights according to the activity is
    gW = sess.run(wgradArr[-1], {x:inputValue})
    print('The observed gradient in the W is:\n{}'.format(gW))
    

    # Update the parameters
    updateParams(Reward - meanR,
                 aVec,
                 inputValue)

    # Update the mean reward
    meanR = gammaDecay * meanR + (1. - gammaDecay) * Reward
    print('The new mean reward is {}'.format(meanR))
    meanRArr.append(meanR)

    # wait for user input in the next cycle
    #input("Press Enter to continue...")
    if i%50 == 0:
        makePlot()



