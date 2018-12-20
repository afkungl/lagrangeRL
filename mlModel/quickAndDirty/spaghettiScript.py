import tensorflow as tf
import numpy as np
import lagrangeRL
import logging
import coloredlogs


# Metaparameters
layers = [100, 100, 3]
labels = [0, 3, 4]
wInits = []
for i in range(len(layers) - 1):
    high = np.sum(layers[:i + 2])
    mid = np.sum(layers[:i + 1])
    low = int(np.sum(layers[:i]))
    print high, mid, low
    magnitude = 1. / np.sqrt((mid - low) / 2.)
    wInits.append(
        (np.random.rand(high - mid, mid - low) - 0.5) * magnitude)
dataSet = 'dataset.txt'
learningRate = 0.1
trueReward = 1.
falseReward = -1.
dtype = tf.float32
gammaDecay = 0.8


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

#######################
# Set up the tensorflow model
# Input
x = tf.placeholder(dtype=dtype, shape=(layers[0]))

# network params
wArrayTf = []
for w in wInits:
    wArrayTf.append(tf.Variable(w, dtype=dtype))

# activities
activities = []
for index, w in enumerate(wInits):
    activities.append(tf.Variable(np.zeros(layers[index + 1]), dtype=dtype))

# action probability
probs = tf.Variable(np.zeros(layers[-1]), dtype=dtype)

# Forward pass action selection
makes = []
for index, w in enumerate(wArrayTf):
    if index == 0:
        makes.append(activities[index].assign(tf_mat_vec_dot(w, x)))
    else:
        makes.append(activities[index].assign(tf.nn.relu(
            tf_mat_vec_dot(w, activities[index - 1]))))

# Update probs
makes.append(probs.assign(tf.nn.softmax(activities[-1])))
actionVector = tf.Variable(np.zeros(layers[-1]), dtype=dtype)
actionIndex = tf.Variable(0,
                          dtype=tf.int64)
with tf.control_dependencies(makes):
    getAction = actionIndex.assign(tf.multinomial([probs], 1)[0][0])
cleanActionVector = actionVector.assign(tf.zeros([layers[-1]]))
makes.append(getAction)
makes.append(cleanActionVector)
with tf.control_dependencies(makes):
    getActionVector = tf.scatter_update(actionVector, actionIndex, 1.)


# Modulator
modulatorTf = tf.placeholder(dtype=dtype,
                           shape=())

# Update the network parameters with gradients
wgradArr = []
for wTf in wArrayTf:
    #wgradArr.append(tf.gradients(tf.pow(tf.reduce_sum(actionVector - probs),2), w, gate_gradients=[actionVector]))
    wgradArr.append(tf.gradients(probs, wTf))
updParArray = []
#error = actionVector - probs
#for (index, w) in enumerate(wArrayTf):
#    updParArray.append(w.assign(w + tf_mat_vec_dot(wgradArr[index][0], error)))


'''
# The gradients
grads = []
for w in wInits:
    grads.append(tf.Variable(w, dtype=dtype))
'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Update the parameters
def updateParams(modulator):
    for upd in updParArray:
        sess.run(upd, {modulatorTf: modulator})

###########################
# for the time being: tests
meanR = 0

for i in range(12):

    print('=============== New Iteration =================')
    # obtain input
    inputExample = myData.getRandomTrainExample()[0]
    inputValue = inputExample['data']

    # get action
    aVec = sess.run(getActionVector, {x: inputValue})
    action = sess.run(actionIndex)
    output = sess.run(activities[-1])
    probabs = sess.run(probs)
    print('The obtained activity in the last layer is {}'.format(output))
    print('The obtained probabilities in the last layer is {}'.format(probabs))
    print('The obtained action in the last layer is {}'.format(action))
    print('The obtained action vectorin the last layer is {}'.format(aVec))

    # observe reward
    Reward = rewardScheme.obtainReward(inputExample['label'], aVec)
    print('The desired action vectorin the last layer is {}'.format(inputExample['label']))
    print('The observed reward is {}'.format(Reward))

    # debug gradient
    grad1 = sess.run(wgradArr[0])
    print('A calculated gradient is {}'.format(grad1))

    # Update the parameters
    updateParams(Reward - np.max([-0.9, meanR]))

    # Update the reward
    meanR = gammaDecay * meanR + (1. - gammaDecay) * Reward
    print('The new mean reward is {}'.format(meanR))
