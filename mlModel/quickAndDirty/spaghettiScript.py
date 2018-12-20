import tensorflow as tf
import numpy as np
import lagrangeRL
import logging
import coloredlogs


# Metaparameters
layers = [100, 3]
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
#activities = []
#for index, w in enumerate(wInits):
#    activities.append(tf.Variable(np.zeros(layers[index + 1]), dtype=dtype))

# action probability
#probs = tf.Variable(np.zeros(layers[-1]), dtype=dtype)

# Forward pass action selection
activities = []
for index, w in enumerate(wArrayTf):
    if index == 0:
        activities.append(tf.nn.sigmoid(tf_mat_vec_dot(w, x)))
    else:
        activities.append(tf.nn.sigmoid(tf_mat_vec_dot(w, activities[index - 1])))

# Update probs
probs = tf.nn.softmax(activities[-1])
actionVector = tf.Variable(np.zeros(layers[-1]), dtype=dtype)
actionIndex = tf.Variable(0,
                          dtype=tf.int64)
with tf.control_dependencies(activities + [probs]):
    getAction = actionIndex.assign(tf.multinomial([probs], 1)[0][0])
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
for wTf in wArrayTf:
    #wgradArr.append(tf.gradients(activities[-1], wTf)[0])
    wgradArr.append((-1.)*tf.gradients(tf.norm(actionVectorPh - probs), wTf)[0])

# create the update arrays
updParArray = []
for index, wTf in enumerate(wArrayTf):
	#updParArray.append(tf.assign(wTf,
	#	   						 wTf + modulatorTf * tf_mat_vec_dot(wgradArr[index],
	#	   						 					  actionVector - probs)))
	with tf.control_dependencies(wgradArr):
		updParArray.append(tf.assign(wTf,
									 wTf + learningRate * modulatorTf * wgradArr[index]))


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

###########################
# for the time being: tests
meanR = 0

for i in range(10000):

    print('=============== New Iteration =================')
    print('Iteration number: {}'.format(i))
    # clean action
    sess.run(cleanActionVector)

    # obtain input
    inputExample = myData.getRandomTrainExample()[0]
    inputValue = inputExample['data']

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

    # Update the parameters
    updateParams(Reward - meanR,
    			 aVec,
    			 inputValue)

    # Update the mean reward
    meanR = gammaDecay * meanR + (1. - gammaDecay) * Reward
    print('The new mean reward is {}'.format(meanR))

    # wait for user input in the next cycle
    #input("Press Enter to continue...")

    """
    # get action
    probabs = sess.run(probs, {x: inputValue})
    aVec = sess.run(getActionVector, {x:inputValue})
    action = sess.run(actionIndex)
    output = sess.run(activities[-1])
    print('The obtained activity in the last layer is {}'.format(output))
    print('The obtained probabilities in the last layer is {}'.format(probabs))
    print('The obtained action in the last layer is {}'.format(action))
    print('The obtained action vector in the last layer is {}'.format(aVec))
    

    # observe reward
    Reward = rewardScheme.obtainReward(inputExample['label'], aVec)
    print('The desired action vector in the last layer is {}'.format(inputExample['label']))
    print('The observed reward is {}'.format(Reward))
    input("Press Enter to continue...")

    # debug gradient
    print(wgradArr[-1])
    grad1 = sess.run(wgradArr[-1], {x: inputValue})
    #print('A calculated gradient of wgradArr is {}'.format(grad1))

    # Update the parameters
    updateParams(Reward - np.max([-0.9, meanR]), inputValue)

    # Update the reward
    meanR = gammaDecay * meanR + (1. - gammaDecay) * Reward
    print('The new mean reward is {}'.format(meanR))
	"""