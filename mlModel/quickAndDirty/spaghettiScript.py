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
learningRate = 0.01
trueReward = 1.
falseReward = -1.
dtype = tf.float32


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
        makes.append(activities[index].assign(
            tf_mat_vec_dot(w, activities[index - 1])))

# Update probs
makes.append(probs.assign(tf.nn.softmax(activities[-1])))

with tf.control_dependencies(makes):

    getAction = tf.multinomial([probs], 1)

# Modulator
modulator = tf.placeholder(dtype=dtype,
                           shape=())

sess = tf.Session()
sess.run(tf.global_variables_initializer())

###########################
# for the time being: tests
