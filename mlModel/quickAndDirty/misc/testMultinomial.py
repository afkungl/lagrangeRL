import tensorflow as tf
import numpy as np

# metadata
dtype = tf.float32
length = 4

x = tf.placeholder(dtype=dtype, shape=length)
actionIndex = tf.multinomial(tf.log([x]), 1)[0][0]
actionVector = tf.Variable(np.zeros(length), dtype=dtype)
cleanActionVector = actionVector.assign(tf.zeros([length]))
getActionVector = tf.scatter_update(actionVector, actionIndex, 1.)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
	liste = [[0,0,0,0],
			 [1,0,0,0],
			 [3,2,2,2],
			 [6,0,0,0],
			 [10,0,0,0],
			 [100,0,0,0]]

	for l in liste:
		print('===== new ======')
		print('The probability vector is: {}'.format(l))
		print('The  is: {}'.format(sess.run(getActionVector,
											{x: np.array(l) + 0.000000001})))
		sess.run(cleanActionVector)

	print('======== only for 1,0,0,0 ========')
	print('The probability vector is: {}'.format([1,0,0,0]))
	for i in range(20):
		
		print('The  is: {}'.format(sess.run(getActionVector,
											{x: np.array([1,0,0,0]) + 0.000001})))
		sess.run(cleanActionVector)		