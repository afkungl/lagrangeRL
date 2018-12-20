import numpy as np 
import tensorflow as tf

eta = 0.1
x = tf.Variable([0,-1], dtype=tf.float64)
f = tf.pow((x - 10.), 2)
dx = tf.gradients(f, x)
updX = x.assign(x - eta * dx[0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20):
	der = sess.run(dx)
	print('Current value of dx is: {}'.format(der))
	newX = sess.run(updX)
	print('Current value of x is: {}'.format(newX))