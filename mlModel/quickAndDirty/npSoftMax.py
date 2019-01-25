import numpy as np 


def npSoftMax(x):

	y = np.exp(x)

	return y/np.sum(y)


if __name__ == "__main__":

	liste = [[0,0,0,0],
			 [1,0,0,0],
			 [3,2,2,2],
			 [6,0,0,0],
			 [10,0,0,0],
			 [100,0,0,0]]

	for l in liste:
		print('===== new ======')
		print('The original vector is: {}'.format(l))
		print('The softmax is: {}'.format(npSoftMax(l)))