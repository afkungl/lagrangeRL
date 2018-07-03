import numpy as np

class dataHandlerMnist(object):

	def __init__(self, labels, test, train):
		"""
			Data handler to centralize the data mamangement. The data is assumed to be in MNIST format: txt file, <<,>> delimiter, data rowvise with first data in the row is the label

				Keywords:
					--- labels: list of the labels, position in this list will define the one-hot coding scheme
					--- test: path to test set
					--- train: path to training set
		"""

		self.labels = labels
		self.nLabel = len(labels)
		self.pathTest = test
		self.pathTrain = train

	def loadTestSet(self):
		"""
			load the test set
		"""

		self.testSet = np.loadtxt(self.test, delimiter=',')

	def loadTrainSet(self):
		"""
			load the training set
		"""

		self.trainSet = np.loadtxt(self.train, delimiter=',')