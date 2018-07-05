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

		self.labels = np.array(labels)
		self.nLabel = len(labels)
		self.pathTest = test
		self.pathTrain = train

	def loadTestSet(self):
		"""
			load the test set
		"""

		self.testSet = np.loadtxt(self.pathTest, delimiter=',')
		self.nTest = len(self.testSet[:,0])
		self.counterTest = 0

	def loadTrainSet(self):
		"""
			load the training set
		"""

		self.trainSet = np.loadtxt(self.pathTrain, delimiter=',')
		self.nTrain = len(self.trainSet[:,0])
		self.counterTrain = 0

	def getNextTestExample(self):
		"""
			get the next example in the test set and proceed the iterator
		"""

		data = self.testSet[self.counterTest,1:]
		label = self.testSet[self.counterTest,0]
		labelVector = (label == self.labels).astype(int)

		# Iterate the counter
		self.counterTest = (self.counterTest + 1)%self.nTest

		return [{'data': data,
				 'label': labelVector}]

	def getRandomTestExample(self):
		"""
			get a random example from the test set
		"""

		index = np.random.randint(self.nTest)
		data = self.testSet[index,1:]
		label = self.testSet[index,0]
		labelVector = (label == self.labels).astype(int)

		return [{'data': data,
				 'label': labelVector}]

	def getNextTrainExample(self):
		"""
			get the next example in the training set and proceed the iterator
		"""

		data = self.trainSet[self.counterTrain,1:]
		label = self.trainSet[self.counterTrain,0]
		labelVector = (label == self.labels).astype(int)

		# Iterate the counter
		self.counterTrain = (self.counterTrain + 1)%self.nTrain

		return [{'data': data,
				 'label': labelVector}]

	def getRandomTrainExample(self):
		"""
			get a random example from the training set
		"""

		index = np.random.randint(self.nTrain)
		data = self.trainSet[index,1:]
		label = self.trainSet[index,0]
		labelVector = (label == self.labels).astype(int)

		return [{'data': data,
				 'label': labelVector}]