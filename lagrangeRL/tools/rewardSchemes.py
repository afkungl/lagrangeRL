import numpy as np

class maxClassification(object):

	def __init__(self, rewardCorrect, rewardFalse):
		"""
			Simple reward scheme for a hardmax classifier

			Keywords:
				--- rewardCorrect: amount of obtained reward if the classification is correct
				--- rewardFalse: amount of obtained reward if the classification is false
		"""

		self.rewardCorrect = rewardCorrect
		self.rewardFalse = rewardFalse

	def obtainReward(self, predicted, expected):
		"""

			Obtain the reward based on the output of the label layer layer of a network and the expected label.

			The calculation is only based on a majority vote scheme.

			Keywords:
				--- predicted: the output of the neural network
				--- expected: the expected label vector

			Returns:
				--- reward: the calculated reward
		"""

		expectedIndex = np.argmax(expected)
		predictgedIndex = np.argmax(predicted)

		if expectedIndex == predictgedIndex:
			reward = self.rewardCorrect
		else:
			reward = self.rewardFalse

		return reward