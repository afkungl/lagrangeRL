import time
import datetime


class timer(object):
	"""
		A simple class to easily measure the elapsed time in the program for logging purposes
	"""

	def start(self):
		"""
			Start measuring the time now
		"""

		self.startTime = time.time()

	def stop(self):
		"""
			Stop the measurement and return the difference in a datetime.timedelta object
		"""

		self.stopTime = time.time()
		elapsed = datetime.timedelta(seconds=self.stopTime - self.startTime)

		return elapsed