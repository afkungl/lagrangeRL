import lagrangeRL


myData = lagrangeRL.tools.dataHandler.dataHandlerMnist([0,1,2,3],
													   'dataset.txt',
													   'dataset.txt')

myData.loadTestSet()
myData.loadTrainSet()


# Test the test getters
print("testing the getters for test set")
print("Get next example")
for i in range(7):
	print(myData.getNextTestExample())
print("Get random example")
for i in range(7):
	print(myData.getRandomTestExample())

# Test the train getters
print("testing the getters for train set")
print("Get next example")
for i in range(7):
	print(myData.getNextTrainExample())
print("Get random example")
for i in range(7):
	print(myData.getRandomTrainExample())