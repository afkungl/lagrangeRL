#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFileTest.json'

# Run the experiment

instances = np.arange(100, 3100, 100)
classRatio = []
for instance in instances:
    fileName = 'Checkpoint/checkpoint_iter{}.json'.format(instance)
    myExperiment = basicExperiment.expMlWna(jsonFile, overwriteOutput=True)
    myExperiment.enableCheckpointing(5000)
    #myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter3000.json')
    myExperiment.loadCheckpoint(fileName)
    confMatrix = myExperiment.runTesting('threeDigitsMnist.txt')
    classRatio.append(np.sum(confMatrix.diagonal())/np.sum(confMatrix).astype(np.float32))
    print('The confusion matrix is:')
    print(confMatrix)
    del myExperiment

# plot the class ratio
f, ax = plt.subplots(1)
ax.plot(instances, classRatio)
ax.set_xlabel('# Iterations')
ax.set_ylabel('class ratio')
f.savefig('Output/classRatio.png')