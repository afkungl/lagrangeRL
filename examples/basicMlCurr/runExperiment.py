#!/usr/bin/env python

from mlModel.experiments import basicExperiment


# learning on the tiny dataset
jsonFile = 'paramFileTiny.json'
myExperiment = basicExperiment.basicExperiment(jsonFile)
myExperiment.enableCheckpointing(5000)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()
del myExperiment

# learning on the small dataset
jsonFile = 'paramFileSmall.json'
myExperiment = basicExperiment.basicExperiment(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter15000.json')
myExperiment.continueFromCheckpoint()
del myExperiment

# learning on the medium dataset
jsonFile = 'paramFileMedium.json'
myExperiment = basicExperiment.basicExperiment(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter60000.json')
myExperiment.continueFromCheckpoint()
del myExperiment

# learning on the medium dataset
jsonFile = 'paramFileFull.json'
myExperiment = basicExperiment.basicExperiment(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter130000.json')
myExperiment.continueFromCheckpoint()
del myExperiment
