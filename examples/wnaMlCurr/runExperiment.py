#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment

# learning on the tiny dataset
jsonFile = 'paramFileTiny.json'
myExperiment = basicExperiment.expMlWna(jsonFile)
myExperiment.enableCheckpointing(5000)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()
del myExperiment

# learning on the small dataset
jsonFile = 'paramFileSmall.json'
myExperiment = basicExperiment.expMlWna(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter15000.json')
myExperiment.continueFromCheckpoint()
del myExperiment

# learning on the medium dataset
jsonFile = 'paramFileMedium.json'
myExperiment = basicExperiment.expMlWna(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter60000.json')
myExperiment.continueFromCheckpoint()
del myExperiment

# learning on the medium dataset
jsonFile = 'paramFileFull.json'
myExperiment = basicExperiment.expMlWna(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(5000)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter200000.json')
myExperiment.continueFromCheckpoint()
del myExperiment