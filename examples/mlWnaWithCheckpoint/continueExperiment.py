#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment
myExperiment = basicExperiment.expMlWna(jsonFile, overwriteOutput=True)
myExperiment.enableCheckpointing(500)
myExperiment.loadCheckpoint('Checkpoint/checkpoint_iter2000.json')
myExperiment.continueFromCheckpoint()
