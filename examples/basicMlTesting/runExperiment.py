#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment
myExperiment = basicExperiment.basicExperiment(jsonFile)
myExperiment.enableCheckpointing(100)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()