#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment
myExperiment = basicExperiment.expMlWna(jsonFile)
myExperiment.enableCheckpointing(25)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()
