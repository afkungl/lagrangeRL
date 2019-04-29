#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment
myExperiment = basicExperiment.expMlWna(jsonFile)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()
