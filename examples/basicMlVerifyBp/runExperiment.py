#!/usr/bin/env python

from mlModel.experiments import basicExperiment

# Meta parameters
jsonFile = 'paramFile.json'

# Run the experiment
myExperiment = basicExperiment.expMlVarifyBp(jsonFile)
myExperiment.initializeExperiment()
myExperiment.runFullExperiment()