import lagrangeRL
import json

with open('params.json', 'r') as f:
    params = json.load(f)

experiment = lagrangeRL.experiments.timeContinuousClassificationDelayedReward(
    params)

# Initilize the experiment
experiment.initialize()
experiment.runSimulation()
