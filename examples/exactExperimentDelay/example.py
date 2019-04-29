import lagrangeRL
import json

with open('params.json', 'r') as f:
	params = json.load(f)

experiment = lagrangeRL.experiments.expExactLagrangeDelay(params)

# Initilize the experiment
experiment.initialize()
experiment.runSimulation()
