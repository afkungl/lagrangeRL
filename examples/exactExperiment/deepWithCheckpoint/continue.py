import lagrangeRL
import json

with open('params.json', 'r') as f:
	params = json.load(f)

experiment = lagrangeRL.experiments.expExactLagrange(params)

# Initilize the experiment
experiment.loadCheckpoint('Checkpoint/checkpoint_iter600.json')
experiment.enableCheckpointing(5)
experiment.continueFromCheckpoint()
