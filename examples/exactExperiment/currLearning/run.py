import lagrangeRL
import json

################
## NOTE: The datsets are not part of the repository as they 
##       are potentially too large

# run on the tiny dataset
with open('paramsTiny.json', 'r') as f:
    params = json.load(f)
experiment = lagrangeRL.experiments.expExactLagrange(params)
experiment.initialize()
experiment.enableCheckpointing(5)
experiment.runSimulation()
del experiment

# run on the small dataset
with open('paramsSmall.json', 'r') as f:
    params = json.load(f)
experiment = lagrangeRL.experiments.expExactLagrange(params)
experiment.loadCheckpoint('Checkpoint/checkpoint_iter10.json')
experiment.enableCheckpointing(5)
experiment.continueFromCheckpoint()
del experiment

# run on the medium dataset
with open('paramsMedium.json', 'r') as f:
    params = json.load(f)
experiment = lagrangeRL.experiments.expExactLagrange(params)
experiment.loadCheckpoint('Checkpoint/checkpoint_iter20.json')
experiment.enableCheckpointing(5)
experiment.continueFromCheckpoint()
del experiment

# run on the full dataset
with open('paramsFull.json', 'r') as f:
    params = json.load(f)
experiment = lagrangeRL.experiments.expExactLagrange(params)
experiment.loadCheckpoint('Checkpoint/checkpoint_iter30.json')
experiment.enableCheckpointing(5)
experiment.continueFromCheckpoint()
del experiment
