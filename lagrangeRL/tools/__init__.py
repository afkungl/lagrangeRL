tools = ['networks',
		 'inputModels',
		 'targetModels',
		 'activationFunctions',
		 'dataHandler',
		 'rewardSchemes',
		 'tfTools',
		 'visualization']

for tool in tools:
	exec('from . import {}'.format(tool))