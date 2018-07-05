tools = ['networks',
		 'inputModels',
		 'targetModels',
		 'activationFunctions',
		 'dataHandler',
		 'rewardSchemes']

for tool in tools:
	exec('from . import {}'.format(tool))