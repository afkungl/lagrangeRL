tools = ['networks',
		 'inputModels',
		 'targetModels',
		 'activationFunctions']

for tool in tools:
	exec('from . import {}'.format(tool))