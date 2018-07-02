tools = ['networks',
		 'inputModels',
		 'targetModels']

for tool in tools:
	exec('from . import {}'.format(tool))