tools = ['networks',
         'inputModels',
         'targetModels',
         'activationFunctions',
         'dataHandler',
         'rewardSchemes',
         'tfTools',
         'visualization',
         'weightDecayModels']

for tool in tools:
    exec('from . import {}'.format(tool))
