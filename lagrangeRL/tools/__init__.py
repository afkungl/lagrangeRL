tools = ['networks',
         'inputModels',
         'targetModels',
         'activationFunctions',
         'dataHandler',
         'rewardSchemes',
         'tfTools',
         'visualization',
         'weightDecayModels',
         'misc']

for tool in tools:
    exec('from . import {}'.format(tool))
