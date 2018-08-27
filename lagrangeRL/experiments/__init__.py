experiments = ['trialBasedClassification',
               'trialBasedClassificationSmoothed',
               'timeContinuousClassification',
               'timeContinuousClassificationSmoothed',
               'timeContinuousClassificationSmOu',
               'timeContinuousClassificationSmOuApproxLagrange',
               'timeContinuousClassificationDelayedReward',
               'timeContinuousClassificationDelayedRewardSmoothed',
               'timeContinuousClassificationVerifyBackprop',
               'timeContOuDelayed']

for experiment in experiments:
    exec('from .{0} import {0}'.format(experiment))
