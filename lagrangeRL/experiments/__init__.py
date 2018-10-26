experiments = ['trialBasedClassification',
               'trialBasedClassificationSmoothed',
               'timeContinuousClassification',
               'timeContinuousClassificationSmoothed',
               'timeContinuousClassificationSmOu',
               'timeContinuousClassificationSmOuApproxLagrange',
               'timeContinuousClassificationDelayedReward',
               'timeContinuousClassificationDelayedRewardSmoothed',
               'timeContinuousClassificationVerifyBackprop',
               'timeContOuDelayed',
               'slimExperiment',
               'slimExperimentDelayed',
               'slimExperimentVerifyBp',
               'slimExperimentReg',
               'slimExperimentRegDelay',
               'slimExperimentRegVerifyBp',
               'experimentWithFirst',
               'experimentWithFirst2',
               'experimentWithFirstTest',
               'experimentWithFirst2FullRew',
               'expOptimizedLagrange']

for experiment in experiments:
    exec('from .{0} import {0}'.format(experiment))
