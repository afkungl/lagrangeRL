experiments = ['expApproxLagrange',
               'expApproxLagrangeDelay',
               'expApproxLagrangeVerifyBp',
               'expExactLagrange']

for experiment in experiments:
    exec('from .{0} import {0}'.format(experiment))
