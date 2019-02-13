experiments = ['expApproxLagrange',
               'expApproxLagrangeDelay',
               'expApproxLagrangeVerifyBp']

for experiment in experiments:
    exec('from .{0} import {0}'.format(experiment))
