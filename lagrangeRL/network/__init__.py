networkModels = ['lagrangeElig',
                 'lagrangeEligTf',
                 'lagrangeEligTfApprox']

for net in networkModels:
    exec('from {0} import {0}'.format(net))
