networkModels = ['lagrangeElig',
                 'lagrangeEligTf',
                 'lagrangeEligTfApprox',
                 'lagrangeEligTfApproxReg',
                 'lagrangeFromFirst',
                 'lagrangeFromFirst2',
                 'lagrangeFromFirstTest']

for net in networkModels:
    exec('from {0} import {0}'.format(net))
