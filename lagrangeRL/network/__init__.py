networkModels = ['lagrangeElig',
                 'lagrangeEligTf',
                 'lagrangeEligTfApprox',
                 'lagrangeEligTfApproxReg',
                 'lagrangeFromFirst',
                 'lagrangeFromFirst2',
                 'lagrangeFromFirstTest',
                 'lagrangeTfOptimized']

for net in networkModels:
    exec('from {0} import {0}'.format(net))
