networkModels = ['lagrangeElig',
                 'lagrangeEligTf',
                 'lagrangeEligTfApprox',
                 'lagrangeEligTfApproxReg',
                 'lagrangeFromFirst',
                 'lagrangeFromFirst2',
                 'lagrangeFromFirstTest',
                 'lagrangeTfOptimized',
                 'lagrangeTfOptimized2',
                 'lagrangeTfDirect',
                 'lagrangeTfNodePert']

for net in networkModels:
    exec('from {0} import {0}'.format(net))
