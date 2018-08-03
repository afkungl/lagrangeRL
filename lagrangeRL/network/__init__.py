networkModels = ['lagrangeElig',
				 'lagrangeEligTf']

for net in networkModels:
	exec('from {0} import {0}'.format(net))