import json 
import numpy as np
import sys
import time
import pandas as pd


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print("Usage: python gen_ics.py <ic_type>")
		sys.exit(1)

	ic_type = sys.argv[1]
	assert ic_type == 'ones' or ic_type == 'random' or ic_type == 'random_fit', 'Error: only ones, random, or random_fit (fitted helicity parameters + random moments) IC types are supported'

	rng = np.random.default_rng(seed=int(time.time()))

	amp_ranges = {}
	amp_ranges['Qu'] = [-10, 10]
	amp_ranges['Qd'] = [-10, 10]
	amp_ranges['Qs'] = [-10, 10]
	amp_ranges['GT'] = [-10, 10]
	amp_ranges['G2'] = [-10, 10]
	# amp_ranges['Qt'] = [-10, 10]
	amp_ranges['I3u'] = [-10, 10]
	amp_ranges['I3d'] = [-10, 10]
	amp_ranges['I3s'] = [-10, 10]
	amp_ranges['I3T'] = [-10, 10]
	amp_ranges['I4'] = [-10, 10]
	amp_ranges['I5'] = [-10, 10]
	# amp_ranges['It'] = [-10, 10]

	if ic_type == 'random_fit':
		header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['a', 'b', 'c']]
		# df = pd.read_csv('mc_data/replica_params.csv', names=headers, header=0)
		fit_df = pd.read_csv('../dsa_mc/dipoles/replica_params.csv')
		fit_df = fit_df.dropna(axis=1, how='all')
		fit_df.columns = header

	ic_dict = {}
	for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
		if ic_type == 'ones': ic_dict[amp] = [0,0,1]
		elif ic_type == 'random': ic_dict[amp] = [rng.uniform(low=amp_ranges[amp][0], high=amp_ranges[amp][1]) for i in range(3)]

		elif ic_type == 'random_fit':
			if 'I' in amp or amp == 'Qt': ic_dict[amp] = [rng.uniform(low=amp_ranges[amp][0], high=amp_ranges[amp][1]) for i in range(3)]
			else: 
				nrep = 151
				ic_dict[amp] = [fit_df.loc[fit_df.index[nrep], f'{amp}{ib}'] for ib in ['a', 'b', 'c']]

	outfile = f"../dsa_mc/dipoles/mc_ICs_{ic_type}.json"
	with open(outfile, "w") as file: json.dump(ic_dict, file)
	print('Generated initial condition file', outfile)







