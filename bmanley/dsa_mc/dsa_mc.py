import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import jv, kv
import sys
import dijet_utils as dutils


if __name__ == '__main__':

	if len(sys.argv) != 4:
		print("Usage: python dsa_mc.py <sample_size> <output_file> <root s (GeV)>")
		sys.exit(1)

	sample_size = int(sys.argv[1])
	outfile = sys.argv[2]
	root_s = float(sys.argv[3])
	fixed_s = root_s**2

	print(int(time.time()))
	# basic acceptance-rejection method of generating a finite sample
	dj = dutils.DijetXsec()
	rng = np.random.default_rng(seed=int(time.time()))

	ranges = {'Q': [np.sqrt(5), 20],
			  'rapidity': [4.62, 9.20],
			  'delta': [0.2, 2],
			  'pT': [1, 15],
			  'phi_kp': [0, 2*np.pi],
			  'phi_Dp': [0, 2*np.pi],
			  'log_xsec': [-1, 6],
			  'z': [0.1, 0.9]
			}

	data = []
	count = 0
	print('=== output file name:', outfile)
	print('=== root s (GeV) =', root_s)
	print('=== starting generation of', sample_size, 'events')

	while len(data) < sample_size:

		ran_Q = rng.uniform(low=ranges['Q'][0], high=ranges['Q'][1])
		ran_x = np.exp(-rng.uniform(low=ranges['rapidity'][0], high=ranges['rapidity'][1]))
		ran_delta = rng.uniform(low=ranges['delta'][0], high=ranges['delta'][1])
		ran_pT = rng.uniform(low=ranges['pT'][0], high=ranges['pT'][1])
		ran_z = rng.uniform(low=ranges['z'][0], high=ranges['z'][1])
		ran_y = (ran_Q**2)/(ran_x*fixed_s)
		ran_phi_kp = rng.uniform(low=ranges['phi_kp'][0], high=ranges['phi_kp'][1])
		ran_phi_Dp = rng.uniform(low=ranges['phi_Dp'][0], high=ranges['phi_Dp'][1])

		# physical constraints ###############################
		if ran_y > 1: continue
		if ran_delta/ran_pT > 0.25: continue
		# if ran_Q*np.sqrt(ran_z*(1-ran_z)) < 3: continue
		######################################################

		ran_kinematic_vars = {'s': fixed_s, 'Q': ran_Q, 'x': ran_x, 'delta': ran_delta, 'pT': ran_pT, 'z': ran_z, 'y': ran_y, 'phi_kp': ran_phi_kp, 'phi_Dp': ran_phi_Dp}
		ran_dsa = dj.get_xsec(ran_kinematic_vars, 'DSA')
		ran_unpolar = dj.get_xsec(ran_kinematic_vars, 'unpolarized')
		ran_xsec = np.exp(rng.uniform(low=ranges['log_xsec'][0], high=ranges['log_xsec'][1]))

		if ran_xsec < np.abs(ran_dsa):
			if count == 0:
				sys.stdout.write(f'\r[{int((count*100/sample_size))}%] done ({count}/{sample_size})')
				sys.stdout.flush()

			data.append(list(ran_kinematic_vars.values()) + [ran_dsa, ran_unpolar])
			count += 1
			if np.mod((count/sample_size)*100, 1) == 0:
				sys.stdout.write(f'\r[{(count*100/sample_size)}%] done ({count}/{sample_size})')
				sys.stdout.flush()

	print(int(time.time()))
	try:
		np.save(outfile, data)
		print('\n=== saved mc data in file', outfile)
	except:
		print('unable to save data')

	
















