import numpy as np
import time
import sys
import dijet
from dataclasses import asdict


def get_random_point(rng, ranges):
	ran_point = {}
	for var, irange in ranges.items(): 
		ran_point[var] = rng.uniform(low=irange[0], high=irange[1])
	return ran_point



if __name__ == '__main__':

	if len(sys.argv) != 4:
		print("Usage: python dsa_mc.py <sample_size> <output_file> <root s (GeV)>")
		sys.exit(1)

	sample_size = int(sys.argv[1])
	outfile = sys.argv[2]
	root_s = float(sys.argv[3])
	fixed_s = root_s**2

	print('=== DSA for dijets MC generator ===')
	print('== output file name:', outfile)
	print('== root s =', root_s, 'GeV')

	ranges = {
			'Q2': [0, 100],
			'y': [0.05, 0.95],   
			'|t|': [0.01, 0.07],   # |t| = \Delta**2
			'pT': [0, 10],
			# 'z': [0.2, 0.8]
			'z': [0.2, 0.5]
			}

	print('== parameter ranges:')
	for ivar, irange in ranges.items():
		print(ivar, irange)

	print('== starting generation of', sample_size, 'points')


	dj = dijet.DIJET(nreplica=1, constrained_moments=True)
	dj.load_params('replica_params_pp.csv')
	dj.set_params(3)

	data = []
	rng = np.random.default_rng(seed=int(time.time()))
	count = 0
	while len(data) < sample_size:

		rp = get_random_point(rng, ranges)

		# rp['pT'] = 1.0

		# physical constraints ###############################
		rp['x'] = rp['Q2']/(fixed_s*rp['y'])
		rp['delta'] = np.sqrt(rp['|t|'])

		if rp['x'] > 0.01: continue					  			# small x limit
		if rp['delta']/rp['pT'] > 0.25: continue	  			# correlation limit
		if rp['Q2']*rp['z']*(1-rp['z']) < (2**2): continue 		# no oscillations
		######################################################

		ran_kins = dijet.Kinematics(
			s=fixed_s, Q=np.sqrt(rp['Q2']), x=rp['x'], delta=rp['delta'], pT=rp['pT'], z=rp['z'], y=rp['y']
		)

		# compute numerator and denominator of asymmetry (and harmonics)
		ran_funcs = [
			dj.angle_integrated_numerator(ran_kins, diff='dy'),
			dj.angle_integrated_numerator(ran_kins, weight='cos(phi_kp)', diff='dy'),
			dj.angle_integrated_numerator(ran_kins, weight='cos(phi_Dp)', diff='dy'),
			dj.angle_integrated_numerator(ran_kins, weight='cos(phi_Dp)cos(phi_kp)', diff='dy'),
			dj.angle_integrated_numerator(ran_kins, weight='sin(phi_Dp)sin(phi_kp)', diff='dy'),
			dj.angle_integrated_denominator(ran_kins, diff='dy')
		]

		ran_funcs.append(ran_funcs[0]/ran_funcs[5])
		ran_funcs.append(ran_funcs[1]/ran_funcs[5])
		ran_funcs.append(ran_funcs[2]/ran_funcs[5])
		ran_funcs.append(ran_funcs[3]/ran_funcs[5])
		ran_funcs.append(ran_funcs[4]/ran_funcs[5])

		ran_var_list = []
		for key, var in asdict(ran_kins).items():
			if key in ['s', 'Q', 'x', 'delta', 'pT', 'z', 'y']:
				ran_var_list.append(var)

		data.append(ran_var_list + ran_funcs)

		# print progress
		if count == 0:
			sys.stdout.write(f'\r[{int((count*100/sample_size))}%] done ({count}/{sample_size})')
			sys.stdout.flush()
		count += 1
		if np.mod((count/sample_size)*100, 1) == 0:
			sys.stdout.write(f'\r[{(count*100/sample_size)}%] done ({count}/{sample_size})')
			sys.stdout.flush()

	try:
		np.save(outfile, data)
		print('\n=== saved mc data in file', outfile)
	except:
		print('unable to save data')

	
















