import numpy as np
import time
import sys
import dijet
from dataclasses import asdict


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

	# basic acceptance-rejection method of generating a finite sample
	ranges = {
			'Q': [4, 10],
			'rapidity': [4.62, 9.20],
			'delta': [0.1, 0.2],
			'pT': [0, 15],
			'z': [0.1, 0.9],
			'phi_Dp': [0, 2*np.pi],
			'phi_kp': [0, 2*np.pi]
			}

	print('== parameter ranges:')
	for ivar, irange in ranges.items():
		print(ivar, irange)

	dj = dijet.DIJET(nreplica=1, constrained_moments=True)
	dj.load_params('replica_params_pp.csv')
	dj.set_params(3)

	data = []
	rng = np.random.default_rng(seed=int(time.time()))
	count = 0

	print('== starting generation of', sample_size, 'events')

	while len(data) < sample_size:

		ran_Q = rng.uniform(low=ranges['Q'][0], high=ranges['Q'][1])
		ran_x = np.exp(-rng.uniform(low=ranges['rapidity'][0], high=ranges['rapidity'][1]))
		ran_delta = rng.uniform(low=ranges['delta'][0], high=ranges['delta'][1])
		ran_pT = rng.uniform(low=ranges['pT'][0], high=ranges['pT'][1])
		ran_z = rng.uniform(low=ranges['z'][0], high=ranges['z'][1])
		ran_y = (ran_Q**2)/(ran_x*fixed_s)
		ran_phi_Dp = rng.uniform(low=ranges['phi_Dp'][0], high=ranges['phi_Dp'][1])
		ran_phi_kp = rng.uniform(low=ranges['phi_kp'][0], high=ranges['phi_kp'][1])

		# physical constraints ###############################
		if ran_y > 1: continue
		if ran_delta/ran_pT > 0.25: continue
		if ran_Q*np.sqrt(ran_z*(1-ran_z)) < 2: continue
		######################################################

		ran_kinematic_vars = dijet.Kinematics(s=fixed_s, Q= ran_Q, x= ran_x, delta= ran_delta, pT= ran_pT, z= ran_z, y= ran_y, phi_Dp=ran_phi_Dp, phi_kp=ran_phi_kp)

		ran_numerator = dj.angle_integrated_numerator(ran_kinematic_vars)
		ran_denominator = dj.angle_integrated_denominator(ran_kinematic_vars)

		ran_numerator = dj.numerator(ran_kinematic_vars)
		ran_denominator = dj.denominator(ran_kinematic_vars)
		ran_value = rng.uniform(low=0, high=0.1)

		if ran_value < np.abs(ran_numerator/ran_denominator):

			ran_corrs = [
				dj.numerator(ran_kinematic_vars, weight='cos(phi_kp)', diff='dy'),
				dj.numerator(ran_kinematic_vars, weight='cos(phi_Dp)', diff='dy'),
				dj.numerator(ran_kinematic_vars, weight='cos(phi_Dp)cos(phi_kp)', diff='dy'),
				dj.numerator(ran_kinematic_vars, weight='sin(phi_Dp)sin(phi_kp)', diff='dy')
			]


			kinematics_list = [ran_kinematic_vars.s, ran_kinematic_vars.Q]
			kinematics_list += [ran_kinematic_vars.x, ran_kinematic_vars.delta] 
			kinematics_list += [ran_kinematic_vars.pT, ran_kinematic_vars.z, ran_kinematic_vars.y]
			kinematics_list += [ran_kinematic_vars.phi_Dp, ran_kinematic_vars.phi_kp]

			# data.append(kinematics_list + [ran_denominator, ran_numerator] + ran_corrs)
			data.append(kinematics_list + [ran_denominator, ran_numerator])

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

	
















