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

	if len(sys.argv) != 3:
		print("Usage: python dsa_mc.py <sample_size> <output_file>")
		sys.exit(1)

	sample_size = int(sys.argv[1])
	outfile = sys.argv[2]

	print('=== DSA for dijets MC generator (TEST) ===')
	print('== output file name:', outfile)

	ranges = {
			'x': [-1, 1],
			'y': [-1, 1],   
			'z': [-1, 1]
			}

	print('== parameter ranges:')
	for ivar, irange in ranges.items():
		print(ivar, irange)

	print('== starting generation of', sample_size, 'points')


	# dj = dijet.DIJET(nreplica=1, constrained_moments=True)
	# dj.load_params('replica_params_pp.csv')
	# dj.set_params(3)

	data = []
	rng = np.random.default_rng(seed=int(time.time()))
	count = 0
	while len(data) < sample_size:

		rp = get_random_point(rng, ranges)

		ran_func = np.exp(-(rp['x']**2) - (rp['y']**2) - (rp['z']**2)) 

		data.append(list(rp.values()) + [ran_func])

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


	neff = 0
	rsum = 0
	epsilon = 10**(-2)
	for dat in data:
		if dat[2] < epsilon and dat[2] > -epsilon:
			rsum += dat[3]
			neff += 1

	# rsum *= 4*(2/(2*epsilon))*(1/sample_size)
	rsum *= 4*(1/neff)

	# print(neff/sample_size)
	# print((2*epsilon) / 2)

	print(rsum)



	
















