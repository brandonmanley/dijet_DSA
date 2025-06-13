import numpy as np
import sys
sys.path.append('/Users/brandonmanley/Desktop/PhD/oam_pheno/dijet_dsa')
import dsa_mc.dijet as dijet
import concurrent.futures


# find moment params that satisfy \int^0.1_10^-5 (OAM) dx < 0.5
def get_param_guess():
	ps = {}
	for amp in ['I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
		ps[amp] = {}
		for basis in ['eta', 's10', '1']:
			ps[amp][basis] = np.random.uniform(-10, 10)
	return ps


def find_params(irep, Q2=10, fit='pp'):
	dj = dijet.DIJET(fit_type=fit)  
	while True:
		guess = get_param_guess()
		dj.set_temp_params(guess, irep)
		reasonable = True
		for xmax in [1e-4, 1e-3, 1e-2, 1e-1]:
			val = dj.get_IntegratedPDF('oam', Q2, xmax=xmax)
			if np.abs(val) > 1.0:
				reasonable = False
				break
		if reasonable:
			return irep, guess


if __name__ == '__main__':

	Q2 = 10
	n_replicas = 300  # or up to 397
	fit = 'pp'

	results = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		futures = [executor.submit(find_params, irep, Q2, fit) for irep in range(1, n_replicas + 1)]
		for future in concurrent.futures.as_completed(futures):
			irep, guess = future.result()
			print(f'Finished replica {irep}')
			results.append((irep, guess))

	# organize into mparams dict
	mparams = {fit: [guess for _, guess in sorted(results)]}

	print(mparams)
