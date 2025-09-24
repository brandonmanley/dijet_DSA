import numpy as np
import dijet
import concurrent.futures


# find moment params that satisfy \int^0.1_10^-5 (OAM) dx < const. 
def get_param_guess(rep_range=10.0):
	ps = {}
	for amp in ['I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
		ps[amp] = {}
		for basis in ['eta', 's10', '1']:
			ps[amp][basis] = np.random.uniform(-rep_range, rep_range)
	return ps


def find_params(irep, Q2=10, fit='pp', oam_limit=3.0, rep_range=10.0):
	dj = dijet.DIJET(fit_type=fit)  
	while True:
		guess = get_param_guess(rep_range)
		dj.set_temp_params(guess, irep)
		reasonable = True
		for xmax in [1e-4, 1e-3, 1e-2, 1e-1]:
			val = dj.get_IntegratedPDF('oam', Q2, xmax=xmax)
			if np.abs(val) > oam_limit:
				reasonable = False
				break
		if reasonable:
			return irep, guess


def save_params(params, filename):

	# reformat and save parameters
	pars = {'pp': []}
	header = ['nrep'] + [f'{amp}{basis}' for amp in ['I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5'] for basis in ['eta', 's10', '1']]
	header_str = ",".join(header)

	for irep in range(len(params['pp'])):
		for fit in ['pp']:
			params = []
			for amp in ['I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
				for basis in ['eta', 's10', '1']:
					params.append(mparams[fit][irep][amp][basis])
			pars[fit].append([irep+1] + params)

	np.savetxt(filename, pars['pp'], delimiter=",", header=header_str, comments='')


if __name__ == '__main__':

	Q2 = 10
	n_replicas = 300  # or up to 397
	rep_range = 10
	oam_limit = 3
	fit = 'pp'

	print(f'##### starting {n_replicas} replicas')
	print(f'##### replica range: [-{rep_range}, {rep_range}]')
	print(f'##### limit on OAM moment: {oam_limit}')

	results = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		futures = [executor.submit(find_params, irep, Q2, fit, oam_limit, rep_range) for irep in range(1, n_replicas + 1)]
		for future in concurrent.futures.as_completed(futures):
			irep, guess = future.result()
			print(f'Finished replica {irep} ({len(results)}/{n_replicas} done)')
			results.append((irep, guess))

	# organize into mparams dict
	mparams = {fit: [guess for _, guess in sorted(results)]}

	save_params(mparams, 'dipoles/moment_params_oam3_range10_300reps_largeNcq.csv')




