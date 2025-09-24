import numpy as np
import dijet
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os


def compute_replica(irep, modes, djs, pT_values, roots, space, npoints, r0, denom):

	harmonics = ['1', 'cos(phi_kp)', 'cos(phi_Dp)', 'cos(phi_Dp)cos(phi_kp)', 'sin(phi_Dp)sin(phi_kp)']
	results = {}
	for mode in modes:
		djs[mode].set_params(irep + 1)
		rep = {}
		for harm in harmonics:
			rep[f'<{harm}>'] = djs[mode].get_integrated_xsec(
				pT_values, roots**2, space, weight=harm, points=npoints, kind='num', r0=r0
			) / denom
		rep['denom'] = denom
		results[mode] = rep
	return irep, results


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("outdir", help="Output directory")
	args = parser.parse_args()
	os.makedirs(args.outdir, exist_ok=True)

	######## OPTIONS ############
	IR_reg = ['gauss', 1.0]
	fit = 'pp'

	roots = 50
	pT_values = np.linspace(0.5, 10, 20)
	space = {
		'y': [0.05, 0.95],
		'z': [0.2, 0.5],
		'Q2': [16, 100],
		't': 0.1,
		'phi_Dp': [0, 2*np.pi],
		'phi_kp': [0, 2*np.pi]
	}
	r0 = 0.5
	npoints = 4
	nreps = 300

	############################

	print('Chosen options -------------------------------------')
	print('IR regulator:', IR_reg)
	print('Kinematic regulator Q0:', r0)
	print('fit parameters:', fit)
	print('root s:', roots)
	print('pT values:', pT_values)
	print('phase space:', space)
	print('integration points:', npoints)
	print('number of replicas:', nreps)
	print('----------------------------------------------------')


	djs = {}
	djs['p'] = dijet.DIJET(fit_type=fit, constrained_moments=True, IR_reg=IR_reg, nucleon='p')
	djs['n'] = dijet.DIJET(fit_type=fit, constrained_moments=True, IR_reg=IR_reg, nucleon='n')
	modes = ['p', 'n']

	replicas = {key: [] for key in modes}
	replicas['space'] = space
	replicas['pT values'] = pT_values
	replicas['roots'] = roots
	replicas['r0'] = r0

	r0str = str(r0).replace('.', 'p')
	repstr = str(nreps)
	xistr = str(IR_reg[1]).replace('.', 'p')
	tstr = '' if isinstance(space['t'], (list, tuple)) else 't'+str(space['t']).replace('.', 'p')
	zstr = '' if isinstance(space['z'], (list, tuple)) else 'z'+str(space['z']).replace('.', 'p')

	test = False
	if test: teststr = '_test'
	else: teststr = ''

	outfile = f'{args.outdir}/dsa_p{tstr}{zstr}_rs{roots}_Q0{r0str}_xi{xistr}_pn_{repstr}reps_largeNcq{teststr}.npy'

	print('')
	print('===> starting replica loop, outgoing file is', outfile)
	print('')

	denom = djs[modes[0]].get_integrated_xsec(
		pT_values, roots**2, space, points=npoints, kind='den', r0=r0
	)

	with ProcessPoolExecutor() as executor:
		futures = [
			executor.submit(compute_replica, irep, modes, djs,
							pT_values, roots, space, npoints, r0, denom)
			for irep in range(nreps)
		]
		for future in as_completed(futures):
			irep, results = future.result()
			for mode in modes:
				replicas[mode].append(results[mode])

			# periodic save to avoid data loss
			if (irep + 1) % 10 == 0:
				np.save(outfile, replicas, allow_pickle=True)

	# final save
	np.save(outfile, replicas, allow_pickle=True)
	print('')
	print('saved to', outfile)



