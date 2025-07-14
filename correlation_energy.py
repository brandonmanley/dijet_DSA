import numpy as np
import sys
import dijet
import random


if __name__ == '__main__':
	dj = dijet.DIJET(fit_type='pp', constrained_moments=True)

	space = {
		'y' : [0.05, 0.95],
		'z' : [0.2, 0.5],
		'Q2': [16, 100],
		# 't' : [0.05, 0.1],
		't': 0.1,
		'phi_Dp': [0, 2*np.pi],
		'phi_kp': [0, 2*np.pi]
	}
	pT_values = [3.0]
	roots_values = np.linspace(45, 140, 20)

	replicas = {}
	replicas['space'] = space
	replicas['pT'] = pT_values[0]
	replicas['roots values'] = roots_values
	replicas['reps'] = []

	npoints = 6
	nreps = 100

	replicas['denom'] = []
	for roots in roots_values:
		replicas['denom'].append(dj.get_integrated_xsec(pT_values, roots**2, space, points=npoints, kind='den', r0=2.0)[0])


	correlations = ['1', 'cos(phi_Dp)', 'cos(phi_Dp)cos(phi_kp)', 'sin(phi_Dp)sin(phi_kp)']
	for irep in range(1, nreps):
		dj.set_params(irep)

		rep = {corr: [] for corr in correlations}
		for corr in correlations:
			for roots in roots_values:
				rep[corr].append(dj.get_integrated_xsec(pT_values, roots**2, space, weight=corr, points=npoints, kind='num', r0=2.0)[0])
			rep[corr] = np.array(rep[corr])/np.array(replicas['denom'])

		replicas['reps'].append(rep)
		
		# rep = {}
		# rep['<1>'] = dj.get_integrated_xsec(pT_values, roots**2, space, weight='1', points=npoints, kind='num', r0=2.0)[0]/denom
		# rep['<cos(phi_Dp)>'] = dj.get_integrated_xsec(pT_values, roots**2, space, weight='cos(phi_Dp)', points=npoints, kind='num', r0=2.0)/denom
		# rep['<cos(phi_Dp)cos(phi_kp)>'] = dj.get_integrated_xsec(pT_values, roots**2, space, weight='cos(phi_Dp)cos(phi_kp)', points=npoints, kind='num', r0=2.0)/denom
		# rep['<sin(phi_Dp)sin(phi_kp)>'] = dj.get_integrated_xsec(pT_values, roots**2, space, weight='sin(phi_Dp)sin(phi_kp)', points=npoints, kind='num', r0=2.0)/denom
		# rep['denom'] = denom

		# replicas[roots].append(rep)
		
		np.save('data/dsa_corr_oam3_range10_roots.npy', replicas, allow_pickle=True)


