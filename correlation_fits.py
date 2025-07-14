import numpy as np
import dijet

if __name__ == '__main__':
	
	djs = {}
	djs['pp'] = dijet.DIJET(fit_type='pp', constrained_moments=False)
	djs['dis'] = dijet.DIJET(fit_type='dis', constrained_moments=False)
	fits = ['dis', 'pp']

	roots = 100
	pT_values = np.linspace(0.5, 10, 20)
	space = {
		'y' : [0.05, 0.95],
		'z' : [0.2, 0.5],
		'Q2': [16, 100],
		't' : [0.05, 0.1],
		'phi_Dp': [0, 2*np.pi],
		'phi_kp': [0, 2*np.pi]
	}

	replicas = {}
	for key in fits: replicas[key] = []
	replicas['space'] = space
	replicas['pT values'] = pT_values
	replicas['roots'] = roots

	npoints = 8
	nreps = 100

	denom = djs[fits[0]].get_integrated_xsec(pT_values, roots**2, space, points=npoints, kind='den', r0=2.0)
	for irep in range(1, nreps):
		for ifit, fit in enumerate(fits):
			djs[fit].set_params(irep)
		
			rep = {}
			rep['<1>'] = djs[fit].get_integrated_xsec(pT_values, roots**2, space, weight='1', points=npoints, kind='num', r0=2.0)/denom
			rep['<cos(phi_Dp)>'] = djs[fit].get_integrated_xsec(pT_values, roots**2, space, weight='cos(phi_Dp)', points=npoints, kind='num', r0=2.0)/denom
			rep['<cos(phi_Dp)cos(phi_kp)>'] = djs[fit].get_integrated_xsec(pT_values, roots**2, space, weight='cos(phi_Dp)cos(phi_kp)', points=npoints, kind='num', r0=2.0)/denom
			rep['<sin(phi_Dp)sin(phi_kp)>'] = djs[fit].get_integrated_xsec(pT_values, roots**2, space, weight='sin(phi_Dp)sin(phi_kp)', points=npoints, kind='num', r0=2.0)/denom
			rep['denom'] = denom
		
			replicas[fit].append(rep)

		np.save('data/dsa_fit_comparison.npy', replicas, allow_pickle=True)


