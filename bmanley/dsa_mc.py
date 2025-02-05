import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import jv, kv
import sys


def double_bessel(kvar, indices, amp, lamIR = 0.3, IR_reg = [None, 0], deta=0.05):

	ia, ib, ic, id = indices
	pf = kvar['pT']/lamIR
	Qf = (kvar['Q']*np.sqrt(kvar['z'] * (1 - kvar['z'])))/lamIR
	prefactor = (pf**ic)*(Qf**id)

	# polarized dipoles
	if amp != 'N':
		pdipole_dfs = polarized_dipoles
		bas = np.sqrt(3/(2*np.pi))
		prefactor *= (1/(2*bas))
		target_eta = round((bas/deta) * np.log(kvar['s']/(lamIR**2)))*deta
		dipole = pdipole_dfs[np.isclose(pdipole_dfs['eta'], target_eta, atol=deta * 0.5)][['s10', amp]]
		s10 = dipole['s10'].to_numpy()
		amp_values = dipole[amp].to_numpy()
		u = -s10*(1/(2*bas))
		size = deta

		# IR region
		if IR_reg[0] == 'gauss':	amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
		elif IR_reg[0] == 'skin':	amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(lamIR*IR_reg[2])) - 1)))
		
		if IR_reg[0] == 'cut':	amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
		else:	amp_values = np.where(s10 < 0, 0, amp_values)

	# unpolarized dipole
	else:
		ndipole_df = unpolarized_dipole
		closest_y = ndipole_df['y'][np.isclose(ndipole_df['y'], np.log(1/kvar['x']), atol=0.005)].iloc[0]
		dipole = ndipole_df[np.isclose(ndipole_df['y'], closest_y, atol=0.005)]
		# print(dipole)
		u = dipole['ln(r)'].to_numpy()
		amp_values = 15*2.57*(dipole['N[ln(r)]'].to_numpy()) # 15*2.57 (GeV^{-2}) is value of \int d^2 b from fit in 2407.10581  
		amp_values = np.where(u > np.log(1/lamIR), 0, amp_values)

		# print(amp_values)
		size = 0.01  # from evolution code

	exp_term = np.exp(u*(2 + ic + id))
	jv_term = jv(ia, pf*np.exp(u))
	kv_term = kv(ib, Qf*np.exp(u))
	total_sum = size*np.sum(exp_term * jv_term * kv_term * amp_values)

	return prefactor * total_sum




def get_coeff(flavor, kvar):

	if 'TT' in flavor: prefactor = -(alpha_em*Zfsq*(Nc**2)*(kvar['Q']**2)*kvar['z']*(1-kvar['z']))/(2*(np.pi**4)*kvar['s'])
	elif 'LT' in flavor: prefactor = -(alpha_em*Zfsq*(Nc**2)*(kvar['Q']**2)*((kvar['z']*(1-kvar['z']))**1.5))/(np.sqrt(2)*(np.pi**4)*kvar['s'])
	elif 'TT_unpolar' in flavor: prefactor = (4*alpha_em*Zfsq*(Nc**2)*(kvar['Q']**2)*(kvar['z']**2)*((1-kvar['z'])**2)*((kvar['z'])**2 + (1-kvar['z'])**2))/(2*(np.pi**4))
	elif 'LL_unpolar' in flavor: prefactor = (8*alpha_em*Zfsq*(Nc**2)*(kvar['Q']**2)*(kvar['z']**3)*((1-kvar['z'])**3))/(2*(np.pi**4))
	elif 'TmT_unpolar' in flavor: prefactor = (4*alpha_em*Zfsq*(Nc**2)*(kvar['Q']**2)*(kvar['z']**2)*((1-kvar['z'])**2)*((1- 2*kvar['z'])**2 - 1))/(2*(np.pi**4))

	if flavor == 'A_TT':
		Q_11 = double_bessel(kvar, [1,1,0,0], 'Q')
		G2_11 = double_bessel(kvar, [1,1,0,0], 'G2')
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		return prefactor*(((1 - 2*kvar['z'])**2)*Q_11 + 2*(kvar['z']**2 + (1-kvar['z'])**2)*G2_11)*N_11


	elif flavor == 'B_TT': 
		Q_11 = double_bessel(kvar, [1,1,0,0], 'Q')
		Q_21_10 = double_bessel(kvar, [2,1,1,0], 'Q')
		G2_11 = double_bessel(kvar, [1,1,0,0], 'G2')
		G2_21_10 = double_bessel(kvar, [2,1,1,0], 'G2')
		I3_11 = double_bessel(kvar, [1,1,0,0], 'I3')
		I3_01_10 = double_bessel(kvar, [0,1,1,0], 'I3')
		I4_11 = double_bessel(kvar, [1,1,0,0], 'I4')
		I4_21_10 = double_bessel(kvar, [2,1,1,0], 'I4')
		I4_10_01 = double_bessel(kvar, [1,0,0,1], 'I4')
		I5_11 = double_bessel(kvar, [1,1,0,0], 'I5')
		I5_10_01 = double_bessel(kvar, [1,0,0,1], 'I5')
		I5_01_10 = double_bessel(kvar, [0,1,1,0], 'I5')
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		N_01_10 = double_bessel(kvar, [0,1,1,0], 'N')

		b_TT = 0.5*((1-2*kvar['z'])**2)*N_01_10*Q_11
		b_TT += (kvar['z']**2 + (1-kvar['z'])**2)*N_01_10*G2_11 + N_11*(0.5*Q_11 + I3_11 - I3_01_10)
		b_TT += (kvar['z']**2 + (1-kvar['z'])**2)*N_11*(Q_21_10 + 2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10)
		return prefactor*(1-(2*kvar['z']))*b_TT


	elif flavor == 'A_LT': 
		Q_11 = double_bessel(kvar, [1,1,0,0], 'Q')
		Q_00 = double_bessel(kvar, [0,0,0,0], 'Q')
		G2_11 = double_bessel(kvar, [1,1,0,0], 'G2')
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		N_00 = double_bessel(kvar, [0,0,0,0], 'N')
		return prefactor*(1- 2*kvar['z'])*(N_00*(2*G2_11 - Q_11) - N_11*Q_00)


	elif flavor == 'B_LT': 
		Q_00 = double_bessel(kvar, [0,0,0,0], 'Q')
		Q_11 = double_bessel(kvar, [1,1,0,0], 'Q')
		Q_01_10 = double_bessel(kvar, [0,1,1,0], 'Q')
		Q_10_10 = double_bessel(kvar, [1,0,1,0], 'Q')
		G2_11 = double_bessel(kvar, [1,1,0,0], 'G2')
		G2_01_10 = double_bessel(kvar, [0,1,1,0], 'G2')
		I3_11 = double_bessel(kvar, [1,1,0,0], 'I3')
		I3_01_10 = double_bessel(kvar, [0,1,1,0], 'I3')
		I3_10_10 = double_bessel(kvar, [1,0,1,0], 'I3')
		I4_11 = double_bessel(kvar, [1,1,0,0], 'I4')
		I4_21_10 = double_bessel(kvar, [2,1,1,0], 'I4')
		I4_10_01 = double_bessel(kvar, [1,0,0,1], 'I4')
		I5_11 = double_bessel(kvar, [1,1,0,0], 'I5')
		I5_10_01 = double_bessel(kvar, [1,0,0,1], 'I5')
		I5_01_10 = double_bessel(kvar, [0,1,1,0], 'I5')
		N_00 = double_bessel(kvar, [0,0,0,0], 'N')
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		N_10_10 = double_bessel(kvar, [1,0,1,0], 'N')
		N_01_10 = double_bessel(kvar, [0,1,1,0], 'N')

		b_LT = N_00*(I3_11 - I3_01_10)
		b_LT += N_11*I3_10_10 
		b_LT += (kvar['z']**2 + (1-kvar['z'])**2)*N_00*(Q_01_10 - Q_11)
		b_LT -= (kvar['z']**2 + (1-kvar['z'])**2)*N_11*Q_10_10 
		b_LT -= 0.5*((1- 2*kvar['z'])**2)*N_10_10*Q_11
		b_LT -= 0.5*((1- 2*kvar['z'])**2)*(N_11 - N_01_10)*Q_00
		b_LT += ((1- 2*kvar['z'])**2)*N_00*(3*G2_11 - 2*G2_01_10 + I4_21_10 + I4_10_01 - I5_11 + I5_01_10 + I5_10_01)
		b_LT += ((1- 2*kvar['z'])**2)*N_10_10*G2_11

		return prefactor*b_LT


	elif flavor == 'C_LT': 
		Q_00 = double_bessel(kvar, [0,0,0,0], 'Q')
		Q_11 = double_bessel(kvar, [1,1,0,0], 'Q')
		G2_11 = double_bessel(kvar, [1,1,0,0], 'G2')
		G2_01_10 = double_bessel(kvar, [0,1,1,0], 'G2')
		G2_10_10 = double_bessel(kvar, [1,0,1,0], 'G2')
		I3_11 = double_bessel(kvar, [1,1,0,0], 'I3')
		I4_11 = double_bessel(kvar, [1,1,0,0], 'I4')
		I4_10_01 = double_bessel(kvar, [1,0,0,1], 'I4')
		I4_10_10 = double_bessel(kvar, [1,0,1,0], 'I4')
		I4_01_10 = double_bessel(kvar, [0,1,1,0], 'I4')
		I4_11_20 = double_bessel(kvar, [1,1,2,0], 'I4')
		I4_00_11 = double_bessel(kvar, [0,0,1,1], 'I4')
		I4_11_11 = double_bessel(kvar, [1,1,1,1], 'I4')
		I4_00_20 = double_bessel(kvar, [0,0,2,0], 'I4')
		I5_11 = double_bessel(kvar, [1,1,0,0], 'I5')
		I5_11_20 = double_bessel(kvar, [1,1,2,0], 'I5')
		I5_10_01 = double_bessel(kvar, [1,0,0,1], 'I5')
		I5_00_11 = double_bessel(kvar, [0,0,1,1], 'I5')
		I5_11_11 = double_bessel(kvar, [1,1,1,1], 'I5')
		I5_10_10 = double_bessel(kvar, [1,0,1,0], 'I5')
		I5_00_20 = double_bessel(kvar, [0,0,2,0], 'I5')
		N_00 = double_bessel(kvar, [0,0,0,0], 'N')
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')

		c_LT = (kvar['z']**2 + (1-kvar['z'])**2)*N_11*Q_11
		c_LT -= N_00*I3_11
		c_LT += 0.5*((1- 2*kvar['z'])**2)*N_11*Q_00
		c_LT += ((1- 2*kvar['z'])**2)*N_00*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
		c_LT += ((1- 2*kvar['z'])**2)*N_11*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)

		return prefactor*c_LT


	elif flavor == 'A_TT_unpolar':
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		return prefactor*(N_11**2)


	elif flavor == 'B_TT_unpolar':
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		N_21_10 = double_bessel(kvar, [2,1,1,0], 'N')
		return prefactor*2*(kvar['z']-0.5)*(2*(N_11**2) - N_11*N_21_10)


	elif flavor == 'A_LL_unpolar':
		N_00 = double_bessel(kvar, [0,0,0,0], 'N')
		return prefactor*(N_00**2)


	elif flavor == 'B_LL_unpolar':
		N_00 = double_bessel(kvar, [0,0,0,0], 'N')
		N_10_10 = double_bessel(kvar, [1,0,1,0], 'N')
		return -prefactor*2*(kvar['z']-0.5)*N_00*N_10_10


	elif flavor == 'A_TmT_unpolar':
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		return prefactor*(N_11**2)


	elif flavor == 'B_TmT_unpolar':
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		N_01_10 = double_bessel(kvar, [0,1,1,0], 'N')
		return prefactor*(kvar['z']-0.5)*(-2*(N_11**2) + N_11*N_01_10)


	elif flavor == 'C_TmT_unpolar':
		N_11 = double_bessel(kvar, [1,1,0,0], 'N')
		return -2*prefactor*(kvar['z']-0.5)*(N_11**2)


	else:
		print('requested coefficient', flavor, 'does not exist')




def get_DSA(polarized_dipoles, unpolarized_dipole, kvar):

	prefactor = alpha_em/(4*(np.pi**2)*(kvar['Q']**2))

	tt_term = (2-kvar['y'])*(get_coeff('A_TT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*get_coeff('B_TT', kvar))
	lt_term = np.sqrt(2-2*kvar['y'])*(np.cos(kvar['phi_kp'])*get_coeff('A_LT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(kvar['phi_kp'])*get_coeff('B_LT', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(kvar['phi_kp'])*get_coeff('C_LT', kvar))

	return prefactor*(tt_term + lt_term)


def get_unpolar_xsec(polarized_dipoles, unpolarized_dipole, kvar):

	prefactor = alpha_em/(4*(np.pi**2)*(kvar['Q']**2))

	tt_term = (1 + (1-kvar['y'])**2)*(get_coeff('A_TT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*get_coeff('B_TT_unpolar', kvar))
	tmt_term = -2*(1-kvar['y'])*(np.cos(2*kvar['phi_kp'])*get_coeff('A_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(2*kvar['phi_kp'])*get_coeff('B_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(2*kvar['phi_kp'])*get_coeff('C_TmT_unpolar', kvar))
	ll_term = 4*(1-kvar['y'])*(get_coeff('A_LL_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*get_coeff('B_LL_unpolar', kvar))

	return prefactor*(tt_term + tmt_term + ll_term)



def to_array(data_files, tag, deta=0.05):
	dipoles = []

	if 'nocutoff' in tag:
		nsteps = len(data_files[0])
		s10_range = np.arange(-nsteps + 1, nsteps)
		eta_range = np.arange(nsteps)
	
		for is10 in s10_range:
			for ieta in eta_range:
				if ieta > is10 and ieta < is10+nsteps: 
					dipoles.append([is10*deta, ieta*deta] + [idipole[is10-ieta+nsteps, ieta] for idipole in data_files])
				else: 
					# fixme: does not account for different initial conditions than 1
					dipoles.append([is10*deta, ieta*deta] + [1 for idipole in data_files])       
		return dipoles

	else:
		for is10, s10_row in enumerate(data_files[0]):
			for ieta, eta in enumerate(s10_row):
				dipole_values = [ifile[is10, ieta] for ifile in data_files]
				dipoles.append([(is10+1)*deta, (ieta+1)*deta] + dipole_values)
		return dipoles




def load_dipoles(kind, deta):
	
	# load unpolarized dipole data
	if kind == 'unpolarized':
		unpolar_input_file = '/Users/brandonmanley/Desktop/PhD/rcbkdipole/build/bin/dipole_data/unpolarized_dipole_ymin4.610000_ymax9.210000.dat'
		print('=== loading unpolarized dipole data')
		return pd.read_csv(unpolar_input_file, sep=r'\s+', header=None, names=['y', 'ln(r)', 'N[ln(r)]'])

	# load polarized dipole data
	else: 
		dstr = str(deta)[2:]
		# tag = '_nocutoff_rc'
		tag = '_rc'
		print('=== loading polarized dipole data with tag:', tag)
		
		polar_indir = '/Users/brandonmanley/Desktop/PhD/moment_evolution/evolved_dipoles/largeNc&Nf/d'+dstr+'_ones/'
		polar_input_files = [polar_indir+'d'+dstr+'_NcNf3_ones_'+ia+tag+'.dat' for ia in ['Q', 'G2', 'I3', 'I4', 'I5']]
		polar_data_files = [np.loadtxt(ifile) for ifile in polar_input_files]
		return pd.DataFrame(to_array(polar_data_files, tag), columns = ['s10', 'eta', 'Q', 'G2', 'I3', 'I4', 'I5'])




if __name__ == '__main__':

	# constants
	alpha_em = 1/137.0
	Zfsq = 2/3.0
	Nc = 3.0

	if len(sys.argv) != 4:
		print("Usage: python dsa_mc.py <sample_size> <output_file> <root s (GeV)>")
		sys.exit(1)

	sample_size = int(sys.argv[1])
	outfile = sys.argv[2]
	root_s = float(sys.argv[3])

	# basic acceptance-rejection method of generating a finite sample
	rng = np.random.default_rng(seed=int(time.time()))

	ranges = {'Q': [np.sqrt(5), 10],
			  'rapidity': [4.61, 9.21],
			  'delta': [0.2, 1],
			  'pT': [1, 10],
			  'phi_kp': [0, 2*np.pi],
			  'phi_Dp': [0, 2*np.pi],
			  'log_xsec': [-50, -5],
			  'z': [0.1, 0.9]
			}

	deta = 0.05
	polarized_dipoles = load_dipoles('polarized', deta)
	unpolarized_dipole = load_dipoles('unpolarized', deta)

	data = []
	count = 0
	print('=== starting generation of', sample_size, 'events')
	while len(data) < sample_size:

		# ran_s = rng.uniform(low=ranges['s'][0], high=ranges['s'][1])
		ran_s = root_s**2
		ran_Q = rng.uniform(low=ranges['Q'][0], high=ranges['Q'][1])
		ran_x = np.exp(-rng.uniform(low=ranges['rapidity'][0], high=ranges['rapidity'][1]))
		# if np.log(1/ran_x) < 4.61 or np.log(1/ran_x) > 9.21: continue

		ran_delta = rng.uniform(low=ranges['delta'][0], high=ranges['delta'][1])
		ran_pT = rng.uniform(low=ranges['pT'][0], high=ranges['pT'][1])
		ran_z = rng.uniform(low=ranges['z'][0], high=ranges['z'][1])
		ran_y = (ran_Q**2)/(ran_x*ran_s)
		if ran_y > 1: continue
		ran_phi_kp = rng.uniform(low=ranges['phi_kp'][0], high=ranges['phi_kp'][1])
		ran_phi_Dp = rng.uniform(low=ranges['phi_Dp'][0], high=ranges['phi_Dp'][1])
		# ran_kT = ran_Q*(np.sqrt(1-y)/y)

		ran_kinematic_vars = {'s': ran_s, 'Q': ran_Q, 'x': ran_x, 'delta': ran_delta, 'pT': ran_pT, 'z': ran_z, 'y': ran_y, 'phi_kp': ran_phi_kp, 'phi_Dp': ran_phi_Dp}
		ran_dsa = get_DSA(polarized_dipoles, unpolarized_dipole, ran_kinematic_vars)
		ran_unpolar = get_unpolar_xsec(polarized_dipoles, unpolarized_dipole, ran_kinematic_vars)
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

	try:
		# outfile = 'test_data.npy'
		np.save(outfile, data)
		print('\n=== saved mc data in file', outfile)
	except:
		print('unable to save data')

	
















