import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import jv, kv
import sys
import multiprocessing


def double_bessel(pT, Q, z, xBj, indices, amp, dipoles, lamIR = 0.3, IR_reg = [None, 0], deta=0.05):

	ia, ib, ic, id = indices
	pf = pT/lamIR
	Qf = (Q*np.sqrt(z * (1 - z)))/lamIR
	prefactor = (pf**ic)*(Qf**id)

	# polarized dipoles
	if amp != 'N':
		pdipole_dfs = dipoles
		bas = np.sqrt(3/(2*np.pi))
		prefactor *= (1/(2*bas))
		target_eta = round((bas/deta) * np.log((Q**2) / (xBj*(lamIR ** 2)))) * deta
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
		ndipole_df = dipoles
		closest_y = ndipole_df['y'][np.isclose(ndipole_df['y'], np.log(1/xBj), atol=0.005)].iloc[0]
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




def get_coeff(flavor, pT, Q, z, xBj, polarized_dipoles, unpolarized_dipole):

	alpha_em = 1/137.0
	Zfsq = 2/3.0
	Nc = 3.0

	if flavor == 'A_TT':
		Q_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'Q', polarized_dipoles)
		G2_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'G2', polarized_dipoles)
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)

		s = (Q**2)/xBj
		prefactor = -(alpha_em*Zfsq*(Nc**2)*(Q**2)*z*(1-z))/(2*(np.pi**4)*s)
		return prefactor*(((1 - 2*z)**2)*Q_11 + 2*(z**2 + (1-z)**2)*G2_11)*N_11


	elif flavor == 'B_TT': 
		Q_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'Q', polarized_dipoles)
		Q_21_10 = double_bessel(pT, Q, z, xBj, [2,1,1,0], 'Q', polarized_dipoles)
		G2_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'G2', polarized_dipoles)
		G2_21_10 = double_bessel(pT, Q, z, xBj, [2,1,1,0], 'G2', polarized_dipoles)
		I3_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I3', polarized_dipoles)
		I3_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'I3', polarized_dipoles)
		I4_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I4', polarized_dipoles)
		I4_21_10 = double_bessel(pT, Q, z, xBj, [2,1,1,0], 'I4', polarized_dipoles)
		I4_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I4', polarized_dipoles)
		I5_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I5', polarized_dipoles)
		I5_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I5', polarized_dipoles)
		I5_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'I5', polarized_dipoles)
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)
		N_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'N', unpolarized_dipole)

		b_TT = 0.5*((1-2*z)**2)*N_01_10*Q_11
		b_TT += (z**2 + (1-z)**2)*N_01_10*G2_11 + N_11*(0.5*Q_11 + I3_11 - I3_01_10)
		b_TT += (z**2 + (1-z)**2)*N_11*(Q_21_10 + 2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10)

		s = (Q**2)/xBj
		prefactor = (alpha_em*Zfsq*(Nc**2)*(Q**2)*z*(1-z))/(2*(np.pi**4)*s)
		return prefactor*(1-(2*z))*b_TT


	elif flavor == 'A_LT': 
		Q_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'Q', polarized_dipoles)
		Q_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'Q', polarized_dipoles)
		G2_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'G2', polarized_dipoles)
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)
		N_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'N', unpolarized_dipole)

		s = (Q**2)/xBj
		prefactor = -(alpha_em*Zfsq*(Nc**2)*(Q**2)*((z*(1-z))**1.5))/(np.sqrt(2)*(np.pi**4)*s)

		return prefactor*(1- 2*z)*(N_00*(2*G2_11 - Q_11) - N_11*Q_00)


	elif flavor == 'B_LT': 
		Q_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'Q', polarized_dipoles)
		Q_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'Q', polarized_dipoles)
		Q_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'Q', polarized_dipoles)
		Q_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'Q', polarized_dipoles)
		G2_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'G2', polarized_dipoles)
		G2_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'G2', polarized_dipoles)
		I3_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I3', polarized_dipoles)
		I3_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'I3', polarized_dipoles)
		I3_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'I3', polarized_dipoles)
		I4_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I4', polarized_dipoles)
		I4_21_10 = double_bessel(pT, Q, z, xBj, [2,1,1,0], 'I4', polarized_dipoles)
		I4_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I4', polarized_dipoles)
		I5_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I5', polarized_dipoles)
		I5_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I5', polarized_dipoles)
		I5_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'I5', polarized_dipoles)
		N_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'N', unpolarized_dipole)
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)
		N_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'N', unpolarized_dipole)
		N_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'N', unpolarized_dipole)

		b_LT = N_00*(I3_11 - I3_01_10)
		b_LT += N_11*I3_10_10 
		b_LT += (z**2 + (1-z)**2)*N_00*(Q_01_10 - Q_11)
		b_LT -= (z**2 + (1-z)**2)*N_11*Q_10_10 
		b_LT -= 0.5*((1- 2*z)**2)*N_10_10*Q_11
		b_LT -= 0.5*((1- 2*z)**2)*(N_11 - N_01_10)*Q_00
		b_LT += ((1- 2*z)**2)*N_00*(3*G2_11 - 2*G2_01_10 + I4_21_10 + I4_10_01 - I5_11 + I5_01_10 + I5_10_01)
		b_LT += ((1- 2*z)**2)*N_10_10*G2_11

		s = (Q**2)/xBj
		prefactor = -(alpha_em*Zfsq*(Nc**2)*(Q**2)*((z*(1-z))**1.5))/(np.sqrt(2)*(np.pi**4)*s)
		return prefactor*b_LT


	elif flavor == 'C_LT': 
		Q_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'Q', polarized_dipoles)
		Q_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'Q', polarized_dipoles)
		G2_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'G2', polarized_dipoles)
		G2_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'G2', polarized_dipoles)
		G2_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'G2', polarized_dipoles)
		I3_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I3', polarized_dipoles)
		I4_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I4', polarized_dipoles)
		I4_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I4', polarized_dipoles)
		I4_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'I4', polarized_dipoles)
		I4_01_10 = double_bessel(pT, Q, z, xBj, [0,1,1,0], 'I4', polarized_dipoles)
		I4_11_20 = double_bessel(pT, Q, z, xBj, [1,1,2,0], 'I4', polarized_dipoles)
		I4_00_11 = double_bessel(pT, Q, z, xBj, [0,0,1,1], 'I4', polarized_dipoles)
		I4_11_11 = double_bessel(pT, Q, z, xBj, [1,1,1,1], 'I4', polarized_dipoles)
		I4_00_20 = double_bessel(pT, Q, z, xBj, [0,0,2,0], 'I4', polarized_dipoles)
		I5_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'I5', polarized_dipoles)
		I5_11_20 = double_bessel(pT, Q, z, xBj, [1,1,2,0], 'I5', polarized_dipoles)
		I5_10_01 = double_bessel(pT, Q, z, xBj, [1,0,0,1], 'I5', polarized_dipoles)
		I5_00_11 = double_bessel(pT, Q, z, xBj, [0,0,1,1], 'I5', polarized_dipoles)
		I5_11_11 = double_bessel(pT, Q, z, xBj, [1,1,1,1], 'I5', polarized_dipoles)
		I5_10_10 = double_bessel(pT, Q, z, xBj, [1,0,1,0], 'I5', polarized_dipoles)
		I5_00_20 = double_bessel(pT, Q, z, xBj, [0,0,2,0], 'I5', polarized_dipoles)
		N_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'N', unpolarized_dipole)
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)

		c_LT = (z**2 + (1-z)**2)*N_11*Q_11
		c_LT -= N_00*I3_11
		c_LT += 0.5*((1- 2*z)**2)*N_11*Q_00
		c_LT += ((1- 2*z)**2)*N_00*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
		c_LT += ((1- 2*z)**2)*N_11*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)

		s = (Q**2)/xBj
		prefactor = -(alpha_em*Zfsq*(Nc**2)*(Q**2)*((z*(1-z))**1.5))/(np.sqrt(2)*(np.pi**4)*s)
		return prefactor*c_LT


	elif flavor == 'A_TT_unpolar':
		N_11 = double_bessel(pT, Q, z, xBj, [1,1,0,0], 'N', unpolarized_dipole)
		prefactor = (4*alpha_em*Zfsq*(Nc**2)*(Q**2)*(z**2)*((1-z)**2))/((2*np.pi**4))
		return prefactor*(z**2 + (1-z)**2)*(N_11**2)


	elif flavor == 'A_LL_unpolar':
		N_00 = double_bessel(pT, Q, z, xBj, [0,0,0,0], 'N', unpolarized_dipole)
		prefactor = (8*alpha_em*Zfsq*(Nc**2)*(Q**2)*(z**3)*((1-z)**3))/((2*np.pi**4))
		return prefactor*(N_00**2)


	else:
		print('requested coefficient', flavor, 'does not exist')




def get_DSA(polarized_dipoles, unpolarized_dipole, kinematic_vars):

	alpha_em = 1/137.0
	Zfsq = 2/3.0
	Nc = 3.0

	s, Q, x, delta, pT, z, y, phi_kp, phi_Dp = kinematic_vars
	prefactor = alpha_em/(4*(np.pi**2)*(Q**2)*y)

	tt_term = (2-y)*(get_coeff('A_TT', pT, Q, z, x, polarized_dipoles, unpolarized_dipole) + (delta/pT)*np.cos(phi_Dp)*get_coeff('B_TT', pT, Q, z, x, polarized_dipoles, unpolarized_dipole))
	lt_term = np.sqrt(2-2*y)*(np.cos(phi_kp)*get_coeff('A_LT', pT, Q, z, x, polarized_dipoles, unpolarized_dipole) + (delta/pT)*np.cos(phi_Dp)*np.cos(phi_kp)*get_coeff('B_LT', pT, Q, z, x, polarized_dipoles, unpolarized_dipole) + (delta/pT)*np.sin(phi_Dp)*np.sin(phi_kp)*get_coeff('C_LT', pT, Q, z, x, polarized_dipoles, unpolarized_dipole))

	return prefactor*(tt_term + lt_term)



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




def generate_data_chunk(chunk_size, ranges, rng, sample_size, polarized_dipoles, unpolarized_dipole, shared_data, progress_dict, process_id):
	# data_chunk = []

	count = 0
	while len(shared_data) < chunk_size:
		ran_s = rng.uniform(low=ranges['s'][0], high=ranges['s'][1])
		ran_Q = rng.uniform(low=ranges['Q'][0], high=ranges['Q'][1])
		ran_x = rng.uniform(low=ranges['x'][0], high=ranges['x'][1])
		if np.log(1/ran_x) < 4.61 or np.log(1/ran_x) > 9.21: continue

		ran_delta = rng.uniform(low=ranges['delta'][0], high=ranges['delta'][1])
		ran_pT = rng.uniform(low=ranges['pT'][0], high=ranges['pT'][1])
		ran_z = rng.uniform(low=ranges['z'][0], high=ranges['z'][1])
		ran_y = (ran_Q**2)/(ran_x*ran_s)
		if ran_y > 1: continue
		ran_phi_kp = rng.uniform(low=ranges['phi_kp'][0], high=ranges['phi_kp'][1])
		ran_phi_Dp = rng.uniform(low=ranges['phi_Dp'][0], high=ranges['phi_Dp'][1])

		ran_kinematic_vars = [ran_s, ran_Q, ran_x, ran_delta, ran_pT, ran_z, ran_y, ran_phi_kp, ran_phi_Dp]
		ran_dsa = get_DSA(polarized_dipoles, unpolarized_dipole, ran_kinematic_vars)
		ran_xsec = np.exp(rng.uniform(low=ranges['log_xsec'][0], high=ranges['log_xsec'][1]))

		if ran_xsec < np.abs(ran_dsa):
			# data_chunk.append(ran_kinematic_vars + [ran_dsa])
			shared_data.append(ran_kinematic_vars + [ran_dsa])
			count += 1

		# Update progress for the current process
		progress_dict[process_id] = count

	return shared_data




def parallel_generate_data(sample_size, ranges, polarized_dipoles, unpolarized_dipole):
	num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
	chunk_size = sample_size // num_processes  # Split the task into chunks

	# Create a manager for shared memory and progress tracking
	with multiprocessing.Manager() as manager:
		shared_data = manager.list()  # Shared list to store generated data
		progress_dict = manager.dict()  # Shared dictionary to track progress

		# Create a pool of processes
		with multiprocessing.Pool(processes=num_processes) as pool:
			# Start the worker processes, passing the progress dictionary and process ID
			pool.starmap(generate_data_chunk, [(chunk_size, ranges, np.random.default_rng(), sample_size, polarized_dipoles, unpolarized_dipole, shared_data, progress_dict, process_id) for process_id in range(num_processes)])

		# Print the progress of each process periodically (from the main process)
		while len(shared_data) < sample_size:
			sys.stdout.write(f'\rProgress: {len(shared_data)}/{sample_size} ({(len(shared_data)/sample_size)*100:.2f}%)')
			sys.stdout.flush()
			time.sleep(1)  # Sleep for a second before updating the progress display

		# Convert shared list to a regular Python list (optional)
		data = list(shared_data)

	return data




if __name__ == '__main__':

	# constants
	alpha_em = 1/137.0
	Zfsq = 2/3.0
	Nc = 3.0

	# basic acceptance-rejection method of generating a finite sample
	sample_size = 1000
	rng = np.random.default_rng()

	ranges = {'s': [45**2, 140**2],
			  'Q': [np.sqrt(5), 10],
			  'x': [0.001, 0.01],
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

	print('=== starting parallel generation of', sample_size, 'events')
	data = parallel_generate_data(sample_size, ranges, polarized_dipoles, unpolarized_dipole)

	try:
		outfile = 'test_data.npy'
		np.save(outfile, data)
		print('\n=== saved mc data in file', outfile)
	except:
		print('unable to save data')

	
















