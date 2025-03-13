import numpy as np
import pandas as pd
import time
from scipy.special import jv, kv
import json
from scipy.interpolate import RegularGridInterpolator
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class DijetXsec:
	def __init__(self, deta=0.05, tag='_rc'): 
		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Zusq = (2.0/3.0)**2
		self.Zdsq = (-1.0/3.0)**2
		self.Zssq = (-1.0/3.0)**2
		self.Nf = 3.0
		self.lambdaIR = 0.3

		# other constants
		self.s10_values = np.arange(0.0, 15.05, 0.05)


		# path to data (REPLACE WITH LOC OF DSA_MC) ################################
		current_dir = '/Users/brandonmanley/Desktop/PhD/dijet_dsa/bmanley/dsa_mc/'
		# current_dir = os.getcwd()
		############################################################################

		# load unpolarized dipole amplitude
		unpolar_input_file = current_dir + '/dipoles/unpolarized_dipole_ymin4.610000_ymax9.210000.dat'
		self.ndipole = pd.read_csv(unpolar_input_file, sep=r'\s+', header=None, names=['y', 'ln(r)', 'N[ln(r)]'])

		# load polarized dipole amplitudes 
		self.tag = tag
		self.deta = deta
		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = current_dir + f'/dipoles/{deta_str}_basis/'

		# ic_file = current_dir + '/dipoles/mc_ICs_random_fit.json'
		ic_file = current_dir + '/dipoles/mc_ICs_random_fit_2.json'
		with open(ic_file, "r") as file:	
			ics = json.load(file)

		self.pdipoles = {}
		amps = ['Qu', 'Qd', 'Qs', 'G', 'G2', 'Qt', 'I3u', 'I3d', 'I3s', 'I3t', 'I4', 'I5', 'It']

		for iamp, amp in enumerate(amps):
			if amp == 'G' or amp == 'Qt' or amp == 'I3t' or amp == 'It': continue

			input_data = [ics[ia][ib]*np.loadtxt(f'{polar_indir}{deta_str}_NcNf{int(self.Nf)}_{ia}{basis}_{amp}.dat') for ia in amps for ib, basis in enumerate(['a', 'b', 'c'])]	
			input_dipole = np.sum(np.stack(input_data, axis=0), axis=0)
			assert len(input_data) == 39, f'number of input files {len(input_data)} wrong'
			self.pdipoles[amp] = input_dipole



	def get_dipoles(self):
		return [self.ndipole, self.pdipoles]



	def to_array(self, data):
		dipole = []
	
		if 'nocutoff' in self.tag:
			nsteps = len(data)
			s10_range = np.arange(-nsteps + 1, nsteps)
			eta_range = np.arange(nsteps)
		
			for is10 in s10_range:
				for ieta in eta_range:
					if ieta > is10 and ieta < is10+nsteps: 
						dipole.append([is10*self.deta, ieta*self.deta] + data[is10-ieta+nsteps, ieta])
					else: 
						# fixme: does not account for different initial conditions than 1
						dipole.append([is10*self.deta, ieta*self.deta] + [1])       
			return np.array(dipole)
	
		else:
			for is10, s10_row in enumerate(data):
				for ieta, eta in enumerate(s10_row):
					dipole.append([(is10+1)*self.deta, (ieta+1)*self.deta, data[is10, ieta]])
			return np.array(dipole)



	def double_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):

		Q, pT, z, x, s = kvar['Q'], kvar['pT'], kvar['z'], kvar['x'], kvar['s']
		prefactor = 1

		# polarized dipoles
		if amp != 'N':
			bas = np.sqrt(3/(2*np.pi))
			prefactor *= (1/(2*bas))

			target_eta_index = round((bas/self.deta)*np.log((kvar['s']*kvar['y'])/(self.lambdaIR**2)))
			amp_values = self.pdipoles[amp][:, target_eta_index]
			u = -self.s10_values*(1/(2*bas))
			size = self.deta

			if IR_reg[0] == 'gauss':
				amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
			elif IR_reg[0] == 'skin':
				amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(self.lambdaIR*IR_reg[2])) - 1)))
			elif IR_reg[0] == 'cut':
				amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
			else:
				pass 
				# don't need line below since not using evolution where s_{10} < 0
				# amp_values = np.where(s10 < 0, 0, amp_values)

		# unpolarized dipole
		else:
			# Extract and process columns
			u = self.f_ndipole['ln(r)'].to_numpy()
			amp_values = 14*(3.894*(10**5))*(self.f_ndipole['N[ln(r)]'].to_numpy()) # 14 (mb) is value of \int d^2 b from fit in 2407.10581  
			size = 0.01  # from evolution code
			amp_values = np.where(u > np.log(1/self.lambdaIR), 0, amp_values)
	
		# Compute the Riemann sum in a vectorized way for each set of indices
		results = []
		for i_a, i_b, i_c, i_d in indices_array:
			pf = pT/self.lambdaIR
			Qf = (Q*np.sqrt(z*(1-z)))/self.lambdaIR
			prefactor = (pf**i_c)*(Qf**i_d)
			exp_term = np.exp(u*(2+i_c+i_d))
			jv_term = jv(i_a, pf*np.exp(u))
			kv_term = kv(i_b, Qf*np.exp(u))
	
			# Perform the sum
			total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
			results.append(prefactor*total_sum)

		if len(indices_array) > 1: return results 
		else: return results[0]



	def get_coeff(self, flavor, kvar):

		Zfsq = self.Zusq + self.Zdsq + self.Zssq
		prefactor = self.alpha_em*(self.Nc**2)*(kvar['Q']**2)

		if 'TT_unpolar' in flavor: prefactor *= Zfsq*(4*(kvar['z']**2)*((1-kvar['z'])**2)*((kvar['z'])**2 + (1-kvar['z'])**2))/(2*(np.pi**4))
		elif 'LL_unpolar' in flavor: prefactor *= Zfsq*(8*(kvar['z']**3)*((1-kvar['z'])**3))/(2*(np.pi**4))
		elif 'TmT_unpolar' in flavor: prefactor *= Zfsq*(-8*(kvar['z']**3)*((1-kvar['z'])**3))/(2*(np.pi**4))
		elif 'TT' in flavor: prefactor *= -(kvar['z']*(1-kvar['z']))/(2*(np.pi**4)*kvar['s']*kvar['y'])
		elif 'LT' in flavor: prefactor *= -((kvar['z']*(1-kvar['z']))**1.5)/(np.sqrt(2)*(np.pi**4)*kvar['s']*kvar['y'])

		if flavor == 'A_TT':
			Qu_11 = self.double_bessel(kvar, [[1,1,0,0]], 'Qu')
			Qd_11 = self.double_bessel(kvar, [[1,1,0,0]], 'Qd')
			Qs_11 = self.double_bessel(kvar, [[1,1,0,0]], 'Qs')
			G2_11 = self.double_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11 = self.double_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(((1 - 2*kvar['z'])**2)*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zdsq*Qs_11)+ 2*(kvar['z']**2 + (1-kvar['z'])**2)*Zfsq*G2_11)*N_11

		elif flavor == 'B_TT': 
			Qu_11, Qu_21_10 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qu')
			Qd_11, Qd_21_10 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qd')
			Qs_11, Qs_21_10 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qs')
			G2_11, G2_21_10 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'G2')
			I3u_11, I3u_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3u')
			I3d_11, I3d_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3d')
			I3s_11, I3s_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3s')
			I4_11, I4_21_10, I4_10_01 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0], [1,0,0,1]], 'I4')
			I5_11, I5_10_01, I5_01_10 = self.double_bessel(kvar, [[1,1,0,0], [1,0,0,1], [0,1,1,0]], 'I5')
			N_11, N_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'N')

			b_TT = 0.5*((1-2*kvar['z'])**2)*N_01_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_TT += Zfsq*(kvar['z']**2 + (1-kvar['z'])**2)*N_01_10*G2_11 + N_11*(0.5*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)+ (self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11) - (self.Zusq*I3u_01_10 + self.Zdsq*I3d_01_10 + self.Zssq*I3s_01_10)) 
			b_TT += (kvar['z']**2 + (1-kvar['z'])**2)*N_11*(self.Zusq*Qu_21_10 + self.Zdsq*Qd_21_10 + self.Zssq*Qs_21_10 + 2*Zfsq*G2_21_10 + Zfsq*I4_21_10 + Zfsq*I4_10_01 - Zfsq*I5_11 + Zfsq*I5_10_01 + Zfsq*I5_01_10)
			
			return prefactor*(1-(2*kvar['z']))*b_TT

		elif flavor == 'A_LT': 
			Qu_11, Qu_00 = self.double_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qu')
			Qd_11, Qd_00 = self.double_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qd')
			Qs_11, Qs_00 = self.double_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qs')
			G2_11 = self.double_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11, N_00 = self.double_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'N')
			
			return prefactor*(1- 2*kvar['z'])*(N_00*(2*Zfsq*G2_11 - (self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)) - N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00))

		elif flavor == 'B_LT': 
			Qu_00, Qu_11, Qu_01_10, Qu_10_10 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qu')
			Qd_00, Qd_11, Qd_01_10, Qd_10_10 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qd')
			Qs_00, Qs_11, Qs_01_10, Qs_10_10 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qs')
			G2_11, G2_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'G2')
			I3u_11, I3u_01_10, I3u_10_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3u')
			I3d_11, I3d_01_10, I3d_10_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3d')
			I3s_11, I3s_01_10, I3s_10_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3s')
			I4_11, I4_21_10, I4_10_01 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0], [1,0,0,1]], 'I4')
			I5_11, I5_10_01, I5_01_10 = self.double_bessel(kvar, [[1,1,0,0], [1,0,0,1], [0,1,1,0]], 'I5')
			N_00, N_11, N_10_10, N_01_10 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0], [1,0,1,0], [0,1,1,0]], 'N')

			b_LT = N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11 - self.Zusq*I3u_01_10 - self.Zdsq*I3d_01_10 - self.Zssq*I3s_01_10)
			b_LT += N_11*(self.Zusq*I3u_10_10 + self.Zdsq*I3d_10_10 + self.Zssq*I3s_10_10)
			b_LT += (kvar['z']**2 + (1-kvar['z'])**2)*N_00*(self.Zusq*Qu_01_10 - self.Zusq*Qu_11 + self.Zdsq*Qd_01_10 - self.Zdsq*Qd_11 + self.Zssq*Qs_01_10 - self.Zssq*Qs_11)
			b_LT -= (kvar['z']**2 + (1-kvar['z'])**2)*N_11*(self.Zusq*Qu_10_10 + self.Zdsq*Qd_10_10 + self.Zssq*Qs_10_10)
			b_LT -= 0.5*((1- 2*kvar['z'])**2)*N_10_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_LT -= 0.5*((1- 2*kvar['z'])**2)*(N_11 - N_01_10)*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			b_LT += ((1- 2*kvar['z'])**2)*Zfsq*N_00*(3*G2_11 - 2*G2_01_10 + I4_21_10 + I4_10_01 - I5_11 + I5_01_10 + I5_10_01)
			b_LT += ((1- 2*kvar['z'])**2)*Zfsq*N_10_10*G2_11
			
			return prefactor*b_LT

		elif flavor == 'C_LT': 
			Qu_00, Qu_11 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qu')
			Qd_00, Qd_11 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qd')
			Qs_00, Qs_11 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qs')
			G2_11, G2_01_10, G2_10_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'G2')
			I3u_11 = self.double_bessel(kvar, [[1,1,0,0]], 'I3u')
			I3d_11 = self.double_bessel(kvar, [[1,1,0,0]], 'I3d')
			I3s_11 = self.double_bessel(kvar, [[1,1,0,0]], 'I3s')
			I4_11, I4_10_01, I4_10_10, I4_01_10, I4_11_20, I4_00_11, I4_11_11, I4_00_20 = self.double_bessel(kvar, [[1,1,0,0], [1,0,0,1], [1,0,1,0], [0,1,1,0], [1,1,2,0], [0,0,1,1], [1,1,1,1], [0,0,2,0]], 'I4')
			I5_11, I5_11_20, I5_10_01, I5_00_11, I5_11_11, I5_10_10, I5_00_20 = self.double_bessel(kvar, [[1,1,0,0], [1,1,2,0], [1,0,0,1], [0,0,1,1], [1,1,1,1], [1,0,1,0], [0,0,2,0]], 'I5')
			N_00, N_11 = self.double_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'N')

			c_LT = (kvar['z']**2 + (1-kvar['z'])**2)*N_11*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			c_LT -= N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11)
			c_LT += 0.5*((1- 2*kvar['z'])**2)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			c_LT += ((1- 2*kvar['z'])**2)*N_00*Zfsq*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
			c_LT += ((1- 2*kvar['z'])**2)*N_11*Zfsq*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)
			
			return prefactor*c_LT

		elif flavor == 'A_TT_unpolar':
			N_11 = self.double_bessel(kvar, [[1,1,0,0]], 'N')
			
			return prefactor*(N_11**2)

		elif flavor == 'B_TT_unpolar':
			N_11, N_21_10 = self.double_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'N')
			
			return prefactor*2*(kvar['z']-0.5)*(2*(N_11**2) - N_11*N_21_10)

		elif flavor == 'A_LL_unpolar':
			N_00 = self.double_bessel(kvar, [[0,0,0,0]], 'N')
			
			return prefactor*(N_00**2)

		elif flavor == 'B_LL_unpolar':
			N_00, N_10_10 = self.double_bessel(kvar, [[0,0,0,0], [1,0,1,0]], 'N')
			
			return -prefactor*2*(kvar['z']-0.5)*N_00*N_10_10

		elif flavor == 'A_TmT_unpolar':
			N_11 = self.double_bessel(kvar, [[1,1,0,0]], 'N')
			
			return prefactor*(N_11**2)

		elif flavor == 'B_TmT_unpolar':
			N_11, N_01_10 = self.double_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'N')
			
			return prefactor*(kvar['z']-0.5)*(-2*(N_11**2) + N_11*N_01_10)

		elif flavor == 'C_TmT_unpolar':
			N_11 = self.double_bessel(kvar, [[1,1,0,0]], 'N')
			
			return -2*prefactor*(kvar['z']-0.5)*(N_11**2)

		else:
			print('requested coefficient', flavor, 'does not exist')



	# returns xsec in units of femptobarns (fb)
	def get_xsec(self, kvar, kind):

		xsec_prefactor = self.alpha_em/(4*(np.pi**2)*(kvar['Q']**2))
		self.filter_dipole(kvar)

		if kind == 'DSA':
			tt_term = (2-kvar['y'])*(self.get_coeff('A_TT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_TT', kvar))
			lt_term = np.sqrt(2-2*kvar['y'])*(np.cos(kvar['phi_kp'])*self.get_coeff('A_LT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(kvar['phi_kp'])*self.get_coeff('B_LT', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(kvar['phi_kp'])*self.get_coeff('C_LT', kvar))
			xsec = xsec_prefactor*(tt_term + lt_term)
			xsec /= 2.57*(10**(-12))
			return xsec

		elif kind == 'unpolarized':
			tt_term = (1 + (1-kvar['y'])**2)*(self.get_coeff('A_TT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_TT_unpolar', kvar))
			tmt_term = -2*(1-kvar['y'])*(np.cos(2*kvar['phi_kp'])*self.get_coeff('A_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(2*kvar['phi_kp'])*self.get_coeff('B_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(2*kvar['phi_kp'])*self.get_coeff('C_TmT_unpolar', kvar))
			ll_term = 4*(1-kvar['y'])*(self.get_coeff('A_LL_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_LL_unpolar', kvar))
			xsec = xsec_prefactor*(tt_term + tmt_term + ll_term)
			xsec /= 2.57*(10**(-12))
			return xsec


	def get_correlation(self, kvar, kind):

		if kind == '<1>': num = (2-kvar['y'])*self.get_coeff('A_TT', kvar)
		elif kind == '<cos(phi_Dp)>': num = 0.5*(2-kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('B_TT', kvar)
		elif kind == '<cos(phi_kp)>': num = 0.5*np.sqrt(2-2*kvar['y'])*self.get_coeff('A_LT', kvar)
		elif kind == '<cos(phi_Dp)cos(phi_kp)>': num = 0.25*np.sqrt(2-2*kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('B_LT', kvar)
		elif kind == '<sin(phi_Dp)sin(phi_kp)>': num = 0.25*np.sqrt(2-2*kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('C_LT', kvar)
		else: raise ValueError(f'Error: Correlation {kind} not recognized')

		den = (1 + (1-kvar['y'])**2)*self.get_coeff('A_TT_unpolar', kvar) + 4*(1-kvar['y'])*self.get_coeff('A_LL_unpolar', kvar)
			
		return num/den


	# filter unpolarized dipole before double bessel to save time
	def filter_dipole(self, kvar): 

		# filter N(Y = \ln(1/x), r) for unpolarized dipole
		closest_y = self.ndipole['y'][np.isclose(self.ndipole['y'], np.log(1/kvar['x']), atol=0.01)]
		if closest_y.empty: raise ValueError('Requested rapdity, Y=', np.log(1/x), ', does not exist in the data file')
		else: closest_y = closest_y.iloc[0]
		self.f_ndipole = self.ndipole[np.isclose(self.ndipole['y'], closest_y, atol=0.005)]





# function to plot histograms (documentation can be found in README)
def plot_histogram(dfs, plot_var, weights, constraints={}, **options):

	labels = {
		'num_dsa': 'DSA',
		'den_dsa': 'Total',
		'<1>': r'$A_{LL}$',
		'<cos(phi_kp)>': r'$\langle \cos(\phi_{kp}) \rangle$',
		'<cos(phi_Dp)>': r'$\langle \cos(\phi_{\Delta p}) \rangle$',
		'<cos(phi_Dp)cos(phi_kp)>': r'$\langle \cos(\phi_{kp}) \cos(\phi_{\Delta p}) \rangle$',
		'<sin(phi_Dp)sin(phi_kp)>': r'$\langle \sin(\phi_{kp}) \sin(\phi_{\Delta p}) \rangle$'
	}

	asp_ratio = 6/3
	psize = 4

	fig, axs = plt.subplots(1, 2, figsize=(asp_ratio*psize, psize), sharex=True, gridspec_kw={'width_ratios': [5, 1]})

	colors = ['black', 'red', 'blue', 'green']
	linestyles = ['-', '--', '-.']

	lumi = options.get('lumi', 10) # total integrated luminosity in fb^-1
	lumi *= options.get('efficiency', 1) # correct for detector efficiency

	for idf, df in enumerate(dfs):

		# make bins
		range = [df[plot_var].min(), df[plot_var].max()]
		nbins = options.get('nbins', 10)
		bin_width = options.get('binwidth', (range[1] - range[0])/nbins)
		bins = options.get('bins', np.arange(np.floor(range[0]/bin_width)*bin_width, np.ceil(range[1]/bin_width)*bin_width, bin_width))

		# enforce constraints
		mask = pd.Series(True, index=df.index)
		for var, (low, high) in constraints.items(): mask &= df[var].between(low, high)
		fixed_df = df[mask]

		if fixed_df.empty:
			print('Error: selected dataframe is empty - constraints are too strict')
			return

		for iw in weights: assert iw in list(df.columns), f'Error: option for weight {iw} not recognized'

		# make plot data         
		for iw, weight in enumerate(weights):

			total_counts, plot_bins = np.histogram(fixed_df[plot_var], bins=bins, weights=fixed_df['den_dsa'])
			plot_counts, _ = np.histogram(fixed_df[plot_var], bins=bins, weights=fixed_df[weight])
			bin_centers = 0.5*(plot_bins[:-1]+plot_bins[1:])

			# ensure bins are properly averaged
			n_entries, _ = np.histogram(fixed_df[plot_var], bins=bins)
			plot_counts = np.array([icount/(bin_width*ientry) if ientry != 0 else 0 for icount, ientry in zip(plot_counts, n_entries)])
			total_counts = np.array([icount/(bin_width*ientry) if ientry != 0 else 0 for icount, ientry in zip(total_counts, n_entries)])

			# errors are calculated for a given integrated luminosity (5% systematic error added per 1505.05783)
			if '<' in weight:
				stat_errors = np.array([np.sqrt((1+ic)/(lumi*tc)) if tc != 0 else 0 for ic, tc in zip(plot_counts, total_counts)])
			else:
				stat_errors = np.sqrt(total_counts/lumi)
			sys_errors = 0.05*np.abs(plot_counts)  
			total_errors = np.sqrt((stat_errors**2)+(sys_errors)**2)

			# make data plot
			axs[0].errorbar(
				bin_centers, plot_counts, yerr=total_errors,
				fmt=options.get('fmt', 'o'), 
				capsize=3, elinewidth=0.5, capthick=0.5, color=colors[iw + idf]
			)
			if len(dfs) > 1: plot_label = labels[weight] + f'({idf})'
			else: plot_label = labels[weight] 
			axs[0].step(bin_centers, plot_counts, where='mid', linestyle=linestyles[idf], color=colors[iw+idf], label=plot_label, linewidth=1)

		axs[0].legend(frameon=False)

		# make info box for constraints
		info_text = fr'Integrated luminosity: ${lumi}\,\, \mathrm{{fb}}^{{-1}}$'+'\n'
		info_text += fr'$\sqrt{{s}} = {round(np.sqrt(df['s'][0]))}\,\, \mathrm{{GeV}}$'+'\n'
		info_text += '\nCuts:'
		for var, (low, high) in constraints.items():
			if var == 'Q' or var == 'pT' or var == 'delta':
				info_text += '\n'+ fr'$ {low} < {var}\,[\mathrm{{GeV}}] < {high}$'
			elif var == 't':
				info_text += '\n'+ fr'$ {low} < {var}\,[\mathrm{{GeV}}^2] < {high}$'
			else:
				info_text += '\n' + fr'$ {low} < {var} < {high}$'
		axs[1].text(
			0.0, 0.0, info_text, 
			ha='left', va='bottom', 
			fontsize=12, wrap=True, 
			bbox=dict(boxstyle='round', facecolor='white', alpha=0.3)
		)
		axs[1].set_axis_off()

		# set plot info
		if options.get('y_limits'): axs[0].set_ylim(options.get('y_limits'))
		axs[0].set_xlim(options.get('x_limits', [plot_bins[0], plot_bins[-1]]))
		axs[0].set_title(options.get('title', ''))
		axs[0].set_ylabel(options.get('y_label', 'Frequency'), loc='top')
		axs[0].set_xlabel(options.get('x_label', fr'${plot_var}$'), loc='right')
		axs[0].set_yscale(options.get('y_scale', 'linear'))
		axs[0].set_xscale(options.get('x_scale', 'linear'))
		axs[0].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=True)
		axs[0].grid(options.get('grid', False)) 
		axs[0].xaxis.set_major_locator(MaxNLocator(nbins=10))
		if options.get('zero_line', False): axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
		if options.get('min_pT_line', False) and plot_var == 'pT': 
			axs[0].axvline(x=1, color='gray', linestyle='--')
			axs[0].fill_betweenx(axs[0].get_ylim(), axs[0].get_xlim()[0], 1, color='gray', alpha=0.25)

		if options.get('saveas'): fig.savefig(options.get('saveas'), dpi=400, bbox_inches="tight")
		plt.tight_layout()






		
