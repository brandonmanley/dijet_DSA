import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.special import jv, kv


class DijetXsec:
	def __init__(self, deta=0.05, tag='_rc'): 
		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Zfsq = 2.0/3.0

		# load unpolarized dipole amplitude
		unpolar_input_file = '/Users/brandonmanley/Desktop/PhD/rcbkdipole/build/bin/dipole_data/unpolarized_dipole_ymin4.610000_ymax9.210000.dat'
		self.ndipole = pd.read_csv(unpolar_input_file, sep=r'\s+', header=None, names=['y', 'ln(r)', 'N[ln(r)]'])

		# load polarized dipole amplitudes
		self.tag = tag
		self.deta = deta
		polar_indir = '/Users/brandonmanley/Desktop/PhD/moment_evolution/evolved_dipoles/largeNc&Nf/d'+str(self.deta)[2:]+'_ones/'
		polar_input_files = [polar_indir+'d'+str(self.deta)[2:]+'_NcNf3_ones_'+ia+self.tag+'.dat' for ia in ['Q', 'G2', 'I3', 'I4', 'I5']]
		polar_data_files = [np.loadtxt(ifile) for ifile in polar_input_files]
		self.pdipoles = pd.DataFrame(self.to_array(polar_data_files), columns = ['s10', 'eta', 'Q', 'G2', 'I3', 'I4', 'I5'])


	def to_array(self, data_files):
		dipoles = []
	
		if 'nocutoff' in self.tag:
			nsteps = len(data_files[0])
			s10_range = np.arange(-nsteps + 1, nsteps)
			eta_range = np.arange(nsteps)
		
			for is10 in s10_range:
				for ieta in eta_range:
					if ieta > is10 and ieta < is10+nsteps: 
						dipoles.append([is10*self.deta, ieta*self.deta] + [idipole[is10-ieta+nsteps, ieta] for idipole in data_files])
					else: 
						# fixme: does not account for different initial conditions than 1
						dipoles.append([is10*self.deta, ieta*self.deta] + [1 for idipole in data_files])       
			return dipoles
	
		else:
			for is10, s10_row in enumerate(data_files[0]):
				for ieta, eta in enumerate(s10_row):
					dipole_values = [ifile[is10, ieta] for ifile in data_files]
					dipoles.append([(is10+1)*self.deta, (ieta+1)*self.deta] + dipole_values)
			return dipoles


	def double_bessel(self, kvar, indices, amp, lamIR = 0.3, IR_reg = [None, 0]):

		Q, pT, z, x, s = kvar['Q'], kvar['pT'], kvar['z'], kvar['x'], kvar['s']
		ia, ib, ic, id = indices
		pf = pT/lamIR
		Qf = (Q*np.sqrt(z*(1-z)))/lamIR
		prefactor = (pf**ic)*(Qf**id)
	
		# polarized dipoles
		if amp != 'N':
			bas = np.sqrt(3/(2*np.pi))
			prefactor *= (1/(2*bas))

			# select A(s_{10}, \eta = \eta(x))
			target_eta = round((bas/self.deta)*np.log(s/(lamIR**2)))*self.deta
			dipole = self.pdipoles[np.isclose(self.pdipoles['eta'], target_eta, atol=self.deta*0.5)][['s10', amp]]
	
			s10 = dipole['s10'].to_numpy()
			amp_values = dipole[amp].to_numpy()
			u = -s10*(1/(2*bas))
			size = self.deta
	
			if IR_reg[0] == 'gauss':
				amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
			elif IR_reg[0] == 'skin':
				amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(lamIR*IR_reg[2])) - 1)))
			elif IR_reg[0] == 'cut':
				amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
			else:
				amp_values = np.where(s10 < 0, 0, amp_values)

		# unpolarized dipole
		else:
			# select N(r, x = xBj)
			closest_y = self.ndipole['y'][np.isclose(self.ndipole['y'], np.log(1/x), atol=0.1)].iloc[0]
			dipole = self.ndipole[np.isclose(self.ndipole['y'], closest_y, atol=0.005)]
	
			# Extract and process columns
			u = dipole['ln(r)'].to_numpy()
			amp_values = 15*2.57*(dipole['N[ln(r)]'].to_numpy()) # 15*2.57 (GeV^{-2}) is value of \int d^2 b from fit in 2407.10581  
			amp_values = np.where(u > np.log(1/lamIR), 0, amp_values)
			size = 0.01  # from evolution code
	
		# Compute the Riemann sum in a vectorized way
		exp_term = np.exp(u*(2+ic+id))
		jv_term = jv(ia, pf*np.exp(u))
		kv_term = kv(ib, Qf*np.exp(u))
	
		# Perform the sum
		total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
		return prefactor*total_sum


	def get_coeff(self, kvar, flavor):

		prefactor = self.alpha_em*self.Zfsq*(self.Nc**2)
		if 'TT' in flavor: prefactor *= -((kvar['Q']**2)*kvar['z']*(1-kvar['z']))/(2*(np.pi**4)*kvar['s'])
		elif 'LT' in flavor: prefactor *= -((kvar['Q']**2)*((kvar['z']*(1-kvar['z']))**1.5))/(np.sqrt(2)*(np.pi**4)*kvar['s'])
		elif 'TT_unpolar' in flavor: prefactor *= (4*(kvar['Q']**2)*(kvar['z']**2)*((1-kvar['z'])**2)*((kvar['z'])**2 + (1-kvar['z'])**2))/(2*(np.pi**4))
		elif 'LL_unpolar' in flavor: prefactor *= (8*(kvar['Q']**2)*(kvar['z']**3)*((1-kvar['z'])**3))/(2*(np.pi**4))
		elif 'TmT_unpolar' in flavor: prefactor *= (4*(kvar['Q']**2)*(kvar['z']**2)*((1-kvar['z'])**2)*((1- 2*kvar['z'])**2 - 1))/(2*(np.pi**4))

		if flavor == 'A_TT':
			Q_11 = self.double_bessel(kvar, [1,1,0,0], 'Q')
			G2_11 = self.double_bessel(kvar, [1,1,0,0], 'G2')
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			return prefactor*(((1 - 2*kvar['z'])**2)*Q_11 + 2*(kvar['z']**2 + (1-kvar['z'])**2)*G2_11)*N_11

		elif flavor == 'B_TT': 
			Q_11 = self.double_bessel(kvar, [1,1,0,0], 'Q')
			Q_21_10 = self.double_bessel(kvar, [2,1,1,0], 'Q')
			G2_11 = self.double_bessel(kvar, [1,1,0,0], 'G2')
			G2_21_10 = self.double_bessel(kvar, [2,1,1,0], 'G2')
			I3_11 = self.double_bessel(kvar, [1,1,0,0], 'I3')
			I3_01_10 = self.double_bessel(kvar, [0,1,1,0], 'I3')
			I4_11 = self.double_bessel(kvar, [1,1,0,0], 'I4')
			I4_21_10 = self.double_bessel(kvar, [2,1,1,0], 'I4')
			I4_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I4')
			I5_11 = self.double_bessel(kvar, [1,1,0,0], 'I5')
			I5_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I5')
			I5_01_10 = self.double_bessel(kvar, [0,1,1,0], 'I5')
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			N_01_10 = self.double_bessel(kvar, [0,1,1,0], 'N')

			b_TT = 0.5*((1-2*kvar['z'])**2)*N_01_10*Q_11
			b_TT += (kvar['z']**2 + (1-kvar['z'])**2)*N_01_10*G2_11 + N_11*(0.5*Q_11 + I3_11 - I3_01_10)
			b_TT += (kvar['z']**2 + (1-kvar['z'])**2)*N_11*(Q_21_10 + 2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10)
			return prefactor*(1-(2*kvar['z']))*b_TT

		elif flavor == 'A_LT': 
			Q_11 = self.double_bessel(kvar, [1,1,0,0], 'Q')
			Q_00 = self.double_bessel(kvar, [0,0,0,0], 'Q')
			G2_11 = self.double_bessel(kvar, [1,1,0,0], 'G2')
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			N_00 = self.double_bessel(kvar, [0,0,0,0], 'N')
			return prefactor*(1- 2*kvar['z'])*(N_00*(2*G2_11 - Q_11) - N_11*Q_00)

		elif flavor == 'B_LT': 
			Q_00 = self.double_bessel(kvar, [0,0,0,0], 'Q')
			Q_11 = self.double_bessel(kvar, [1,1,0,0], 'Q')
			Q_01_10 = self.double_bessel(kvar, [0,1,1,0], 'Q')
			Q_10_10 = self.double_bessel(kvar, [1,0,1,0], 'Q')
			G2_11 = self.double_bessel(kvar, [1,1,0,0], 'G2')
			G2_01_10 = self.double_bessel(kvar, [0,1,1,0], 'G2')
			I3_11 = self.double_bessel(kvar, [1,1,0,0], 'I3')
			I3_01_10 = self.double_bessel(kvar, [0,1,1,0], 'I3')
			I3_10_10 = self.double_bessel(kvar, [1,0,1,0], 'I3')
			I4_11 = self.double_bessel(kvar, [1,1,0,0], 'I4')
			I4_21_10 = self.double_bessel(kvar, [2,1,1,0], 'I4')
			I4_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I4')
			I5_11 = self.double_bessel(kvar, [1,1,0,0], 'I5')
			I5_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I5')
			I5_01_10 = self.double_bessel(kvar, [0,1,1,0], 'I5')
			N_00 = self.double_bessel(kvar, [0,0,0,0], 'N')
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			N_10_10 = self.double_bessel(kvar, [1,0,1,0], 'N')
			N_01_10 = self.double_bessel(kvar, [0,1,1,0], 'N')

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
			Q_00 = self.double_bessel(kvar, [0,0,0,0], 'Q')
			Q_11 = self.double_bessel(kvar, [1,1,0,0], 'Q')
			G2_11 = self.double_bessel(kvar, [1,1,0,0], 'G2')
			G2_01_10 = self.double_bessel(kvar, [0,1,1,0], 'G2')
			G2_10_10 = self.double_bessel(kvar, [1,0,1,0], 'G2')
			I3_11 = self.double_bessel(kvar, [1,1,0,0], 'I3')
			I4_11 = self.double_bessel(kvar, [1,1,0,0], 'I4')
			I4_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I4')
			I4_10_10 = self.double_bessel(kvar, [1,0,1,0], 'I4')
			I4_01_10 = self.double_bessel(kvar, [0,1,1,0], 'I4')
			I4_11_20 = self.double_bessel(kvar, [1,1,2,0], 'I4')
			I4_00_11 = self.double_bessel(kvar, [0,0,1,1], 'I4')
			I4_11_11 = self.double_bessel(kvar, [1,1,1,1], 'I4')
			I4_00_20 = self.double_bessel(kvar, [0,0,2,0], 'I4')
			I5_11 = self.double_bessel(kvar, [1,1,0,0], 'I5')
			I5_11_20 = self.double_bessel(kvar, [1,1,2,0], 'I5')
			I5_10_01 = self.double_bessel(kvar, [1,0,0,1], 'I5')
			I5_00_11 = self.double_bessel(kvar, [0,0,1,1], 'I5')
			I5_11_11 = self.double_bessel(kvar, [1,1,1,1], 'I5')
			I5_10_10 = self.double_bessel(kvar, [1,0,1,0], 'I5')
			I5_00_20 = self.double_bessel(kvar, [0,0,2,0], 'I5')
			N_00 = self.double_bessel(kvar, [0,0,0,0], 'N')
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')

			c_LT = (kvar['z']**2 + (1-kvar['z'])**2)*N_11*Q_11
			c_LT -= N_00*I3_11
			c_LT += 0.5*((1- 2*kvar['z'])**2)*N_11*Q_00
			c_LT += ((1- 2*kvar['z'])**2)*N_00*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
			c_LT += ((1- 2*kvar['z'])**2)*N_11*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)
			return prefactor*c_LT

		elif flavor == 'A_TT_unpolar':
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TT_unpolar':
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			N_21_10 = self.double_bessel(kvar, [2,1,1,0], 'N')
			return prefactor*2*(kvar['z']-0.5)*(2*(N_11**2) - N_11*N_21_10)

		elif flavor == 'A_LL_unpolar':
			N_00 = self.double_bessel(kvar, [0,0,0,0], 'N')
			return prefactor*(N_00**2)

		elif flavor == 'B_LL_unpolar':
			N_00 = self.double_bessel(kvar, [0,0,0,0], 'N')
			N_10_10 = self.double_bessel(kvar, [1,0,1,0], 'N')
			return -prefactor*2*(kvar['z']-0.5)*N_00*N_10_10

		elif flavor == 'A_TmT_unpolar':
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TmT_unpolar':
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			N_01_10 = self.double_bessel(kvar, [0,1,1,0], 'N')
			return prefactor*(kvar['z']-0.5)*(-2*(N_11**2) + N_11*N_01_10)

		elif flavor == 'C_TmT_unpolar':
			N_11 = self.double_bessel(kvar, [1,1,0,0], 'N')
			return -2*prefactor*(kvar['z']-0.5)*(N_11**2)

		else:
			print('requested coefficient', flavor, 'does not exist')


def get_xsec(self, kvar, kind):

	prefactor = self.alpha_em/(4*(np.pi**2)*(kvar['Q']**2))

	if kind == 'DSA':
		tt_term = (2-kvar['y'])*(self.get_coeff('A_TT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_TT', kvar))
		lt_term = np.sqrt(2-2*kvar['y'])*(np.cos(kvar['phi_kp'])*self.get_coeff('A_LT', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(kvar['phi_kp'])*self.get_coeff('B_LT', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(kvar['phi_kp'])*self.get_coeff('C_LT', kvar))
		return prefactor*(tt_term + lt_term)

	elif kind == 'unpolarized':
		tt_term = (1 + (1-kvar['y'])**2)*(self.get_coeff('A_TT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_TT_unpolar', kvar))
		tmt_term = -2*(1-kvar['y'])*(np.cos(2*kvar['phi_kp'])*self.get_coeff('A_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(2*kvar['phi_kp'])*self.get_coeff('B_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(2*kvar['phi_kp'])*self.get_coeff('C_TmT_unpolar', kvar))
		ll_term = 4*(1-kvar['y'])*(self.get_coeff('A_LL_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_LL_unpolar', kvar))
		return prefactor*(tt_term + tmt_term + ll_term)












		
