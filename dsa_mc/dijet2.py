import numpy as np
import pandas as pd
import time
from scipy.special import jv, kv
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
from dataclasses import dataclass

import _pickle as cPickle
import sys
import zlib


def load(name): 
	compressed = open(name,"rb").read()
	data = cPickle.loads(zlib.decompress(compressed))
	return data


def save(data, name):  
	compressed = zlib.compress(cPickle.dumps(data))
	f = open(name,"wb")
	try:
		f.writelines(compressed)
	except:
		f.write(compressed)
		f.close()


def regulate_IR(values, r, params):
	
	if params[0] == 'gauss':
		values *= np.exp(-(r**2)*params[1])

	elif params[0] == 'skin':
		values *= 1.0/(1 + np.exp(params[1]*((r/params[2]) - 1)))
		
	return values


# @dataclass(frozen=True)
class Kinematics:
    x: float
    Q: float
    z: float
    pT: float
    y: float
    delta: float = 0.0
    phi_Dp: float = 0.0
    phi_kp: float = 0.0


class DIJET:
	def __init__(self, nreplica=1, lambdaIR=0.3, deta=0.05, gauss_param=0.0): 
		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Zusq = (2.0/3.0)**2
		self.Zdsq = (-1.0/3.0)**2
		self.Zssq = (-1.0/3.0)**2
		self.Zfsq = self.Zusq + self.Zdsq + self.Zssq 
		self.Nf = 3.0
		self.lambdaIR = lambdaIR
		self.gauss_param = gauss_param

		# grid values for dipoles
		self.deta = 0.05
		self.s10_values = np.arange(0.0, 15.0 + self.deta, self.deta)

		self.dlogr = 0.01
		self.logr_values = np.arange(-13.8, 1.2, self.dlogr)

		self.load_basis_dipoles()
		self.set_params(nreplica)

		# self.load_dipoles(nreplica)


	def load_basis_dipoles(self):

		current_dir = os.path.dirname(os.path.abspath(__file__))

		# load polarized dipole amplitudes 
		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = current_dir + f'/dipoles/{deta_str}-etamax15-rc/'

		self.basis_dipoles = {}
		# amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5', 'ITu', 'ITd', 'ITs']
		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']
		# amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2']

		for iamp in amps:
			if iamp in ['GT', 'QTu', 'QTd', 'QTs', 'I3T', 'ITu', 'ITd', 'ITs']: continue

			self.basis_dipoles[iamp] = {}
			input_dipole = 0
			for j, jamp in enumerate(amps):

				self.basis_dipoles[iamp][jamp] = {}
				for jbasis in ['eta', 's10','1']:

					self.basis_dipoles[iamp][jamp][jbasis] = load(f'{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')


	def set_params(self, nreplica=1):

		current_dir = os.path.dirname(os.path.abspath(__file__))

		#--load unpolarized dipole amplitude
		# mv parameters from 0902.1112
		unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_ymax9.81_AAMS09.dat'
		self.normN = 0.5*32.77*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# mv parameters from 2311.10491
		# unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_ymax9.21_CKM24.dat'
		# self.normN = 13.9*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		self.ndipole = pd.read_csv(unpolar_input_file, sep=r'\s+', header=None, names=['y', 'ln(r)', 'N'])


		# load initial condition parameters
		fdf = pd.read_csv(current_dir + '/dipoles/replica_params_old.csv')
		header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]
		fdf = fdf.dropna(axis=1, how='all')
		fdf.columns = header
		sdf = fdf[fdf['nrep'] == nreplica]
		assert len(sdf) == 1, 'Error: more than 1 replica selected...'

		ic_params = {}

		for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
		# for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2']:
			ic_params[amp] = {}
			for basis in ['eta','s10','1']:
				if amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2']:
					# print('im here!!', amp, basis)
					ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]
				else:
					# print('should not be here')
					ic_params[amp][basis] = random.uniform(-10, 10)


		self.pdipoles = {}
		# amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5', 'ITu', 'ITd', 'ITs']
		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']
		# amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2']

		for iamp in amps:
			if iamp in ['GT', 'QTu', 'QTd', 'QTs', 'I3T', 'ITu', 'ITd', 'ITs']: continue

			input_dipole = 0
			for j, jamp in enumerate(amps):
				for jbasis in ['eta', 's10','1']:
					temp_dipole = self.basis_dipoles[iamp][jamp][jbasis]
					input_dipole += ic_params[jamp][jbasis]*temp_dipole

			self.pdipoles[iamp] = input_dipole



	def get_dipoles(self):
		return [self.ndipole, self.pdipoles]


	def get_filtered_dipole(self): 
		return self.f_ndipole


	# def fourier_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):
	# 	# self.filter_dipole(kvar)

	# 	x, Q, z, pT = kvar.x, kvar.Q, kvar.z, kvar.pT
	# 	Qeff = Q*np.sqrt(z*(1-z))

	# 	results = []

	# 	# unpolarized dipole
	# 	if amp == 'N':

	# 		u = self.f_ndipole['ln(r)'].to_numpy()
	# 		amp_values = self.normN*self.f_ndipole['N'].to_numpy()  
	# 		size = 0.01  # from evolution code
	# 		amp_values = np.where(u > np.log(1/self.lambdaIR), 0, amp_values)

	# 		# if IR_reg[0] == 'gauss':
	# 		# 	amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
	# 		# elif IR_reg[0] == 'skin':
	# 		# 	amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(self.lambdaIR*IR_reg[2])) - 1)))
	# 		# elif IR_reg[0] == 'cut':
	# 		# 	amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
	# 		# else:
	# 		# 	pass 
	# 		# 	# don't need line below since not using evolution where s_{10} < 0
	# 		# 	# amp_values = np.where(s10 < 0, 0, amp_values)

	# 		for i_a, i_b, i_c, i_d in indices_array:

	# 			r_term = np.exp(2*u)
	# 			c_term = (pT*np.exp(u))**i_c
	# 			d_term = (Qeff*np.exp(u))**i_d
	# 			jv_term = jv(i_a, pT*np.exp(u))
	# 			kv_term = kv(i_b, Qeff*np.exp(u))
		
	# 			# Perform the sum
	# 			total_sum = size*np.sum(r_term*c_term*d_term*jv_term*kv_term*amp_values)
	# 			results.append(total_sum)


	# 		if len(indices_array) > 1: return results 
	# 		else: return results[0]


	# 	# polarized dipoles
	# 	else:
	# 		bas = np.sqrt(3/(2*np.pi))

	# 		# photon-proton center of mass energy squared
	# 		wsq = (Q**2)*((1/x) - 1)
	# 		# wsq = kvar['s']

	# 		target_eta_index = round((bas/self.deta)*np.log(wsq/(self.lambdaIR**2)))
	# 		amp_values = self.pdipoles[amp][:, target_eta_index]
	# 		r = (1/self.lambdaIR)*np.exp(-self.s10_values*(1/(2*bas)))
	# 		size = self.deta

	# 		# # if going to low-moderate Q^2, need one of these IR regulators
	# 		# if IR_reg[0] == 'gauss':
	# 		# 	amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
	# 		# elif IR_reg[0] == 'skin':
	# 		# 	amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(self.lambdaIR*IR_reg[2])) - 1)))
	# 		# elif IR_reg[0] == 'cut':
	# 		# 	amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
	# 		# else:
	# 		# 	pass 
	# 		# 	# don't need line below since not using evolution where s_{10} < 0
	# 		# 	# amp_values = np.where(s10 < 0, 0, amp_values)

	
	# 		# Compute the Riemann sum in a vectorized way for each set of indices
	# 		for i_a, i_b, i_c, i_d in indices_array:
	# 			r_term = (0.5/bas)*r**2
	# 			c_term = (pT*r)**i_c
	# 			d_term = (Qeff*r)**i_d
	# 			jv_term = jv(i_a, pT*r)
	# 			kv_term = kv(i_b, Qeff*r)
		
	# 			# Perform the sum
	# 			total_sum = size*np.sum(r_term*c_term*d_term*jv_term*kv_term*amp_values)
	# 			results.append(total_sum)

	# 		if len(indices_array) > 1: return results 
	# 		else: return results[0]



	def fourier_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):
		# self.filter_dipole(kvar)

		x, Q, z, pT = kvar.x, kvar.Q, kvar.z, kvar.pT
		Qeff = Q*np.sqrt(z*(1-z))

		results = []

		# unpolarized dipole
		if amp == 'N':

			r = np.exp(self.f_ndipole['ln(r)'].to_numpy())
			measure = 0.01*(r**2) # dln(r) = 0.01 from evolution code

			amp_values = self.normN*self.f_ndipole['N'].to_numpy()  
			amp_values = np.where(r > 1/self.lambdaIR, 0, amp_values)


		# polarized dipoles
		else:
			bas = np.sqrt(3/(2*np.pi))
			wsq = (Q**2)*((1/x) - 1)
			target_eta_index = round((bas/self.deta)*np.log(wsq/(self.lambdaIR**2)))
			amp_values = self.pdipoles[amp][:, target_eta_index]
			r = (1/self.lambdaIR)*np.exp(-self.s10_values*(1/(2*bas)))
			measure = self.deta*(0.5/bas)*(r**2)


		# need to regulate IR for small-moderate Q^2 
		if self.gauss_param > 0:
			amp_values = regulate_IR(amp_values, r, ['gauss', self.gauss_param])

	
		# Compute the Riemann sum in a vectorized way for each set of indices
		for i_a, i_b, i_c, i_d in indices_array:
			c_term = (pT*r)**i_c
			d_term = (Qeff*r)**i_d
			jv_term = jv(i_a, pT*r)
			kv_term = kv(i_b, Qeff*r)
	
			# Perform the sum
			total_sum = np.sum(measure*c_term*d_term*jv_term*kv_term*amp_values)
			results.append(total_sum)

		if len(indices_array) > 1: return results 
		else: return results[0]




	def get_coeff(self, flavor, kvar):

		prefactor = self.alpha_em*(self.Nc**2)*(kvar.Q**2)
		w2 = (kvar.Q**2)*((1/kvar.x) - 1)

		if 'TT_unpolar' in flavor: 
			# print(flavor, 'TT_unpolar')
			prefactor *= self.Zfsq*(4*(kvar.z**2)*((1-kvar.z)**2)*((kvar.z)**2 + (1-kvar.z)**2))/((2*np.pi)**4)
		elif 'LL_unpolar' in flavor: 
			# print(flavor, 'LL_unpolar')
			prefactor *= self.Zfsq*(8*(kvar.z**3)*((1-kvar.z)**3))/((2*np.pi)**4)
		elif 'TmT_unpolar' in flavor: 
			# print(flavor, 'TmT_unpolar')
			prefactor *= self.Zfsq*(-8*(kvar.z**3)*((1-kvar.z)**3))/((2*np.pi)**4)
		elif 'TT' in flavor: 
			# print(flavor, 'TT')
			prefactor *= -(kvar.z*(1-kvar.z))/(2*(np.pi**4)*w2)
		elif 'LT' in flavor: 
			# print(flavor, 'LT')
			prefactor *= -((kvar.z*(1-kvar.z))**1.5)/(np.sqrt(2)*(np.pi**4)*w2)

		if flavor == 'A_TT':
			Qu_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qu')
			Qd_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qd')
			Qs_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qs')
			G2_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')

			a_TT = prefactor*N_11*((1 - 2*kvar.z)**2)*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			a_TT += prefactor*N_11*2*(kvar.z**2 + (1-kvar.z)**2)*self.Zfsq*G2_11
			return a_TT

		elif flavor == 'B_TT': 
			Qu_11, Qu_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qu')
			Qd_11, Qd_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qd')
			Qs_11, Qs_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'Qs')
			G2_11, G2_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'G2')
			I3u_11, I3u_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3u')
			I3d_11, I3d_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3d')
			I3s_11, I3s_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'I3s')
			I4_11, I4_21_10, I4_10_01 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0], [1,0,0,1]], 'I4')
			I5_11, I5_10_01, I5_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [1,0,0,1], [0,1,1,0]], 'I5')
			N_11, N_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'N')

			b_TT = 0.5*((1-2*kvar.z)**2)*N_01_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_TT += self.Zfsq*(kvar.z**2 + (1-kvar.z)**2)*N_01_10*G2_11 
			b_TT +=  N_11*(self.Zusq*(0.5*Qu_11 + I3u_11 - I3u_01_10) + self.Zdsq*(0.5*Qd_11 + I3d_11 - I3d_01_10) + self.Zssq*(0.5*Qs_11 + I3s_11 - I3s_01_10))
			b_TT += (kvar.z**2 + (1-kvar.z)**2)*N_11*(self.Zusq*Qu_21_10 + self.Zdsq*Qd_21_10 + self.Zssq*Qs_21_10) 
			b_TT += (kvar.z**2 + (1-kvar.z)**2)*N_11*self.Zfsq*(2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10) 
			return prefactor*(1-(2*kvar.z))*b_TT

		elif flavor == 'A_LT': 
			Qu_11, Qu_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qu')
			Qd_11, Qd_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qd')
			Qs_11, Qs_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qs')
			G2_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11, N_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'N')
			
			a_LT = (1- 2*kvar.z)*N_00*2*self.Zfsq*G2_11
			a_LT -= (1- 2*kvar.z)*N_00*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			a_LT -= (1- 2*kvar.z)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			return prefactor*a_LT

		elif flavor == 'B_LT':
			Qu_00, Qu_11, Qu_01_10, Qu_10_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qu')
			Qd_00, Qd_11, Qd_01_10, Qd_10_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qd')
			Qs_00, Qs_11, Qs_01_10, Qs_10_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0], [0,1,1,0], [1,0,1,0]], 'Qs')
			G2_11, G2_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'G2')
			I3u_11, I3u_01_10, I3u_10_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3u')
			I3d_11, I3d_01_10, I3d_10_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3d')
			I3s_11, I3s_01_10, I3s_10_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'I3s')
			I4_11, I4_21_10, I4_10_01 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0], [1,0,0,1]], 'I4')
			I5_11, I5_10_01, I5_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [1,0,0,1], [0,1,1,0]], 'I5')
			N_00, N_11, N_10_10, N_01_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0], [1,0,1,0], [0,1,1,0]], 'N')

			b_LT = N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11 - self.Zusq*I3u_01_10 - self.Zdsq*I3d_01_10 - self.Zssq*I3s_01_10)
			b_LT += N_11*(self.Zusq*I3u_10_10 + self.Zdsq*I3d_10_10 + self.Zssq*I3s_10_10)
			b_LT += (kvar.z**2 + (1-kvar.z)**2)*N_00*(self.Zusq*Qu_01_10 - self.Zusq*Qu_11 + self.Zdsq*Qd_01_10 - self.Zdsq*Qd_11 + self.Zssq*Qs_01_10 - self.Zssq*Qs_11)
			b_LT -= (kvar.z**2 + (1-kvar.z)**2)*N_11*(self.Zusq*Qu_10_10 + self.Zdsq*Qd_10_10 + self.Zssq*Qs_10_10)
			b_LT -= 0.5*((1- 2*kvar.z)**2)*N_10_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_LT -= 0.5*((1- 2*kvar.z)**2)*(N_11 - N_01_10)*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			b_LT += ((1- 2*kvar.z)**2)*self.Zfsq*N_00*(3*G2_11 - 2*G2_01_10 + I4_21_10 + I4_10_01 - I5_11 + I5_01_10 + I5_10_01)
			b_LT += ((1- 2*kvar.z)**2)*self.Zfsq*N_10_10*G2_11
			
			return prefactor*b_LT

		elif flavor == 'C_LT': 
			Qu_00, Qu_11 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qu')
			Qd_00, Qd_11 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qd')
			Qs_00, Qs_11 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'Qs')
			G2_11, G2_01_10, G2_10_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0], [1,0,1,0]], 'G2')
			I3u_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'I3u')
			I3d_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'I3d')
			I3s_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'I3s')
			I4_11, I4_10_01, I4_10_10, I4_01_10, I4_11_20, I4_00_11, I4_11_11, I4_00_20 = self.fourier_bessel(kvar, [[1,1,0,0], [1,0,0,1], [1,0,1,0], [0,1,1,0], [1,1,2,0], [0,0,1,1], [1,1,1,1], [0,0,2,0]], 'I4')
			I5_11, I5_11_20, I5_10_01, I5_00_11, I5_11_11, I5_10_10, I5_00_20 = self.fourier_bessel(kvar, [[1,1,0,0], [1,1,2,0], [1,0,0,1], [0,0,1,1], [1,1,1,1], [1,0,1,0], [0,0,2,0]], 'I5')
			N_00, N_11 = self.fourier_bessel(kvar, [[0,0,0,0], [1,1,0,0]], 'N')

			c_LT = (kvar.z**2 + (1-kvar.z)**2)*N_11*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			c_LT -= N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11)
			c_LT += 0.5*((1- 2*kvar.z)**2)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			c_LT += ((1- 2*kvar.z)**2)*N_00*self.Zfsq*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
			c_LT += ((1- 2*kvar.z)**2)*N_11*self.Zfsq*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)
			
			return prefactor*c_LT

		elif flavor == 'A_TT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TT_unpolar':
			N_11, N_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'N')
			return prefactor*2*(kvar.z-0.5)*(2*(N_11**2) - N_11*N_21_10)

		elif flavor == 'A_LL_unpolar':
			N_00 = self.fourier_bessel(kvar, [[0,0,0,0]], 'N')
			return prefactor*(N_00**2)

		elif flavor == 'B_LL_unpolar':
			N_00, N_10_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,0,1,0]], 'N')
			return -prefactor*2*(kvar.z-0.5)*N_00*N_10_10

		elif flavor == 'A_TmT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TmT_unpolar':
			N_11, N_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'N')
			return prefactor*(kvar.z-0.5)*(-2*(N_11**2) + N_11*N_01_10)

		elif flavor == 'C_TmT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return -2*prefactor*(kvar.z-0.5)*(N_11**2)

		else:
			print('requested coefficient', flavor, 'does not exist')


	# returns xsec in units of femptobarns (fb)
	def get_xsec(self, kvar, kind):

		xsec_prefactor = self.alpha_em/(4*(np.pi**2)*(kvar.Q**2))
		xsec_prefactor *= (0.25*kvar.pT*kvar.y)/(kvar.x*kvar.z*(1-kvar.z))
		self.filter_dipole(kvar)

		if kind == 'DSA':
			tt_term = (2-kvar.y)*(self.get_coeff('A_TT', kvar) + (kvar.delta/kvar.pT)*np.cos(kvar.phi_Dp)*self.get_coeff('B_TT', kvar))
			lt_term = np.sqrt(2-2*kvar.y)*(np.cos(kvar.phi_kp)*self.get_coeff('A_LT', kvar) + (kvar.delta/kvar.pT)*np.cos(kvar.phi_Dp)*np.cos(kvar.phi_kp)*self.get_coeff('B_LT', kvar) + (kvar.delta/kvar.pT)*np.sin(kvar.phi_Dp)*np.sin(kvar.phi_kp)*self.get_coeff('C_LT', kvar))
			xsec = xsec_prefactor*(tt_term + lt_term)
			# xsec *= 0.3894*(10**12) # convert to fb 
			xsec *= 0.3894*(10**9) # convert to pb

			# print(tt_term, lt_term)
			return xsec

		elif kind == 'unpolarized':
			tt_term = (1 + (1-kvar.y)**2)*(self.get_coeff('A_TT_unpolar', kvar) + (kvar.delta/kvar.pT)*np.cos(kvar.phi_Dp)*self.get_coeff('B_TT_unpolar', kvar))
			tmt_term = -2*(1-kvar.y)*(np.cos(2*kvar.phi_kp)*self.get_coeff('A_TmT_unpolar', kvar) + (kvar.delta/kvar.pT)*np.cos(kvar.phi_Dp)*np.cos(2*kvar.phi_kp)*self.get_coeff('B_TmT_unpolar', kvar) + (kvar.delta/kvar.pT)*np.sin(kvar.phi_Dp)*np.sin(2*kvar.phi_kp)*self.get_coeff('C_TmT_unpolar', kvar))
			ll_term = 4*(1-kvar.y)*(self.get_coeff('A_LL_unpolar', kvar) + (kvar.delta/kvar.pT)*np.cos(kvar.phi_Dp)*self.get_coeff('B_LL_unpolar', kvar))
			
			xsec = (1/kvar.y)*xsec_prefactor*(tt_term + tmt_term + ll_term)
			# xsec *= 0.3894*(10**12) # convert to fb 
			xsec *= 0.3894*(10**9) # convert to pb

			# print(tt_term ,tmt_term, ll_term)
			return xsec


		# return unpolarized xsec integrated over azimuthal angle of electron (and \phi_{Dp} \approx 0)
		elif kind == 'unpolarized_integrated':
			tt_term = (1 + (1-kvar.y)**2)*(self.get_coeff('A_TT_unpolar', kvar) + (kvar.delta/kvar.pT)*self.get_coeff('B_TT_unpolar', kvar))
			ll_term = 4*(1-kvar.y)*(self.get_coeff('A_LL_unpolar', kvar) + (kvar.delta/kvar.pT)*self.get_coeff('B_LL_unpolar', kvar))
			
			xsec_prefactor *= (2*np.pi)**2
			xsec = (1/kvar.y)*xsec_prefactor*(tt_term + ll_term)
			# xsec *= 0.3894*(10**12) # convert to fb
			xsec *= 0.3894*(10**9) # convert to pb

			# print(tt_term ,tmt_term, ll_term)
			return xsec


	# returns dxsec/dQ^2 (integrates over x, z, pT, and t)
	def dxsec_dQ2(self, fixed_vars, x_range, z_range, pT_range, t_range):

		npoints = 10
		npoints_x = npoints
		npoints_z = npoints
		npoints_pT = npoints
		npoints_t = npoints

		dx = (x_range[1]-x_range[0])/npoints_x
		dz = (z_range[1]-z_range[0])/npoints_z
		dpT = (pT_range[1]-pT_range[0])/npoints_pT
		dt = (t_range[1]-t_range[0])/npoints_t

		x_values = np.arange(x_range[0], x_range[1], dx)	
		z_values = np.arange(z_range[0], z_range[1], dz)	
		pT_values = np.arange(pT_range[0], pT_range[1], dpT)	
		t_values = np.arange(t_range[0], t_range[1], dt)

		xsec = 0 
		for x in x_values:
			# print('x:',x)
			for z in z_values:
				for pT in pT_values:
					for t in t_values:

						fixed_vars['y'] = (fixed_vars['Q']**2)/(fixed_vars['s']*x)
						if fixed_vars['y'] > 1: continue

						fixed_vars['x'] = x
						fixed_vars['z'] = z
						fixed_vars['pT'] = pT
						fixed_vars['delta'] = np.sqrt(t)



						xsec += dx*dz*dpT*dt*self.get_xsec(fixed_vars, 'unpolarized_integrated')
		return xsec


	# returns dxsec/dpT (integrates over x, z, Q, and t)
	def dxsec_dpT(self, fixed_vars, x_range, z_range, Q_range, t_range):

		npoints = 10
		npoints_x = npoints
		npoints_z = npoints
		npoints_Q = npoints
		npoints_t = npoints

		dx = (x_range[1]-x_range[0])/npoints_x
		dz = (z_range[1]-z_range[0])/npoints_z
		dQ = (Q_range[1]-Q_range[0])/npoints_Q
		dt = (t_range[1]-t_range[0])/npoints_t

		x_values = np.arange(x_range[0], x_range[1], dx)	
		z_values = np.arange(z_range[0], z_range[1], dz)	
		Q_values = np.arange(Q_range[0], Q_range[1], dQ)
		t_values = np.arange(t_range[0], t_range[1], dt)

		xsec = 0 
		for x in x_values:
			# print('x:',x)
			for z in z_values:
				for Q in Q_values:
					for t in t_values:
						# print(x, z, Q, t)

						fixed_vars['y'] = (Q**2)/(fixed_vars['s']*x)
						if fixed_vars['y'] > 1: continue

						fixed_vars['x'] = x
						fixed_vars['z'] = z
						fixed_vars['Q'] = Q
						fixed_vars['delta'] = np.sqrt(t)



						xsec += dx*dz*dQ*dt*self.get_xsec(fixed_vars, 'unpolarized_integrated')
		return xsec


	# returns dxsec/dt (integrates over x, z, Q, and pT)
	def dxsec_dt(self, fixed_vars, x_range, z_range, Q_range, pT_range):

		npoints = 10
		npoints_x = npoints
		npoints_z = npoints
		npoints_Q = npoints
		npoints_pT = npoints

		dx = (x_range[1]-x_range[0])/npoints_x
		dz = (z_range[1]-z_range[0])/npoints_z
		dQ = (Q_range[1]-Q_range[0])/npoints_Q
		dpT = (pT_range[1]-pT_range[0])/npoints_pT

		x_values = np.arange(x_range[0], x_range[1], dx)	
		z_values = np.arange(z_range[0], z_range[1], dz)	
		Q_values = np.arange(Q_range[0], Q_range[1], dQ)
		pT_values = np.arange(pT_range[0], pT_range[1], dpT)

		xsec = 0 
		for x in x_values:
			# print('x:',x)
			for z in z_values:
				for Q in Q_values:
					for pT in pT_values:
						# print(x, z, Q, t)

						fixed_vars['y'] = (Q**2)/(fixed_vars['s']*x)
						if fixed_vars['y'] > 1: continue

						fixed_vars['x'] = x
						fixed_vars['z'] = z
						fixed_vars['Q'] = Q
						fixed_vars['pT'] = pT

						xsec += dx*dz*dQ*dpT*self.get_xsec(fixed_vars, 'unpolarized_integrated')
		return xsec






	# returns correlations of asymmetry
	def get_correlation(self, kvar, kind):
		self.filter_dipole(kvar)

		if kind == '<1>': num = kvar.y*(2-kvar.y)*self.get_coeff('A_TT', kvar)
		elif kind == '<cos(phi_Dp)>': num = 0.5*kvar.y*(2-kvar.y)*(kvar.delta/kvar.pT)*self.get_coeff('B_TT', kvar)
		elif kind == '<cos(phi_kp)>': num = 0.5*kvar.y*np.sqrt(2-2*kvar.y)*self.get_coeff('A_LT', kvar)
		elif kind == '<cos(phi_Dp)cos(phi_kp)>': num = 0.25*kvar.y*np.sqrt(2-2*kvar.y)*(kvar.delta/kvar.pT)*self.get_coeff('B_LT', kvar)
		elif kind == '<sin(phi_Dp)sin(phi_kp)>': num = 0.25*kvar.y*np.sqrt(2-2*kvar.y)*(kvar.delta/kvar.pT)*self.get_coeff('C_LT', kvar)
		else: raise ValueError(f'Error: Correlation {kind} not recognized')

		den = (1 + (1-kvar.y)**2)*self.get_coeff('A_TT_unpolar', kvar) + 4*(1-kvar.y)*self.get_coeff('A_LL_unpolar', kvar)
		return num/den


	# filter unpolarized dipole before double bessel to save time
	def filter_dipole(self, kvar): 

		closest_y = self.ndipole['y'][np.isclose(self.ndipole['y'], np.log(1/kvar.x), atol=0.01)]
		if closest_y.empty: raise ValueError('Requested rapdity, Y=', np.log(1/kvar.x), ', does not exist in the data file')
		else: closest_y = closest_y.iloc[0]
		self.f_ndipole = self.ndipole[np.isclose(self.ndipole['y'], closest_y, atol=0.001)]



	def get_g1(self, kvar):

		g1 = 0.5*(self.Zusq*self.get_ppdfplus(kvar, 'u') + self.Zdsq*self.get_ppdfplus(kvar, 'd') + self.Zssq*self.get_ppdfplus(kvar, 's'))
		return g1


	# # daniel's version 
	def get_ppdfplus(self, kvar, flavor):
		G = self.pdipoles[f'Q{flavor}'] + 2*self.pdipoles['G2']
		x, Q2 = kvar.x, kvar.Q**2

		lam2smx = 1
		Nc = 3.0
		x0 = 1.0
		eta0 = np.sqrt(Nc/(2*np.pi))*np.log(1/x0)
		eta0index = int(np.ceil(eta0/self.deta))+1

		eta = np.sqrt(Nc/(2*np.pi))*np.log(1/x)

		jetai = int(np.ceil(np.sqrt(Nc/(2.*np.pi))*np.log(1./x)/self.deta))
		jetaf = int(np.ceil(np.sqrt(Nc/(2.*np.pi))*np.log(Q2/(x*lam2smx))/self.deta))

		g1plus=0
		
		for j in range(eta0index,jetai):
			si = 0
			sf = int(np.ceil(round((self.deta*(j)-eta0)/self.deta,6))) #1 + np.int(np.ceil((deta*(j-1)-eta0)/deta))
			
			for i in range(si,sf): 
				g1plus = g1plus + self.deta*self.deta * G[i,j]

		for j in range(jetai,jetaf):
			si = int(np.ceil((self.deta*(j)-np.sqrt(Nc/(2*np.pi))*np.log(1./x))/self.deta))
			sf = int(np.ceil(round((self.deta*(j)-eta0)/self.deta,6))) #1 + np.int(np.ceil((deta*(j-1)-eta0)/deta))
			for i in range(si,sf): 
				g1plus = g1plus + self.deta*self.deta * G[i,j]

		g1plus = (-1./(np.pi**2)) * g1plus
		
		return g1plus



	def get_DeltaSigma(self,kvar): # returns \sum_f(delta q_f + delta qb_f)(x)
		DelSigma = self.get_ppdfplus(kvar, 'u') + self.get_ppdfplus(kvar, 'd') + self.get_ppdfplus(kvar, 's')
		return DelSigma


	def get_dipole_value(self, i,j,amp):
		return self.pdipoles[amp][i,j]

	def save_dipole(self, amp, filename):
		save(self.pdipoles[amp], filename)


	def alpha_s(self, running, s0, s):
		Nc=3.
		Nf=3.
		beta2 = (11*Nc-2*Nf)/(12*np.pi) 
		if running==False: return 0.3
		elif running==True: return (np.sqrt(Nc/(2*np.pi))/beta2)*(1/(s0+s))


	def get_DeltaG(self,kvar):
		s10 = np.sqrt(3.0/(2*np.pi))*np.log((kvar.Q**2)/(self.lambdaIR**2))
		s10_index = int(np.ceil(s10/self.deta))
		eta_index = int(np.ceil(np.sqrt(3.0/(2*np.pi))*np.log((kvar.Q**2)/(kvar.x*(self.lambdaIR**2)))/self.deta))
		Del_G = (1/self.alpha_s(True, 0, s10))*((2*3.0)/(np.pi**2))*self.pdipoles['G2'][s10_index, eta_index]
		return Del_G




if __name__ == '__main__':

	test_kins = {'Q': 8, 'z': 0.4, 'x': 0.01, 's': 100**2, 'delta':0.2, 'phi_Dp':0.0, 'phi_kp':0.0}
	test_kins['y'] = (test_kins['Q']**2)/(test_kins['s']*test_kins['x'])

	w2 = (test_kins['Q']**2)*((1/test_kins['x']) - 1)

	test_kins['pT'] = 5.0
	dj = DIJET(1)

	# print('DSA', dj.get_xsec(test_kins, 'DSA'))
	# print('Total', dj.get_xsec(test_kins, 'unpolarized'))
	# print('Total', dj.get_xsec(test_kins, 'unpolarized_integrated'))


	t_range = [0, 0.5]
	x_range = [0.005, 0.01]
	pT_range = [5, 15] 
	z_range = [0.3, 0.7]
	Q_range = [5, 10]

	print(dj.dxsec_dQ2(test_kins, x_range, z_range, pT_range, t_range))
	print(dj.dxsec_dpT(test_kins, x_range, z_range, Q_range, t_range))



		
