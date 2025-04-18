import numpy as np
import pandas as pd
import time
from scipy.special import jv, kv
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

import _pickle as cPickle
import sys
import zlib


def load(name): 
  compressed=open(name,"rb").read()
  data=cPickle.loads(zlib.decompress(compressed))
  return data


def save(data,name):  
  compressed=zlib.compress(cPickle.dumps(data))
  f=open(name,"wb")
  try:
	  f.writelines(compressed)
  except:
	  f.write(compressed)
  f.close()


class DIJET:
	def __init__(self, nreplica=1, lambdaIR=0.3, deta=0.05): 
		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Zusq = (2.0/3.0)**2
		self.Zdsq = (-1.0/3.0)**2
		self.Zssq = (-1.0/3.0)**2
		self.Zfsq = self.Zusq + self.Zdsq + self.Zssq 
		self.Nf = 3.0
		self.lambdaIR = lambdaIR

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

					# print(iamp, jamp, jbasis, ic_params[jamp][jbasis])

					self.basis_dipoles[iamp][jamp][jbasis] = load(f'{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')

					# print(iamp, jamp, jbasis, temp_dipole.shape)

					# if j == 0: input_dipole = ic_params[jamp][jbasis]*temp_dipole
					# else: 
					# input_dipole += ic_params[jamp][jbasis]*temp_dipole

			# input_dipole = np.sum(np.stack(input_data, axis=0), axis=0)
			# assert len(input_data) == 33, f'number of input files {len(input_data)} wrong'
			# self.pdipoles[iamp] = input_dipole


	def set_params(self, nreplica=1):

		current_dir = os.path.dirname(os.path.abspath(__file__))

		#--load unpolarized dipole amplitude
		# mv parameters from 0902.1112
		unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_ymax9.21_AAMS09.dat'
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

		# ic_file = current_dir + '/dipoles/mc_ICs_random_fit.json'
		# # ic_file = current_dir + '/dipoles/mc_ICs_random_fit_oldevo.json'

		# with open(ic_file, "r") as file:	
		# 	ics = json.load(file)


		# load polarized dipole amplitudes 
		# deta_str = 'd'+str(self.deta)[2:]
		# polar_indir = current_dir + f'/dipoles/{deta_str}-etamax15-rc/'
		# polar_indir = current_dir + f'/dipoles/correct/'

		# print(polar_indir)

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



	# def load_dipoles(self, nreplica=1):

		# # path to data
		# current_dir = os.path.dirname(os.path.abspath(__file__))

		# #--load unpolarized dipole amplitude
		# # mv parameters from 0902.1112
		# unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_ymax9.21_AAMS09.dat'
		# self.normN = 0.5*32.77*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# # mv parameters from 2311.10491
		# # unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_ymax9.21_CKM24.dat'
		# # self.normN = 13.9*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# self.ndipole = pd.read_csv(unpolar_input_file, sep=r'\s+', header=None, names=['y', 'ln(r)', 'N'])


		# # unpolar_input_file = current_dir + '/dipoles/n_ymin4.61_logrmin-13.8.dat'
		# # self.ndipole = np.loadtxt(unpolar_input_file)


		# # load initial condition parameters
		# fdf = pd.read_csv('../dsa_mc/dipoles/replica_params_old.csv')
		# header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]
		# fdf = fdf.dropna(axis=1, how='all')
		# fdf.columns = header
		# sdf = fdf[fdf['nrep'] == nreplica]
		# assert len(sdf) == 1, 'Error: more than 1 replica selected...'

		# ic_params = {}

		# for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
		# # for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2']:
		# 	ic_params[amp] = {}
		# 	for basis in ['eta','s10','1']:
		# 		if amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2']:
		# 			ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]
		# 		else:
		# 			ic_params[amp][basis] = random.uniform(-10, 10)

		# # ic_file = current_dir + '/dipoles/mc_ICs_random_fit.json'
		# # # ic_file = current_dir + '/dipoles/mc_ICs_random_fit_oldevo.json'

		# # with open(ic_file, "r") as file:	
		# # 	ics = json.load(file)


		# # load polarized dipole amplitudes 
		# deta_str = 'd'+str(self.deta)[2:]
		# polar_indir = current_dir + f'/dipoles/{deta_str}-etamax15-rc/'
		# # polar_indir = current_dir + f'/dipoles/correct/'

		# # print(polar_indir)

		# self.pdipoles = {}
		# # amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5', 'ITu', 'ITd', 'ITs']
		# amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']
		# # amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2']

		# for iamp in amps:
		# 	if iamp in ['GT', 'QTu', 'QTd', 'QTs', 'I3T', 'ITu', 'ITd', 'ITs']: continue

		# 	input_dipole = 0
		# 	for j, jamp in enumerate(amps):
		# 		for jbasis in ['eta', 's10','1']:

		# 			# print(iamp, jamp, jbasis, ic_params[jamp][jbasis])

		# 			temp_dipole = load(f'{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')

		# 			# print(iamp, jamp, jbasis, temp_dipole.shape)

		# 			# if j == 0: input_dipole = ic_params[jamp][jbasis]*temp_dipole
		# 			# else: 
		# 			input_dipole += ic_params[jamp][jbasis]*temp_dipole

		# 	# input_dipole = np.sum(np.stack(input_data, axis=0), axis=0)
		# 	# assert len(input_data) == 33, f'number of input files {len(input_data)} wrong'
		# 	self.pdipoles[iamp] = input_dipole


		# 	# non-basis
		# 	# tamp = amp
		# 	# if amp in ['Qu', 'Qd', 'Qs']: tamp = 'Q'
		# 	# elif amp in ['I3u', 'I3d', 'I3s']: tamp = 'I3'
		# 	# polar_amp = f'/Users/brandonmanley/Desktop/PhD/moment_evolution/evolved_dipoles/largeNc&Nf/d05_ones/d05_NcNf3_ones_{tamp}_rc.dat'
		# 	# self.pdipoles[amp] = np.loadtxt(polar_amp)



	def get_dipoles(self):
		return [self.ndipole, self.pdipoles]


	def get_filtered_dipole(self): 
		return self.f_ndipole


	def double_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):
		# self.filter_dipole(kvar)

		x, Q, z, pT = kvar['x'], kvar['Q'], kvar['z'], kvar['pT']
		Qeff = Q*np.sqrt(z*(1-z))

		results = []

		# unpolarized dipole
		if amp == 'N':

			# target_y_index = round((np.log(1/x) - 4.61)/0.01)
			# amp_values = 14*0.3894*self.ndipole[target_y_index] # 14 (mb) is value of \int d^2 b from fit in 2407.10581  (0.3894 is conversion to GeV^-2)

			# for i_a, i_b, i_c, i_d in indices_array:
			# 	c_term = (pT)**i_c
			# 	d_term = (Q*np.sqrt(z*(1-z)))**i_d
			# 	exp_term = np.exp(self.logr_values*(2+i_c+i_d))
			# 	Ja_term = jv(i_a, pT*np.exp(self.logr_values))
			# 	Kb_term = kv(i_b, Q*np.sqrt(z*(1-z))*np.exp(self.logr_values))

			# 	# Perform the sum
			# 	total_sum = 0.01*c_term*d_term*np.sum(Ja_term*Kb_term*amp_values)
			# 	results.append(prefactor*total_sum)

			u = self.f_ndipole['ln(r)'].to_numpy()
			amp_values = self.normN*self.f_ndipole['N'].to_numpy()  
			size = 0.01  # from evolution code
			amp_values = np.where(u > np.log(1/self.lambdaIR), 0, amp_values)

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

			for i_a, i_b, i_c, i_d in indices_array:
				pf = pT
				Qf = Qeff
				prefactor = (pf**i_c)*(Qf**i_d)
				exp_term = np.exp(u*(2+i_c+i_d))
				jv_term = jv(i_a, pf*np.exp(u))
				kv_term = kv(i_b, Qf*np.exp(u))
		
				# Perform the sum
				total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
				results.append(prefactor*total_sum)


			if len(indices_array) > 1: return results 
			else: return results[0]


		# polarized dipoles
		else:
			bas = np.sqrt(3/(2*np.pi))
			# prefactor *= (1/(2*bas))

			# photon-proton center of mass energy squared
			wsq = (Q**2)*((1/x) - 1)
			# wsq = kvar['s']

			target_eta_index = round((bas/self.deta)*np.log(wsq/(self.lambdaIR**2)))
			amp_values = self.pdipoles[amp][:, target_eta_index]
			u = -self.s10_values*(1/(2*bas))
			size = self.deta

			# if going to low-moderate Q^2, need to regulate IR
			if False: 
				amp_values = regulate_ir
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

	
			# Compute the Riemann sum in a vectorized way for each set of indices
			for i_a, i_b, i_c, i_d in indices_array:
				pf = pT/self.lambdaIR
				Qf = Qeff/self.lambdaIR
				prefactor = (1/(2*bas))*((1/self.lambdaIR)**2)*(pf**i_c)*(Qf**i_d)
				exp_term = np.exp(u*(2+i_c+i_d))
				jv_term = jv(i_a, pf*np.exp(u))
				kv_term = kv(i_b, Qf*np.exp(u))
		
				# Perform the sum
				total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
				results.append(prefactor*total_sum)

			if len(indices_array) > 1: return results 
			else: return results[0]



	# def double_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):

	# 	Q, pT, z, x, s = kvar['Q'], kvar['pT'], kvar['z'], kvar['x'], kvar['s']
	# 	prefactor = 1

	# 	# polarized dipoles
	# 	if amp != 'N':
	# 		bas = np.sqrt(3/(2*np.pi))
	# 		prefactor *= (1/(2*bas))

	# 		# photon-proton center of mass energy squared 
	# 		w2 = (Q**2)*((1/x) - 1)

	# 		# target_eta_index = round((bas/self.deta)*np.log((kvar['s']*kvar['y'])/(self.lambdaIR**2)))
	# 		target_eta_index = round((bas/self.deta)*np.log(w2/(self.lambdaIR**2)))
	# 		amp_values = self.pdipoles[amp][:, target_eta_index]
	# 		u = -self.s10_values*(1/(2*bas))
	# 		size = self.deta

	# 		if IR_reg[0] == 'gauss':
	# 			amp_values *= np.exp(-(np.exp(u)**2)*IR_reg[1])
	# 		elif IR_reg[0] == 'skin':
	# 			amp_values *= 1.0/(1 + np.exp(IR_reg[1]*((np.exp(u)/(self.lambdaIR*IR_reg[2])) - 1)))
	# 		elif IR_reg[0] == 'cut':
	# 			amp_values = np.where(s10 < IR_reg[1], 0, amp_values)
	# 		else:
	# 			pass 
	# 			# don't need line below since not using evolution where s_{10} < 0
	# 			# amp_values = np.where(s10 < 0, 0, amp_values)

	# 		# Compute the Riemann sum in a vectorized way for each set of indices
	# 		results = []
	# 		for i_a, i_b, i_c, i_d in indices_array:
	# 			pf = pT/self.lambdaIR
	# 			Qf = (Q*np.sqrt(z*(1-z)))/self.lambdaIR
	# 			prefactor = (pf**i_c)*(Qf**i_d)
	# 			exp_term = np.exp(u*(2+i_c+i_d))
	# 			jv_term = jv(i_a, pf*np.exp(u))
	# 			kv_term = kv(i_b, Qf*np.exp(u))
		
	# 			# Perform the sum
	# 			total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
	# 			results.append(prefactor*total_sum)

	# 		if len(indices_array) > 1: return results 
	# 		else: return results[0]


	# 	# unpolarized dipole
	# 	else:
	# 		# Extract and process columns
	# 		u = self.f_ndipole['ln(r)'].to_numpy()
	# 		amp_values = 14*(3.894*(10**5))*(self.f_ndipole['N'].to_numpy()) # 14 (mb) is value of \int d^2 b from fit in 2407.10581  
	# 		size = 0.01  # from evolution code
	# 		amp_values = np.where(u > np.log(1/self.lambdaIR), 0, amp_values)
	
	# 		# Compute the Riemann sum in a vectorized way for each set of indices
	# 		results = []
	# 		for i_a, i_b, i_c, i_d in indices_array:
	# 			pf = pT
	# 			Qf = (Q*np.sqrt(z*(1-z)))
	# 			prefactor = (pf**i_c)*(Qf**i_d)
	# 			exp_term = np.exp(u*(2+i_c+i_d))
	# 			jv_term = jv(i_a, pf*np.exp(u))
	# 			kv_term = kv(i_b, Qf*np.exp(u))
		
	# 			# Perform the sum
	# 			total_sum = size*np.sum(exp_term*jv_term*kv_term*amp_values)
	# 			results.append(prefactor*total_sum)

	# 		if len(indices_array) > 1: return results 
	# 		else: return results[0]



	def get_coeff(self, flavor, kvar):

		Zfsq = self.Zusq + self.Zdsq + self.Zssq
		prefactor = self.alpha_em*(self.Nc**2)*(kvar['Q']**2)
		w2 = (kvar['Q']**2)*((1/kvar['x']) - 1)
		# w2 = kvar['s']
		# print(w2)

		if 'TT_unpolar' in flavor: prefactor *= Zfsq*(4*(kvar['z']**2)*((1-kvar['z'])**2)*((kvar['z'])**2 + (1-kvar['z'])**2))/((2*np.pi)**4)
		elif 'LL_unpolar' in flavor: prefactor *= Zfsq*(8*(kvar['z']**3)*((1-kvar['z'])**3))/((2*np.pi)**4)
		elif 'TmT_unpolar' in flavor: prefactor *= Zfsq*(-8*(kvar['z']**3)*((1-kvar['z'])**3))/((2*np.pi)**4)
		elif 'TT' in flavor: 
			# print('flavor', flavor)
			prefactor *= -(kvar['z']*(1-kvar['z']))/(2*(np.pi**4)*w2)
		elif 'LT' in flavor: prefactor *= -((kvar['z']*(1-kvar['z']))**1.5)/(np.sqrt(2)*(np.pi**4)*w2)

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
			b_TT += Zfsq*(kvar['z']**2 + (1-kvar['z'])**2)*N_01_10*G2_11 + N_11*(0.5*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11) + (self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11) - (self.Zusq*I3u_01_10 + self.Zdsq*I3d_01_10 + self.Zssq*I3s_01_10)) 
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
			xsec *= 0.3894*(10**12) # convert to fb 

			# print(tt_term, lt_term)
			return xsec

		elif kind == 'unpolarized':
			tt_term = (1 + (1-kvar['y'])**2)*(self.get_coeff('A_TT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_TT_unpolar', kvar))
			tmt_term = -2*(1-kvar['y'])*(np.cos(2*kvar['phi_kp'])*self.get_coeff('A_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*np.cos(2*kvar['phi_kp'])*self.get_coeff('B_TmT_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.sin(kvar['phi_Dp'])*np.sin(2*kvar['phi_kp'])*self.get_coeff('C_TmT_unpolar', kvar))
			ll_term = 4*(1-kvar['y'])*(self.get_coeff('A_LL_unpolar', kvar) + (kvar['delta']/kvar['pT'])*np.cos(kvar['phi_Dp'])*self.get_coeff('B_LL_unpolar', kvar))
			
			xsec = (1/kvar['y'])*xsec_prefactor*(tt_term + tmt_term + ll_term)
			xsec *= 0.3894*(10**12) # convert to fb 

			# print(tt_term ,tmt_term, ll_term)
			return xsec


	# returns correlations of asymmetry
	def get_correlation(self, kvar, kind):
		self.filter_dipole(kvar)

		if kind == '<1>': num = kvar['y']*(2-kvar['y'])*self.get_coeff('A_TT', kvar)
		elif kind == '<cos(phi_Dp)>': num = 0.5*kvar['y']*(2-kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('B_TT', kvar)
		elif kind == '<cos(phi_kp)>': num = 0.5*kvar['y']*np.sqrt(2-2*kvar['y'])*self.get_coeff('A_LT', kvar)
		elif kind == '<cos(phi_Dp)cos(phi_kp)>': num = 0.25*kvar['y']*np.sqrt(2-2*kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('B_LT', kvar)
		elif kind == '<sin(phi_Dp)sin(phi_kp)>': num = 0.25*kvar['y']*np.sqrt(2-2*kvar['y'])*(kvar['delta']/kvar['pT'])*self.get_coeff('C_LT', kvar)
		else: raise ValueError(f'Error: Correlation {kind} not recognized')

		den = (1 + (1-kvar['y'])**2)*self.get_coeff('A_TT_unpolar', kvar) + 4*(1-kvar['y'])*self.get_coeff('A_LL_unpolar', kvar)
		return num/den


	# filter unpolarized dipole before double bessel to save time
	def filter_dipole(self, kvar): 

		closest_y = self.ndipole['y'][np.isclose(self.ndipole['y'], np.log(1/kvar['x']), atol=0.01)]
		if closest_y.empty: raise ValueError('Requested rapdity, Y=', np.log(1/x), ', does not exist in the data file')
		else: closest_y = closest_y.iloc[0]
		self.f_ndipole = self.ndipole[np.isclose(self.ndipole['y'], closest_y, atol=0.001)]



	# def get_ppdfplus(self, kvar, flavor):

	# 	eta_max = round((1/self.deta)*np.sqrt(3/(2*np.pi))*np.log((kvar['Q']**2)/(kvar['x']*(self.lambdaIR**2))))
	# 	s10_min = round((1/self.deta)*np.sqrt(3/(2*np.pi))*np.log(1/kvar['x']))

	# 	ppdf_plus = 0
	# 	for j in range(eta_max):
	# 		for i in range(max(0, j - s10_min), j):
	# 			ppdf_plus += (-1/(np.pi**2))*(self.deta**2)*(self.pdipoles[f'Q{flavor}'][i, j] + 2*self.pdipoles['G2'][i, j])

	# 	return ppdf_plus



	def get_g1(self, kvar):

		g1 = 0.5*(self.Zusq*self.get_ppdfplus(kvar, 'u') + self.Zdsq*self.get_ppdfplus(kvar, 'd') + self.Zssq*self.get_ppdfplus(kvar, 's'))
		return g1


	# # daniel's version 
	def get_ppdfplus(self, kvar, flavor):
		G = self.pdipoles[f'Q{flavor}'] + 2*self.pdipoles['G2']
		x, Q2 = kvar['x'], kvar['Q']**2

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
		if running==False:
			return 0.3
		elif running==True:
			return (np.sqrt(Nc/(2*np.pi))/beta2)*(1/(s0+s))


	def get_DeltaG(self,kvar):
		
		s10 = np.sqrt(3.0/(2*np.pi))*np.log((kvar['Q']**2)/(self.lambdaIR**2))
		s10_index = int(np.ceil(s10/self.deta))
		eta_index = int(np.ceil(np.sqrt(3.0/(2*np.pi))*np.log((kvar['Q']**2)/(kvar['x']*(self.lambdaIR**2)))/self.deta))
		
		Del_G = (1/self.alpha_s(True, 0, s10))*((2*3.0)/(np.pi**2))*self.pdipoles['G2'][s10_index, eta_index]
		
		return Del_G







# function to plot histograms
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






if __name__ == '__main__':

	test_kins = {'Q': 8, 'z': 0.4, 'x': 0.01, 's': 100**2, 'delta':0.2, 'phi_Dp':0.0, 'phi_kp':0.0}
	test_kins['y'] = (test_kins['Q']**2)/(test_kins['s']*test_kins['x'])

	w2 = (test_kins['Q']**2)*((1/test_kins['x']) - 1)

	test_kins['pT'] = 5.0
	dj = DIJET(1)

	print('DSA', dj.get_xsec(test_kins, 'DSA'))
	print('Total', dj.get_xsec(test_kins, 'unpolarized'))
	# print('Total', dj.get_xsec(test_kins, 'unpolarized_integrated'))


	# dj.save_dipole('Qu', 'test_Qu_brandon.dat')

	# print('DSA', dj.get_xsec(test_kins, 'DSA'))
	# print('Total', dj.get_xsec(test_kins, 'unpolarized'))
	# print('g1', dj.get_g1(test_kins))

	# print(dj.double_bessel(test_kins, [[0,0,0,0]], 'N'), dj.double_bessel(test_kins, [[0,0,0,0]], 'Qu')/w2)




		
