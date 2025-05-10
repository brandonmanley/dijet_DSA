import numpy as np
import pandas as pd
import time
from scipy.special import jv, kv
import json
import os
import random

import _pickle as cPickle
import sys
import zlib
from dataclasses import dataclass
from scipy.integrate import quad,fixed_quad, nquad

import vegas
from scipy.integrate import cubature
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from scipy.special import roots_jacobi


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


@dataclass
class Kinematics:
	s: float = 0.0
	Q: float = 0.0
	x: float = 0.0
	delta: float = 0.0
	pT: float = 0.0
	z: float = 0.0
	y: float = 0.0 
	phi_kp: float = 0.0
	phi_Dp: float = 0.0
	


class DIJET:
	def __init__(self, nreplica=1, lambdaIR=0.3, deta=0.05, gauss_param=0.0, mv_only=False, fix_moments=False, old_rap=False, corrected_evo=False, constrained_moments=False): 

		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Nf = 3.0
		self.Zusq = 4.0/9.0
		self.Zdsq = 1.0/9.0
		self.Zssq = 1.0/9.0
		self.Zfsq = self.Zusq + self.Zdsq + self.Zssq 

		# integration parameters
		self.lambdaIR = lambdaIR
		self.gauss_param = gauss_param
		self.deta = deta

		# testing conditionals
		self.mv_only = mv_only
		if mv_only: print('Using only initial conditions for N!')

		self.fix_moments = fix_moments 
		if fix_moments: print('Moment amplitude parameters are fixed')

		self.old_rap = old_rap
		if old_rap: print('Using ln(1/x) in argument of N!')

		self.corrected_evo = corrected_evo
		if corrected_evo: print('Using corrected evolution')

		self.constrained_moments = constrained_moments
		if constrained_moments: print('Using constrained moment parameters')

		self.current_dir = os.path.dirname(os.path.abspath(__file__))

		self.load_dipoles()
		self.load_params()
		self.set_params(nreplica)


	def load_dipoles(self):

		# load polarized dipole amplitudes 
		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = self.current_dir + f'/dipoles/{deta_str}-rc/'
		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']

		if self.corrected_evo: 
			self.deta = 0.08 # only one supported right now
			polar_indir = self.current_dir + f'/dipoles/d08-rc-corrected/'
			amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']

		self.basis_dipoles = {}

		bad_amps = ['GT', 'QTu', 'QTd', 'QTs', 'I3T', 'ITu', 'ITd', 'ITs']
		if self.corrected_evo: 
			bad_amps = ['GT', 'I3T', 'ITu', 'ITd', 'ITs']

		for iamp in amps:
			if iamp in bad_amps: continue

			self.basis_dipoles[iamp] = {}
			input_dipole = 0
			for j, jamp in enumerate(amps):

				self.basis_dipoles[iamp][jamp] = {}
				for jbasis in ['eta', 's10','1']:

					self.basis_dipoles[iamp][jamp][jbasis] = load(f'{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')


		#--load unpolarized dipole amplitude
		# mv parameters from 0902.1112
		# unpolar_input_file = self.current_dir + '/dipoles/narr_ymin4.61_ymax9.81_AAMS09.dat'

		unpolar_file = 'narr_ymin4.61_ymax14.91_AAMS09.dat'
		unpolar_input_file = self.current_dir + '/dipoles/' + unpolar_file
		self.normN = 0.5*32.77*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# mv parameters from 2311.10491 (NOT SUPPORTED YET)
		# unpolar_input_file = self.current_dir + '/dipoles/narr_ymin4.61_ymax9.81_CKM24.dat'
		# self.normN = 13.9*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# unpolar_input_file = self.current_dir + '/dipoles/narr_ymin4.61_ymax9.81_Qs2_0.1.dat'
		# unpolar_input_file = self.current_dir + '/dipoles/narr_ymin4.61_ymax9.81_Qs2_0.2.dat'
		# unpolar_input_file = self.current_dir + '/dipoles/narr_ymin4.61_ymax9.81_Qs2_0.3.dat'

		self.ndipole = np.loadtxt(unpolar_input_file)

		print('loaded N(r^2, s) data from', unpolar_file)
		print('loaded polarized amp data from', polar_indir)

		# grid values for dipoles
		n_psteps = len(self.basis_dipoles['Qu']['Qu']['eta'])
		self.s10_values = np.arange(0.0, self.deta*(n_psteps), self.deta)

		n_usteps = len(self.ndipole[0])
		self.dlogr = 0.01 # from evolution code
		# self.logr_values = np.arange(-13.8, n_usteps*self.dlogr - 13.8, self.dlogr)
		self.logr_values = np.arange(-13.8, round(np.log(1/self.lambdaIR), 2), self.dlogr)


	def load_params(self, params_file='replica_params_dis.csv'):

		# load initial condition parameters
		if params_file == 'replica_params_dis.csv':
			fdf = pd.read_csv(self.current_dir + f'/dipoles/{params_file}')
			header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]

		elif params_file == 'replica_params_pp.csv':
			fdf = pd.read_csv(self.current_dir + f'/dipoles/{params_file}')
			header = ['nrep', 'sol', 'chi'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]

		if self.corrected_evo:
			params_file = 'replica_params_dis_corrected.csv'
			fdf = pd.read_csv(self.current_dir + f'/dipoles/{params_file}')

			header = ['nrep']
			for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2', 'QTu', 'QTd', 'QTs']:
				for ib in ['eta', 's10', '1']:
					if ia in ['GT', 'QTu', 'QTd', 'QTs'] and ib in ['eta', '1']: continue
					header.append(f'{ia}{ib}')


		fdf = fdf.dropna(axis=1, how='all')
		fdf.columns = header
		self.params = fdf

		# load moment parameters (random)
		if 'dis' in params_file:
			mom_params_file = '/dipoles/moment_params_dis.csv'
		else:
			mom_params_file = '/dipoles/moment_params_pp.csv'

		if not self.constrained_moments: mom_params_file = '/dipoles/random_moment_params.csv'

		self.mom_params = pd.read_csv(self.current_dir + mom_params_file)

		print('loaded params from', params_file)
		print('loaded random moment params from', mom_params_file)


	def set_params(self, nreplica=1):

		
		sdf = self.params[self.params['nrep'] == nreplica]
		assert len(sdf) == 1, 'Error: more than 1 replica selected...'
		print('loaded replica', nreplica)

		smdf = self.mom_params[self.mom_params['nrep'] == nreplica]
		assert len(smdf) == 1, 'Error: more than 1 replica selected...'
		# print('loaded replica', nreplica)

		ic_params = {}

		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']
		if self.corrected_evo: 
			amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']

		for amp in amps:
			ic_params[amp] = {}
			for basis in ['eta','s10','1']:
				if amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'QTu', 'QTd', 'QTs']:

					if self.corrected_evo:

						if amp == 'GT':
							if basis == 'eta':
								ic_params[amp][basis] = -(0.5/self.Nc)*(sdf['Queta'].iloc[0] + sdf['Qus10'].iloc[0])
								ic_params[amp][basis] += -(0.5/self.Nc)*(sdf['Qdeta'].iloc[0] + sdf['Qds10'].iloc[0])
								ic_params[amp][basis] += -(0.5/self.Nc)*(sdf['Qseta'].iloc[0] + sdf['Qss10'].iloc[0])
								ic_params[amp][basis] -= sdf['GTs10'].iloc[0]

							elif basis == '1':
								ic_params[amp][basis] = -(0.5/self.Nc)*(sdf['Qu1'].iloc[0] + sdf['Qd1'].iloc[0] + sdf['Qs1'].iloc[0])

							else:
								ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]


						elif amp == 'QTu':
							if basis == 'eta':
								ic_params[amp][basis] = -2*(sdf['Queta'].iloc[0] + sdf['Qus10'].iloc[0]) - sdf['QTus10'].iloc[0]
							elif basis == '1':
								ic_params[amp][basis] = -2*sdf['Qu1'].iloc[0]
							else:
								ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]

						elif amp == 'QTd':
							if basis == 'eta':
								ic_params[amp][basis] = -2*(sdf['Qdeta'].iloc[0] + sdf['Qds10'].iloc[0]) - sdf['QTds10'].iloc[0]
							elif basis == '1':
								ic_params[amp][basis] = -2*sdf['Qd1'].iloc[0]
							else:
								ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]

						elif amp == 'QTs':
							if basis == 'eta':
								ic_params[amp][basis] = -2*(sdf['Qseta'].iloc[0] + sdf['Qss10'].iloc[0]) - sdf['QTss10'].iloc[0]
							elif basis == '1':
								ic_params[amp][basis] = -2*sdf['Qs1'].iloc[0]
							else:
								ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]

						else: 
							ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]

					else:
						ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]
				else: 
					if self.fix_moments: ic_params[amp][basis] = 1
					else: ic_params[amp][basis] = smdf[f'{amp}{basis}'].iloc[0]


		self.pdipoles = {}

		bad_amps = ['GT', 'QTu', 'QTd', 'QTs', 'I3T', 'ITu', 'ITd', 'ITs']
		if self.corrected_evo: 
			bad_amps = ['GT', 'I3T', 'ITu', 'ITd', 'ITs']

		for iamp in amps:
			if iamp in bad_amps: continue

			input_dipole = 0
			for j, jamp in enumerate(amps):
				for jbasis in ['eta', 's10','1']:
					temp_dipole = self.basis_dipoles[iamp][jamp][jbasis]
					input_dipole += ic_params[jamp][jbasis]*temp_dipole

			self.pdipoles[iamp] = input_dipole


	def set_temp_params(self, mom_params, nreplica=1):

		sdf = self.params[self.params['nrep'] == nreplica]
		assert len(sdf) == 1, 'Error: more than 1 replica selected...'
		# print('loaded replica', nreplica)

		ic_params = {}

		for amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']:
			ic_params[amp] = {}
			for basis in ['eta','s10','1']:
				if amp in ['Qu', 'Qd', 'Qs', 'GT', 'G2']: ic_params[amp][basis] = sdf[f'{amp}{basis}'].iloc[0]
				else: 
					if self.fix_moments: ic_params[amp][basis] = 0
					else: ic_params[amp][basis] = mom_params[amp][basis]


		self.pdipoles = {}
		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']
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


	def fourier_bessel(self, kvar, indices_array, amp, IR_reg = [None, 0]):

		x, Q, z, pT = kvar.x, kvar.Q, kvar.z, kvar.pT
		Qeff = Q*np.sqrt(z*(1-z))
		w2 = (Q**2)*((1/x) - 1)
		target_eta = np.log(w2/(self.lambdaIR**2))

		results = []

		# unpolarized dipole
		if amp == 'N':

			r = np.exp(self.logr_values)
			measure = self.dlogr*(r**2)

			maxr_index = len(self.logr_values)

			if self.old_rap:
				target_eta_index = round((1/self.dlogr)*(np.log(1/x) - 4.61))
			else:
				target_eta_index = round((1/self.dlogr)*(target_eta - 4.61))

			if self.mv_only: target_eta_index = 0

			if target_eta_index < 0: raise ValueError(f"requested Y={target_eta} does not exist in N")
			amp_values = self.normN*self.ndipole[target_eta_index, :maxr_index]


		# polarized dipoles
		else:

			bas = np.sqrt(3/(2*np.pi))
			r = (1/self.lambdaIR)*np.exp(-self.s10_values*(1/(2*bas)))
			measure = self.deta*(0.5/bas)*(r**2)

			target_eta_index = round((bas/self.deta)*target_eta)
			if target_eta_index < 0: raise ValueError(f"requested Y={target_eta} does not exist in {amp}")
			amp_values = self.pdipoles[amp][:, target_eta_index]


		# need to regulate IR for small-moderate Q^2 
		if self.gauss_param > 0.0:
			amp_values = regulate_IR(amp_values, r, ['gauss', self.gauss_param])

	
		# Compute the Riemann sum for each set of indices
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

		Q, x, z = kvar.Q, kvar.x, kvar.z
		prefactor = self.alpha_em*(self.Nc**2)*(Q**2)
		w2 = (Q**2)*((1/x) - 1)

		if 'TT_unpolar' in flavor: 
			# print(flavor, 'TT_unpolar')
			prefactor *= self.Zfsq*(4*(z**2)*((1-z)**2)*((z)**2 + (1-z)**2))/((2*np.pi)**4)
		elif 'LL_unpolar' in flavor: 
			# print(flavor, 'LL_unpolar')
			prefactor *= self.Zfsq*(8*(z**3)*((1-z)**3))/((2*np.pi)**4)
		elif 'TmT_unpolar' in flavor: 
			# print(flavor, 'TmT_unpolar')
			prefactor *= self.Zfsq*(-8*(z**3)*((1-z)**3))/((2*np.pi)**4)
		elif 'TT' in flavor: 
			# print(flavor, 'TT')
			prefactor *= -(z*(1-z))/(2*(np.pi**4)*w2)
		elif 'LT' in flavor: 
			# print(flavor, 'LT')
			prefactor *= -((z*(1-z))**1.5)/(np.sqrt(2)*(np.pi**4)*w2)

		if flavor == 'A_TT':
			Qu_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qu')
			Qd_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qd')
			Qs_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'Qs')
			G2_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')

			a_TT = prefactor*N_11*((1 - 2*z)**2)*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			a_TT += prefactor*N_11*2*(z**2 + (1-z)**2)*self.Zfsq*G2_11
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

			b_TT = 0.5*((1-2*z)**2)*N_01_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_TT += self.Zfsq*(z**2 + (1-z)**2)*N_01_10*G2_11 
			b_TT +=  N_11*(self.Zusq*(0.5*Qu_11 + I3u_11 - I3u_01_10) + self.Zdsq*(0.5*Qd_11 + I3d_11 - I3d_01_10) + self.Zssq*(0.5*Qs_11 + I3s_11 - I3s_01_10))
			b_TT += (z**2 + (1-z)**2)*N_11*(self.Zusq*Qu_21_10 + self.Zdsq*Qd_21_10 + self.Zssq*Qs_21_10) 
			b_TT += (z**2 + (1-z)**2)*N_11*self.Zfsq*(2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10) 
			return prefactor*(1-(2*z))*b_TT

		elif flavor == 'A_LT': 
			Qu_11, Qu_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qu')
			Qd_11, Qd_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qd')
			Qs_11, Qs_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'Qs')
			G2_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'G2')
			N_11, N_00 = self.fourier_bessel(kvar, [[1,1,0,0], [0,0,0,0]], 'N')
			
			a_LT = (1- 2*z)*N_00*2*self.Zfsq*G2_11
			a_LT -= (1- 2*z)*N_00*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			a_LT -= (1- 2*z)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
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
			b_LT += (z**2 + (1-z)**2)*N_00*(self.Zusq*Qu_01_10 - self.Zusq*Qu_11 + self.Zdsq*Qd_01_10 - self.Zdsq*Qd_11 + self.Zssq*Qs_01_10 - self.Zssq*Qs_11)
			b_LT -= (z**2 + (1-z)**2)*N_11*(self.Zusq*Qu_10_10 + self.Zdsq*Qd_10_10 + self.Zssq*Qs_10_10)
			b_LT -= 0.5*((1- 2*z)**2)*N_10_10*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			b_LT -= 0.5*((1- 2*z)**2)*(N_11 - N_01_10)*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			b_LT += ((1- 2*z)**2)*self.Zfsq*N_00*(3*G2_11 - 2*G2_01_10 + I4_21_10 + I4_10_01 - I5_11 + I5_01_10 + I5_10_01)
			b_LT += ((1- 2*z)**2)*self.Zfsq*N_10_10*G2_11
			
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

			c_LT = (z**2 + (1-z)**2)*N_11*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			c_LT -= N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11)
			c_LT += 0.5*((1- 2*z)**2)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			c_LT += ((1- 2*z)**2)*N_00*self.Zfsq*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + 2*I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
			c_LT += ((1- 2*z)**2)*N_11*self.Zfsq*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)
			
			return prefactor*c_LT

		elif flavor == 'A_TT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TT_unpolar':
			N_11, N_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'N')
			return prefactor*2*(z-0.5)*(2*(N_11**2) - N_11*N_21_10)

		elif flavor == 'A_LL_unpolar':
			N_00 = self.fourier_bessel(kvar, [[0,0,0,0]], 'N')
			return prefactor*(N_00**2)

		elif flavor == 'B_LL_unpolar':
			N_00, N_10_10 = self.fourier_bessel(kvar, [[0,0,0,0], [1,0,1,0]], 'N')
			return -prefactor*2*(z-0.5)*N_00*N_10_10

		elif flavor == 'A_TmT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TmT_unpolar':
			N_11, N_01_10 = self.fourier_bessel(kvar, [[1,1,0,0], [0,1,1,0]], 'N')
			return prefactor*(z-0.5)*(-2*(N_11**2) + N_11*N_01_10)

		elif flavor == 'C_TmT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return -2*prefactor*(z-0.5)*(N_11**2)

		else:
			print('requested coefficient', flavor, 'does not exist')


	# # returns xsec in units of picobarns (pb) (differential in Q^2, x, \phi, p_T, z, t, \phi_p, \phi_\Delta)
	# def get_xsec(self, kvar, kind, diff):

	# 	Q, x, y, z, pT, delta = kvar.Q, kvar.x, kvar.y, kvar.z, kvar.pT, kvar.delta
	# 	phi_Dp, phi_kp = kvar.phi_Dp, kvar.phi_kp


	# 	if kind == 'DSA':
	# 		dsa_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2))

	# 		tt_term = (2-y)*(self.get_coeff('A_TT', kvar) + (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT', kvar))
	# 		lt_term = np.sqrt(2-2*y)*(np.cos(phi_kp)*self.get_coeff('A_LT', kvar) + (delta/pT)*np.cos(phi_Dp)*np.cos(phi_kp)*self.get_coeff('B_LT', kvar) + (delta/pT)*np.sin(phi_Dp)*np.sin(phi_kp)*self.get_coeff('C_LT', kvar))
			
	# 		if diff == 'dx':
	# 			dsa_prefactor *= (0.25*pT*y)/(x*z*(1-z))
	# 		else: 
	# 			dsa_prefactor *= (0.25*pT)/(z*(1-z))

	# 		xsec = dsa_prefactor*(tt_term + lt_term)
	# 		xsec *= 0.3894*(10**12) # convert to fb 
	# 		# xsec *= 0.3894*(10**9) # convert to pb

	# 		# print(tt_term, lt_term)
	# 		return xsec

	# 	elif kind == 'unpolarized':
	# 		unp_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2)*y)

	# 		tt_term = (1 + (1-y)**2)*(self.get_coeff('A_TT_unpolar', kvar) + (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT_unpolar', kvar))
	# 		tmt_term = -2*(1-y)*(np.cos(2*phi_kp)*self.get_coeff('A_TmT_unpolar', kvar) + (delta/pT)*np.cos(phi_Dp)*np.cos(2*phi_kp)*self.get_coeff('B_TmT_unpolar', kvar) + (delta/pT)*np.sin(phi_Dp)*np.sin(2*phi_kp)*self.get_coeff('C_TmT_unpolar', kvar))
	# 		ll_term = 4*(1-y)*(self.get_coeff('A_LL_unpolar', kvar) + (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_LL_unpolar', kvar))
			
	# 		if diff == 'dx':
	# 			unp_prefactor *= (0.25*pT*y)/(x*z*(1-z))
	# 		else: 
	# 			unp_prefactor *= (0.25*pT)/(z*(1-z))

	# 		xsec = unp_prefactor*(tt_term + tmt_term + ll_term)
	# 		xsec *= 0.3894*(10**12) # convert to fb 
	# 		# xsec *= 0.3894*(10**9) # convert to pb

	# 		# print(tt_term ,tmt_term, ll_term)
	# 		return xsec


	# 	# return unpolarized xsec integrated over azimuthal angles of electron and jets
	# 	elif kind == 'unpolarized_integrated':

	# 		tt_term = (1 + (1-y)**2)*self.get_coeff('A_TT_unpolar', kvar)
	# 		ll_term = 4*(1-y)*self.get_coeff('A_LL_unpolar', kvar)

	# 		if diff == 'dx':
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT)/(z * (1-z) * (Q**2) * x)
	# 		else: 
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT)/(z * (1-z) * (Q**2) * y)
			
	# 		xsec = unp_prefactor*(tt_term + ll_term)
	# 		# xsec *= 0.3894*(10**12) # convert to fb
	# 		xsec *= 0.3894*(10**9) # convert to pb

	# 		# print(tt_term ,tmt_term, ll_term)
	# 		return xsec



	# # returns xsec in units of picobarns (pb) (differential in Q^2, x, p_T, z, t)
	# def get_xsec_angle_integrated(self, kvar, kind, diff):

	# 	Q, x, y, z, pT, delta = kvar.Q, kvar.x, kvar.y, kvar.z, kvar.pT, kvar.delta


	# 	if kind == 'DSA':

	# 		tt_term = (2-y)*self.get_coeff('A_TT', kvar)
			
	# 		if diff == 'dx':
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT * y)/(z * (1-z) * (Q**2) * x)
	# 		else: 
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT)/(z * (1-z) * (Q**2))

	# 		xsec = dsa_prefactor*(tt_term)
	# 		# xsec *= 0.3894*(10**12) # convert to fb 
	# 		xsec *= 0.3894*(10**9) # convert to pb

	# 		return xsec


	# 	elif kind == 'unpolarized':

	# 		tt_term = (1 + (1-y)**2)*self.get_coeff('A_TT_unpolar', kvar)
	# 		ll_term = 4*(1-y)*self.get_coeff('A_LL_unpolar', kvar)

	# 		if diff == 'dx':
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT)/(z * (1-z) * (Q**2) * x)
	# 		else: 
	# 			unp_prefactor = (0.5 * self.alpha_em * np.pi * pT)/(z * (1-z) * (Q**2) * y)
			
	# 		xsec = unp_prefactor*(tt_term + ll_term)
	# 		# xsec *= 0.3894*(10**12) # convert to fb
	# 		xsec *= 0.3894*(10**9) # convert to pb

	# 		return xsec


	# returns numerator of asymmetry in pb or fb (differential in Q^2, x (or y), \phi, p_T, z, t, \phi_p, \phi_\Delta)
	def numerator(self, kinematics, diff='dx'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta
		phi_Dp, phi_kp = kinematics.phi_Dp, kinematics.phi_kp

		numerator_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2))
		if diff == 'dx': 
			numerator_prefactor *= (0.25*pT*y)/(x*z*(1-z))
		elif diff == 'dy':
			numerator_prefactor *= (0.25*pT)/(z*(1-z))
		else:
			raise ValueError('diff should be dx or dy')

		tt_term = (2-y) * self.get_coeff('A_TT', kinematics)
		tt_term += (2-y) * (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT', kinematics)

		lt_term = np.sqrt(2-2*y) * np.cos(phi_kp)*self.get_coeff('A_LT', kinematics) 
		lt_term += np.sqrt(2-2*y) * (delta/pT)*np.cos(phi_Dp)*np.cos(phi_kp)*self.get_coeff('B_LT', kinematics) 
		lt_term += np.sqrt(2-2*y) * (delta/pT)*np.sin(phi_Dp)*np.sin(phi_kp)*self.get_coeff('C_LT', kinematics)

		xsec = numerator_prefactor*(tt_term + lt_term)
		xsec *= 0.3894*(10**12) # convert to fb 
		# xsec *= 0.3894*(10**9) # convert to pb

		return xsec


	# returns denominator of asymmetry in pb or fb (differential in Q^2, x (or y), \phi, p_T, z, t, \phi_p, \phi_\Delta)
	def denominator(self, kinematics, diff='dx'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta
		phi_Dp, phi_kp = kinematics.phi_Dp, kinematics.phi_kp

		denominator_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2)*y)
		if diff == 'dx':
			denominator_prefactor *= (0.25*pT*y)/(x*z*(1-z))
		elif diff == 'dy':
			denominator_prefactor *= (0.25*pT)/(z*(1-z))
		else:
			raise ValueError('diff should be dx or dy')

		tt_term =  (1 + (1-y)**2) * self.get_coeff('A_TT_unpolar', kinematics)
		tt_term += (1 + (1-y)**2) * (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT_unpolar', kinematics)

		tmt_term = -2*(1-y)* np.cos(2*phi_kp)*self.get_coeff('A_TmT_unpolar', kinematics) 
		tmt_term += -2*(1-y)* (delta/pT)*np.cos(phi_Dp)*np.cos(2*phi_kp)*self.get_coeff('B_TmT_unpolar', kinematics) 
		tmt_term += -2*(1-y)* (delta/pT)*np.sin(phi_Dp)*np.sin(2*phi_kp)*self.get_coeff('C_TmT_unpolar', kinematics)

		ll_term = 4*(1-y)* self.get_coeff('A_LL_unpolar', kinematics) 
		ll_term += 4*(1-y)* (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_LL_unpolar', kinematics)

		xsec = denominator_prefactor*(tt_term + tmt_term + ll_term)
		xsec *= 0.3894*(10**12) # convert to fb 
		# xsec *= 0.3894*(10**9) # convert to pb

		# print(tt_term ,tmt_term, ll_term)
		return xsec



	# returns numerator of asymmetry in pb or fb integrated over azimuthal angles (differential in Q^2, x (or y), p_T, z, t)
	def angle_integrated_numerator(self, kinematics, weight='1', diff='dx'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta

		numerator_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2))
		if diff == 'dx':
			numerator_prefactor *= (0.25*pT*y)/(x*z*(1-z))
		elif diff == 'dy':
			numerator_prefactor *= (0.25*pT)/(z*(1-z))
		else:
			raise ValueError('diff should be dx or dy')

		if weight == '1':
			tt_term = (2-y) * self.get_coeff('A_TT', kinematics)
			lt_term = 0
			numerator_prefactor *= 8*(np.pi**3)

		elif weight == 'cos(phi_Dp)':
			tt_term = (2-y) * (delta/pT) * self.get_coeff('B_TT', kinematics)
			lt_term = 0
			numerator_prefactor *= 4*(np.pi**3)


		elif weight == 'cos(phi_Dp)cos(phi_kp)' or weight == 'cos(phi_kp)cos(phi_Dp)':
			tt_term = 0
			lt_term = np.sqrt(2-2*y) * (delta/pT) * self.get_coeff('B_LT', kinematics) 
			numerator_prefactor *= 2*(np.pi**3)


		elif weight == 'sin(phi_Dp)sin(phi_kp)' or weight == 'sin(phi_kp)sin(phi_Dp)':
			tt_term = 0
			lt_term = np.sqrt(2-2*y) * (delta/pT) * self.get_coeff('C_LT', kinematics)
			numerator_prefactor *= 2*(np.pi**3)

		elif weight == 'cos(phi_kp)':
			tt_term = 0
			lt_term = np.sqrt(2-2*y) * self.get_coeff('A_LT', kinematics) 
			numerator_prefactor *= 4*(np.pi**3)

		else:
			raise ValueError(f'weight {weight} not recognized')

		xsec = numerator_prefactor*(tt_term + lt_term)
		xsec *= 0.3894*(10**12) # convert to fb 
		# xsec *= 0.3894*(10**9) # convert to pb

		return xsec



	# returns denominator of asymmetry in pb or fb integrated over azimuthal angles (differential in Q^2, x (or y), p_T, z, t)
	def angle_integrated_denominator(self, kinematics, diff='dx'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta
		phi_Dp, phi_kp = kinematics.phi_Dp, kinematics.phi_kp

		denominator_prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2)*y)
		if diff == 'dx':
			denominator_prefactor *= (0.25*pT*y)/(x*z*(1-z))
		elif diff == 'dy':
			denominator_prefactor *= (0.25*pT)/(z*(1-z))
		else:
			raise ValueError('diff should be dx or dy')

		tt_term =  (1 + (1-y)**2) * self.get_coeff('A_TT_unpolar', kinematics)
		ll_term = 4*(1-y)* self.get_coeff('A_LL_unpolar', kinematics) 

		denominator_prefactor *= 8*(np.pi**3)

		xsec = denominator_prefactor*(tt_term + ll_term)
		xsec *= 0.3894*(10**12) # convert to fb 
		# xsec *= 0.3894*(10**9) # convert to pb

		# print(tt_term ,tmt_term, ll_term)
		return xsec


	# returns asymmetry integrated over azimuthal angles (differential in Q^2, x (or y), p_T, z, t)
	def dsa(self, kinematics, weight='1'):
		numerator = self.angle_integrated_numerator(kinematics, weight=weight)
		denominator = self.angle_integrated_denominator(kinematics)

		return numerator/denominator



	# returns numerator of asymmetry in pb or fb integrated over phase space except for pT (differential in p_T)
	def integrated_numerator(self, pT, s, weight='1', points=50):

		kinematics = Kinematics(pT = pT, s = s)
	
		y_range = [0.05, 0.95]
		z_range = [0.2, 0.4]
		Q2_range = [16, 100]
		t_range = [0.01, 0.04]

		npoints = points
		y_values = np.linspace(y_range[0], y_range[1], npoints)
		z_values = np.linspace(z_range[0], z_range[1], npoints)

		Q2_max_values = 0.01 * s * y_values
		Q2_grids = [np.linspace(Q2_range[0], Q2_max, npoints) for Q2_max in Q2_max_values]

		dy = (y_values[1] - y_values[0])
		dz = (z_values[1] - z_values[0]) 

		result = 0
		for iy, y in enumerate(y_values):
			kinematics.y = y
			if Q2_grids[iy][-1] < Q2_max_values[iy]: continue

			for Q2 in Q2_grids[iy]:
				dQ2 = (Q2_grids[iy][1] - Q2_grids[iy][0])
				kinematics.Q = np.sqrt(Q2)
				x = Q2/(s*y)
				if x > 0.01: continue
				kinematics.x = x

				for z in z_values:
					if np.sqrt(Q2)*np.sqrt(z*(1-z)) < 2: continue
					kinematics.z = z
					kinematics.delta = 1 	# doing t integral analytically so just set to 1 in expressions
					result += dy * dz * dQ2 * self.angle_integrated_numerator(kinematics, weight=weight, diff='dy')


		if weight in ['cos(phi_Dp)', 'cos(phi_Dp)cos(phi_kp)', 'cos(phi_kp)cos(phi_Dp)', 'sin(phi_Dp)sin(phi_kp)', 'sin(phi_kp)sin(phi_Dp)']:
			t_integral = (2.0/3.0)*(t_range[1]**(1.5) - t_range[0]**(1.5))

		elif weight in ['1', 'cos(phi_kp)']:
			t_integral = t_range[1]-t_range[0]

		# print(result, t_integral)

		return result*t_integral


	# returns denominator of asymmetry in pb or fb integrated over phase space except for pT (differential in p_T)
	def integrated_denominator(self, pT, s, points=50):

		kinematics = Kinematics(pT = pT, s = s)
	
		y_range = [0.05, 0.95]
		z_range = [0.2, 0.4]
		Q2_range = [16, 100]
		t_range = [0.01, 0.04]

		npoints = points
		y_values = np.linspace(y_range[0], y_range[1], npoints)
		z_values = np.linspace(z_range[0], z_range[1], npoints)

		Q2_max_values = 0.01 * s * y_values
		Q2_grids = [np.linspace(Q2_range[0], Q2_max, npoints) for Q2_max in Q2_max_values]

		dy = (y_values[1] - y_values[0])
		dz = (z_values[1] - z_values[0]) 

		result = 0
		for iy, y in enumerate(y_values):
			kinematics.y = y
			if Q2_grids[iy][-1] < Q2_max_values[iy]: continue

			for Q2 in Q2_grids[iy]:
				dQ2 = (Q2_grids[iy][1] - Q2_grids[iy][0])
				kinematics.Q = np.sqrt(Q2)
				x = Q2/(s*y)
				if x > 0.01: continue
				kinematics.x = x

				for z in z_values:
					if np.sqrt(Q2)*np.sqrt(z*(1-z)) < 2: continue
					kinematics.z = z
					kinematics.delta = 1 	# doing t integral analytically so just set to 1 in expressions
					result += dy * dz * dQ2 * self.angle_integrated_denominator(kinematics, diff='dy')

		t_integral = t_range[1]-t_range[0]

		# print(result, t_integral)

		return result*t_integral



	# returns denominator of asymmetry in pb or fb integrated over phase space except for pT (differential in p_T)
	# def integrated_denominator(self, pT, s):

	# 	kinematics = Kinematics(pT = pT, s = s)
	
	# 	y_range = [0.05, 0.95]
	# 	z_range = [0.2, 0.4]
	# 	Q2_range = [16, 100]
	# 	t_range = [0.01, 0.04]

	# 	npoints = 70
	# 	y_values = np.linspace(y_range[0], y_range[1], npoints)
	# 	z_values = np.linspace(z_range[0], z_range[1], npoints)
	# 	Q2_values = np.linspace(Q2_range[0], Q2_range[1], npoints)

	# 	dy = (y_values[1] - y_values[0])
	# 	dz = (z_values[1] - z_values[0]) 
	# 	dQ2 = (Q2_values[1] - Q2_values[0])

	# 	result = 0
	# 	for y in y_values:
	# 		kinematics.y = y
	# 		for Q2 in Q2_values:
	# 			kinematics.Q = np.sqrt(Q2)
	# 			x = Q2/(s*y)
	# 			if x > 0.01: continue
	# 			kinematics.x = x

	# 			for z in z_values:
	# 				if np.sqrt(Q2)*np.sqrt(z*(1-z)) < 2: continue
	# 				kinematics.z = z
	# 				result += dy * dz * dQ2 * self.angle_integrated_denominator(kinematics, diff='dy')

	# 	t_integral = t_range[1]-t_range[0]

	# 	# print(result, t_integral)

	# 	return result*t_integral



	# returns double spin asymmetry integrated over phase space except for pT (differential in p_T)
	def integrated_dsa(self, pT, s, weight='1', points=30):

		numerator = self.integrated_numerator(pT, s, weight=weight, points=points)
		denominator = self.integrated_denominator(pT, s, points=points)

		return numerator/denominator




	def integrated_numerator_approx(self, pT, s, weight='1', points=10):

		kinematics = Kinematics(pT=pT, s=s)

		y_range = [0.05, 0.95]
		z_range = [0.2, 0.4]
		Q2_min_fixed = 16
		t_range = [0.01, 0.04]

		N = points  # number of Gauss points

		# Get Gauss–Legendre points and weights on [-1,1]
		y_nodes, y_weights = leggauss(N)
		z_nodes, z_weights = leggauss(N)
		Q2_nodes, Q2_weights = leggauss(N)

		# Map to [a, b]
		def map_to_interval(nodes, weights, a, b):
			mapped_nodes = 0.5 * (b + a) + 0.5 * (b - a) * nodes
			mapped_weights = 0.5 * (b - a) * weights
			return mapped_nodes, mapped_weights

		# Map y and z to physical ranges
		y_values, y_w = map_to_interval(y_nodes, y_weights, *y_range)
		z_values, z_w = map_to_interval(z_nodes, z_weights, *z_range)

		result = 0.0
		for i, y in enumerate(y_values):
			kinematics.y = y
			Q2_max = 0.01 * s * y
			if Q2_max < Q2_min_fixed:
				continue  # skip if small-x region is closed out

			# Map Q² to [Q2_min_fixed, Q2_max(y)]
			Q2_values, Q2_w = map_to_interval(Q2_nodes, Q2_weights, Q2_min_fixed, Q2_max)

			for j, Q2 in enumerate(Q2_values):
				kinematics.Q = np.sqrt(Q2)
				x = Q2 / (s * y)
				kinematics.x = x

				for k, z in enumerate(z_values):
					if np.sqrt(Q2) * np.sqrt(z * (1 - z)) < 2:
						continue  # skip low-mass region
					kinematics.z = z
					kinematics.delta = 1  # doing t integral analytically

					weight_factor = y_w[i] * Q2_w[j] * z_w[k]
					result += weight_factor * self.angle_integrated_numerator(
						kinematics, weight=weight, diff='dy'
					)

		# Analytical t-integral factor
		if weight in ['cos(phi_Dp)', 'cos(phi_Dp)cos(phi_kp)', 'cos(phi_kp)cos(phi_Dp)',
					  'sin(phi_Dp)sin(phi_kp)', 'sin(phi_kp)sin(phi_Dp)']:
			t_integral = (2.0 / 3.0) * (t_range[1] ** 1.5 - t_range[0] ** 1.5)
		elif weight in ['1', 'cos(phi_kp)']:
			t_integral = t_range[1] - t_range[0]

		return result * t_integral




	def integrated_denominator_approx(self, pT, s, points=10):

		kinematics = Kinematics(pT=pT, s=s)

		y_range = [0.05, 0.95]
		z_range = [0.2, 0.4]
		Q2_min_fixed = 16
		t_range = [0.01, 0.04]

		N = points  # number of Gauss points

		# Get Gauss–Legendre points and weights on [-1,1]
		y_nodes, y_weights = leggauss(N)
		z_nodes, z_weights = leggauss(N)
		Q2_nodes, Q2_weights = leggauss(N)

		# Map to [a, b]
		def map_to_interval(nodes, weights, a, b):
			mapped_nodes = 0.5 * (b + a) + 0.5 * (b - a) * nodes
			mapped_weights = 0.5 * (b - a) * weights
			return mapped_nodes, mapped_weights

		# Map y and z to physical ranges
		y_values, y_w = map_to_interval(y_nodes, y_weights, *y_range)
		z_values, z_w = map_to_interval(z_nodes, z_weights, *z_range)

		result = 0.0
		for i, y in enumerate(y_values):
			kinematics.y = y
			Q2_max = 0.01 * s * y
			if Q2_max < Q2_min_fixed:
				continue  # skip if small-x region is closed out

			# Map Q2 to [Q2_min_fixed, Q2_max(y)]
			Q2_values, Q2_w = map_to_interval(Q2_nodes, Q2_weights, Q2_min_fixed, Q2_max)

			for j, Q2 in enumerate(Q2_values):
				kinematics.Q = np.sqrt(Q2)
				x = Q2 / (s * y)
				kinematics.x = x

				for k, z in enumerate(z_values):
					if np.sqrt(Q2) * np.sqrt(z * (1 - z)) < 2:
						continue  # skip low-mass region
					kinematics.z = z
					kinematics.delta = 1  # doing t integral analytically

					weight_factor = y_w[i] * Q2_w[j] * z_w[k]
					result += weight_factor * self.angle_integrated_denominator(kinematics, diff='dy')

		# Analytical t-integral factor
		t_integral = t_range[1] - t_range[0]

		return result * t_integral



	# returns double spin asymmetry integrated over phase space except for pT (differential in p_T)
	def integrated_dsa_approx(self, pT, s, weight='1', points=10):

		numerator = self.integrated_numerator_approx(pT, s, weight=weight, points=points)
		denominator = self.integrated_denominator_approx(pT, s, points=points)

		return numerator/denominator



	# # returns dxsec/dQ^2 (integrates over x, z, pT, and t)
	# def dxsec_dQ2(self, Q, s, x_range, z_range, pT_range, t_range):

	# 	npoints = 14

	# 	x_range = [(Q**2)/(s*0.7), min((Q**2)/(s*0.03), 0.01)]

	# 	dY = (np.log(1/x_range[0])-np.log(1/x_range[1]))/npoints
	# 	dz = (z_range[1]-z_range[0])/npoints
	# 	dpT = (pT_range[1]-pT_range[0])/npoints

	# 	# integral is not t-dependent --> just multiply by interval
	# 	dt = (t_range[1]-t_range[0])

	# 	Y_values = np.linspace(np.log(1/x_range[1]), np.log(1/x_range[0]), npoints)
	# 	z_values = np.linspace(z_range[0], z_range[1], npoints)
	# 	pT_values = np.linspace(pT_range[0], pT_range[1], npoints)

	# 	measure = dY*dz*dpT*dt

	# 	xsec = 0
	# 	for Y in Y_values:
	# 		weight = measure*np.exp(-Y) # from jacobian x->Y

	# 		y = (Q**2)/(s*np.exp(-Y))
	# 		if y > 1: raise ValueError(f'y>1: y={y}')

	# 		for z in z_values:
	# 			for pT in pT_values:
	# 				kins = Kinematics(Q=Q, z=z, x=np.exp(-Y), s=s, y=y, pT=pT)
	# 				xsec += weight*self.get_xsec(kins, 'unpolarized_integrated', 'dx')
	# 	return xsec



	# # returns dxsec/dpT (integrates over x, z, Q, and t)
	# def dxsec_dpT(self, pT, s, x_range, z_range, Q_range, t_range):

	# 	npoints = 14
	# 	dz = (z_range[1]-z_range[0])/npoints
	# 	dQ = (Q_range[1]-Q_range[0])/npoints

	# 	# integral is not t-dependent --> just multiply by interval
	# 	dt = (t_range[1]-t_range[0])

	# 	z_values = np.linspace(z_range[0], z_range[1], npoints)
	# 	Q_values = np.linspace(Q_range[0], Q_range[1], npoints)

	# 	measure = dz*dQ*dt

	# 	xsec = 0

	# 	for Q in Q_values:
	# 		x_range = [(Q**2)/(s*0.7), min((Q**2)/(s*0.03), 0.01)]
	# 		dY = (np.log(1/x_range[0])-np.log(1/x_range[1]))/npoints
	# 		Y_values = np.linspace(np.log(1/x_range[1]), np.log(1/x_range[0]), npoints)


	# 		for Y in Y_values:
	# 			weight = measure*dY*np.exp(-Y) # from jacobian x->Y

	# 			y = (Q**2)/(s*np.exp(-Y))
	# 			if y > 1: raise ValueError(f'y>1: y={y}')

	# 			for z in z_values:
	# 				kins = Kinematics(Q=Q, z=z, x=np.exp(-Y), s=s, y=y, pT=pT)
	# 				xsec += weight*self.get_xsec(kins, 'unpolarized_integrated', 'dx')
	# 	return xsec


	# # returns dxsec/dt (integrates over x, z, Q, and pT)
	# def dxsec_dt(self, t, s, x_range, z_range, Q_range, pT_range):

	# 	npoints = 14

	# 	dz = (z_range[1]-z_range[0])/npoints
	# 	dQ = (Q_range[1]-Q_range[0])/npoints
	# 	dpT = (pT_range[1]-pT_range[0])/npoints

	# 	z_values = np.linspace(z_range[0], z_range[1], npoints)
	# 	pT_values = np.linspace(pT_range[0], pT_range[1], npoints)
	# 	Q_values = np.linspace(Q_range[0], Q_range[1], npoints)

	# 	measure = dz*dQ*dpT

	# 	xsec = 0
	# 	for Q in Q_values:
	# 		x_range = [(Q**2)/(s*0.7), min((Q**2)/(s*0.03), 0.01)]
	# 		dY = (np.log(1/x_range[0])-np.log(1/x_range[1]))/npoints
	# 		Y_values = np.linspace(np.log(1/x_range[1]), np.log(1/x_range[0]), npoints)


	# 		for Y in Y_values:
	# 			weight = measure*dY*np.exp(-Y) # from jacobian x->Y

	# 			y = (Q**2)/(s*np.exp(-Y))
	# 			if y > 1: raise ValueError(f'y>1: y={y}')

	# 			for z in z_values:
	# 				for pT in pT_values:

	# 					kins = Kinematics(Q=Q, z=z, x=np.exp(-Y), s=s, y=y, pT=pT)
	# 					xsec += weight*self.get_xsec(kins, 'unpolarized_integrated', 'dx')
	# 	return xsec




	# # returns dxsec/dy (integrates over t, z, Q, and pT)
	# def dxsec_dy(self, y, s, t_range, z_range, Q_range, pT_range):

	# 	npoints = 14

	# 	dz = (z_range[1]-z_range[0])/npoints
	# 	dQ = (Q_range[1]-Q_range[0])/npoints
	# 	dpT = (pT_range[1]-pT_range[0])/npoints

	# 	# integral is not t-dependent --> just multiply by interval
	# 	dt = (t_range[1]-t_range[0])

	# 	z_values = np.linspace(z_range[0], z_range[1], npoints)
	# 	pT_values = np.linspace(pT_range[0], pT_range[1], npoints)
	# 	Q_values = np.linspace(Q_range[0], Q_range[1], npoints)

	# 	measure = dt*dz*dQ*dpT

	# 	xsec = 0
			
	# 	for Q in Q_values:
	# 		x = (Q**2)/(s*y)
	# 		if x > 0.01: 
	# 			print(f'skipping x={x}')
	# 			continue

	# 		for z in z_values:
	# 			for pT in pT_values:

	# 				kins = Kinematics(Q=Q, z=z, x=x, s=s, y=y, pT=pT)
	# 				xsec += measure*self.get_xsec(kins, 'unpolarized_integrated', 'dy')
	# 	return xsec



	# returns correlations of asymmetry
	# def get_correlation_numerator(self, kvar, kind):
	# 	# self.filter_dipole(kvar)

	# 	y, delta, pT = kvar.y, kvar.delta, kvar.pT

	# 	if kind == '<1>': 
	# 		num = y*(2-y)*self.get_coeff('A_TT', kvar)

	# 	elif kind == '<cos(phi_Dp)>': 
	# 		num = 0.5*y*(2-y)*(delta/pT)*self.get_coeff('B_TT', kvar)

	# 	elif kind == '<cos(phi_kp)>': 
	# 		num = 0.5*y*np.sqrt(2-2*y)*self.get_coeff('A_LT', kvar)

	# 	elif kind == '<cos(phi_Dp)cos(phi_kp)>':
	# 		num = 0.25*y*np.sqrt(2-2*y)*(delta/pT)*self.get_coeff('B_LT', kvar)

	# 	elif kind == '<sin(phi_Dp)sin(phi_kp)>': 
	# 		num = 0.25*y*np.sqrt(2-2*y)*(delta/pT)*self.get_coeff('C_LT', kvar)

	# 	else: 
	# 		raise ValueError(f'Error: Correlation {kind} not recognized')

	# 	den = (1 + (1-y)**2)*self.get_coeff('A_TT_unpolar', kvar) + 4*(1-y)*self.get_coeff('A_LL_unpolar', kvar)
	# 	return num/den


	# returns correlation/dpT 
	# def get_integrated_correlation(self, kind, pT, s, t_range, x_range, z_range, Q_range):

	# 	def corr(x, Q, z, delta):
	# 		y = (Q**2)/(s*x)
	# 		ivar = Kinematics(x=x, Q=Q, z=z, delta=delta, pT=pT, s=s, y=y)
	# 		return self.get_correlation(ivar, kind)

	# 	def x_limits(Q, z, delta):
	# 		return [max(x_range[0], (Q**2)/(ymax*s)), min(x_range[1], (Q**2)/(ymin*s))]

	# 	ymin = 0.05
	# 	ymax = 0.9

	# 	limits = [
	# 		x_limits,
	# 		[Q_range[0], Q_range[1]], 
	# 		[z_range[0], z_range[1]],	
	# 		[np.sqrt(t_range)[0], np.sqrt(t_range)[1]]	# \Delta range
	# 	]

	# 	result, error = nquad(corr, limits)
	# 	return result



	# def get_correlation_integrand(self, kvar, kind):
	# 	# self.filter_dipole(kvar)

	# 	y, pT = kvar.y, kvar.pT

	# 	if kind == '<1>': 
	# 		num = y*(2-y)*self.get_coeff('A_TT', kvar)

	# 	elif kind == '<cos(phi_Dp)>': 
	# 		num = 0.5*y*(2-y)*(1/pT)*self.get_coeff('B_TT', kvar)

	# 	elif kind == '<cos(phi_kp)>': 
	# 		num = 0.5*y*np.sqrt(2-2*y)*self.get_coeff('A_LT', kvar)

	# 	elif kind == '<cos(phi_Dp)cos(phi_kp)>':
	# 		num = 0.25*y*np.sqrt(2-2*y)*(1/pT)*self.get_coeff('B_LT', kvar)

	# 	elif kind == '<sin(phi_Dp)sin(phi_kp)>': 
	# 		num = 0.25*y*np.sqrt(2-2*y)*(1/pT)*self.get_coeff('C_LT', kvar)

	# 	else: 
	# 		raise ValueError(f'Error: Correlation {kind} not recognized')

	# 	den = (1 + (1-y)**2)*self.get_coeff('A_TT_unpolar', kvar) + 4*(1-y)*self.get_coeff('A_LL_unpolar', kvar)
	# 	return num/den



	# def get_integrated_correlation(self, kind, pT, s, t_range, y_range, z_range, Q2_range):

	# 	ivars = Kinematics(pT = pT, s = s)
	
	# 	t_integral = (2.0/3.0)*(t_range[1]**(1.5) - t_range[0]**(1.5))

	# 	y_values = np.linspace(y_range[0], y_range[1], 40)
	# 	z_values = np.linspace(z_range[0], z_range[1], 40)
	# 	Q2_values = np.linspace(Q2_range[0], Q2_range[1], 40)

	# 	dy = (y_values[1] - y_values[0])
	# 	dz = (z_values[1] - z_values[0]) 
	# 	dQ2 = (Q2_values[1] - Q2_values[0])

	# 	result = 0
	# 	for y in y_values:
	# 		ivars.y = y
	# 		for Q2 in Q2_values:
	# 			ivars.Q = np.sqrt(Q2)
	# 			x = Q2/(s*y)
	# 			if x > 0.01: continue
	# 			ivars.x = x
	# 			# print(x)

	# 			for z in z_values:
	# 				if np.sqrt(Q2)*np.sqrt(z*(1-z)) < 2: continue
	# 				ivars.z = z
	# 				result += dy * dz * dQ2 * self.get_correlation_integrand(ivars, kind)

	# 	print(result, t_integral)

	# 	return result*t_integral





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


	# functions to calculate helicity objects below
	def get_g1(self, x, Q2):

		g1 = 0
		for flav in ['u', 'd', 's']:

			G = self.pdipoles[f'Q{flav}'] + 2*self.pdipoles['G2']

			x0 = 1.0
			eta0 = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x0)
			eta0index = int(np.ceil(eta0/self.deta))+1
			eta = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x)
			jetai = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(1./x)/self.deta))
			jetaf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))

			g1plus=0
			for j in range(eta0index,jetai):
				si = 0
				sf = int(np.ceil(round((self.deta*(j)-eta0)/self.deta,6))) #1 + np.int(np.ceil((deta*(j-1)-eta0)/deta))
				
				for i in range(si,sf):
					g1plus += self.deta*self.deta * G[i,j]

			for j in range(jetai, jetaf):
				si = int(np.ceil((self.deta*(j)-np.sqrt(self.Nc/(2*np.pi))*np.log(1./x))/self.deta))
				sf = int(np.ceil(round((self.deta*(j)-eta0)/self.deta,6))) #1 + np.int(np.ceil((deta*(j-1)-eta0)/deta))
				for i in range(si,sf): 
					g1plus += self.deta*self.deta * G[i,j]

			g1plus *= (-1./(np.pi**2))

			if flav == 'u': g1 += self.Zusq*g1plus
			if flav == 'd': g1 += self.Zdsq*g1plus
			if flav == 's': g1 += self.Zssq*g1plus
		
		return g1


	# # daniel's version 
	def get_ppdfplus(self, x, Q2, flavor):
		G = self.pdipoles[f'Q{flavor}'] + 2*self.pdipoles['G2']

		x0 = 1.0
		eta0 = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x0)
		eta0index = int(np.ceil(eta0/self.deta))+1
		eta = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x)
		jetai = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(1./x)/self.deta))
		jetaf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))

		qplus=0
		si = 0
		sf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(self.lambdaIR**2))/self.deta))
		for i in range(si, sf):
			jetai = i + eta0index
			jetaf = i + int(np.ceil(eta/self.deta))

			for j in range(jetai, jetaf):
				qplus += self.deta*self.deta * G[i,j]

		qplus *= (-1./(np.pi**2))
		
		return qplus



	def get_DeltaSigma(self, x, Q2): # returns \sum_f(delta q_f + delta qb_f)(x)

		if self.corrected_evo:
			s10 = np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(self.lambdaIR**2))
			s10_index = int(np.ceil(s10/self.deta))
			eta_index = int(np.ceil(np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))
			Del_Sigma = (1/self.alpha_s(True, 0, s10))*((self.Nc)/(np.pi**2))*(self.pdipoles['QTu'][s10_index, eta_index] + self.pdipoles['QTd'][s10_index, eta_index] + self.pdipoles['QTs'][s10_index, eta_index])
			return Del_Sigma

		else:
			Del_Sigma = self.get_ppdfplus(x, Q2, 'u') + self.get_ppdfplus(x, Q2, 'd') + self.get_ppdfplus(x, Q2, 's')
			return Del_Sigma



	def get_DeltaG(self, x, Q2):
		s10 = np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(self.lambdaIR**2))
		s10_index = int(np.ceil(s10/self.deta))
		eta_index = int(np.ceil(np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))
		Del_G = (1/self.alpha_s(True, 0, s10))*((2*self.Nc)/(np.pi**2))*self.pdipoles['G2'][s10_index, eta_index]
		return Del_G


	# returns Lq
	def get_Lq(self, x, Q2, flavor):
		G = self.pdipoles[f'Q{flavor}'] + 3*self.pdipoles['G2'] - self.pdipoles[f'I3{flavor}'] + 2*self.pdipoles['I4'] - self.pdipoles['I5']

		x0 = 1.0
		eta0 = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x0)
		eta0index = int(np.ceil(eta0/self.deta))+1
		eta = np.sqrt(self.Nc/(2*np.pi))*np.log(1/x)
		jetai = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(1./x)/self.deta))
		jetaf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))

		Lq=0
		si = 0
		sf = int(np.ceil(np.sqrt(self.Nc/(2.*np.pi))*np.log(Q2/(self.lambdaIR**2))/self.deta))
		for i in range(si, sf):
			jetai = i + eta0index
			jetaf = i + int(np.ceil(eta/self.deta))

			for j in range(jetai, jetaf):
				Lq += self.deta*self.deta * G[i,j]

		Lq *= (-1./(np.pi**2))
		
		return Lq


	def get_Lsinglet(self, x, Q2): # returns \sum_f(L_)(x)
		Lsinglet = self.get_Lq(x, Q2, 'u') + self.get_Lq(x, Q2, 'd') + self.get_Lq(x, Q2, 's')
		return Lsinglet


	def get_LG(self, x, Q2):
		s10 = np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(self.lambdaIR**2))
		s10_index = int(np.ceil(s10/self.deta))
		eta_index = int(np.ceil(np.sqrt(self.Nc/(2*np.pi))*np.log(Q2/(x*(self.lambdaIR**2)))/self.deta))
		L_G = -(1/self.alpha_s(True, 0, s10))*((2*self.Nc)/(np.pi**2))*(2*self.pdipoles['I4'][s10_index, eta_index] + 3*self.pdipoles['I5'][s10_index, eta_index])
		return L_G


	def get_IntegratedPDF(self, kind, Q2, xmin=10**(-5), xmax=0.1):
		npoints = 10

		if kind == 'Lq':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: self.get_Lsinglet(x, Q2)), xmin, xm, n=npoints)[0]
		elif kind == 'LG':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: self.get_LG(x, Q2)), xmin, xm, n=npoints)[0]
		elif kind == 'DeltaSigma':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: self.get_DeltaSigma(x, Q2)), xmin, xm, n=npoints)[0]
		elif kind == 'DeltaG':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: self.get_DeltaG(x, Q2)), xmin, xm, n=npoints)[0]
		elif kind == 'helicity':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: 0.5*self.get_DeltaSigma(x, Q2) + self.get_DeltaG(x, Q2)), xmin, xm, n=npoints)[0]
		elif kind == 'oam':
			ifunc = lambda xm: fixed_quad(np.vectorize(lambda x: self.get_Lsinglet(x, Q2) + self.get_LG(x, Q2)), xmin, xm, n=npoints)[0]
		else:
			raise ValueError(f'dont know {kind}')

		return ifunc(xmax)




if __name__ == '__main__':

	test_kins = Kinematics()
	test_kins.x = 0.01
	test_kins.Q = 5
	test_kins.z = 0.4
	test_kins.s = 55**2
	test_kins.delta = 0.2
	test_kins.phi_Dp = 0
	test_kins.phi_kp = 0
	test_kins.pT = 2.0
	test_kins.y = (test_kins.Q**2)/(test_kins.s*test_kins.x)

	print(test_kins.y)

	dj = DIJET(1)


	# testing dsa functions
	# print('numerator', dj.numerator(test_kins))
	# print('denominator', dj.denominator(test_kins))
	# print('angle integrated numerator', dj.angle_integrated_numerator(test_kins))
	# print('angle integrated denominator',  dj.angle_integrated_denominator(test_kins))
	# print('dsa', dj.dsa(test_kins))

	import time

	# start = time.time()
	# for pT in range(1, 15):
	# 	print(pT)
	# 	num = dj.integrated_numerator(pT, test_kins.s, weight='1')
	# 	den = dj.integrated_denominator(pT, test_kins.s)
	# 	dsa = dj.integrated_dsa(pT, test_kins.s, weight='1')

	# 	error = np.sqrt(den/10)

	# 	print('num', num)
	# 	print('den', den)
	# 	print('dsa', dsa)
	# 	print('error', error)
	# 	print('dsa error', np.sqrt((1+dsa)/(den*10)))
		# print('num', num)

	# start = time.time()
	# print('integrated num (10 points)', dj.integrated_dsa(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=10))
	# print('elasp. time: ', time.time() - start)


	# start = time.time()
	# print('integrated num (20 points)', dj.integrated_dsa(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=20))
	# print('elasp. time: ', time.time() - start)

	# start = time.time()
	# print('integrated num (40 points)', dj.integrated_dsa(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=40))
	# print('elasp. time: ', time.time() - start)


	# start = time.time()
	# print('integrated num (60 points)', dj.integrated_dsa(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=60))
	# print('elasp. time: ', time.time() - start)


	start = time.time()
	print('integrated num (approx)', dj.integrated_dsa_approx(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=4))
	print('elasp. time: ', time.time() - start)

	start = time.time()
	print('integrated num (approx)', dj.integrated_dsa_approx(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=5))
	print('elasp. time: ', time.time() - start)

	start = time.time()
	print('integrated num (approx)', dj.integrated_dsa_approx(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=6))
	print('elasp. time: ', time.time() - start)

	start = time.time()
	print('integrated num (approx)', dj.integrated_dsa_approx(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=8))
	print('elasp. time: ', time.time() - start)


	start = time.time()
	print('integrated num (approx)', dj.integrated_dsa_approx(test_kins.pT, test_kins.s, weight='cos(phi_Dp)', points=20))
	print('elasp. time: ', time.time() - start)



	# testing helicity functions
	# print(dj.get_IntegratedPDF('DeltaSigma', 10))


	# t_range = [0.01, 0.04]
	# y_range = [0.05, 0.95]
	# pT_range = [5, 15] 
	# z_range = [0.2, 0.4]
	# Q2_range = [16, 100]

	# print(dj.get_integrated_correlation('<cos(phi_Dp)>', 1, 95**2, t_range, y_range, z_range, Q2_range))

	# print(dj.dxsec_dQ2(8, 320**2, x_range, z_range, pT_range, t_range))
	# print(dj.dxsec_dpT(5, 320**2, x_range, z_range, Q_range, t_range))
	# print(dj.dxsec_dt(0.2, 320**2, x_range, z_range, Q_range, pT_range))


		
