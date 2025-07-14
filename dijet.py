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
from dataclasses import dataclass, replace
from scipy.integrate import quad,fixed_quad, nquad, simps

from numpy.polynomial.legendre import leggauss


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
	elif params[0] == 'exp':
		values *= np.exp(-r*params[1])
	return values


def map_to_interval(ns, ws, a, b):
	mapped_nodes = 0.5 * (b + a) + 0.5 * (b - a) * ns
	mapped_weights = 0.5 * (b - a) * ws
	return mapped_nodes, mapped_weights


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
	def __init__(self, **options):

		# define optional parameters
		replica = options.get('replica', 1)
		self.lambdaIR = options.get('lambdaIR', 0.3)
		self.deta = options.get('deta', 0.05)
		self.IR_params = options.get('IR_reg', [None, 0.0])
		fit_type = options.get('fit_type', 'pp')

		# testing conditionals
		self.mv_only = options.get('mv_only', False)
		self.fix_moments = options.get('fix_moments', False)
		self.old_rap = options.get('old_rap', False)
		self.corrected_evo = options.get('corrected_evo', False)
		self.constrained_moments = options.get('constrained_moments', False)

		# warn about testing conditionals
		if self.mv_only: print('--> !!! Using only initial conditions for N!')
		if self.fix_moments: print('--> !!! Moment amplitude parameters are fixed!')
		if self.old_rap: print('--> !!! Using ln(1/x) in argument of N!')
		if self.corrected_evo: print('--> !!! Using corrected evolution!')
		if self.constrained_moments: print('--> !!! Using constrained moment parameters!')

		# define physical constants
		self.alpha_em = 1/137.0
		self.Nc = 3.0
		self.Nf = 3.0
		self.Zusq = 4.0/9.0
		self.Zdsq = 1.0/9.0
		self.Zssq = 1.0/9.0
		self.Zfsq = self.Zusq + self.Zdsq + self.Zssq

		self.current_dir = os.path.dirname(os.path.abspath(__file__))

		self.load_dipoles()

		if self.constrained_moments: moments = ''
		else: moments = 'random'

		self.load_params(fit_type, moments=moments)
		self.set_params(replica)


	def load_dipoles(self):

		#-- load polarized dipole amplitudes
		deta_str = 'd'+str(self.deta)[2:]
		polar_indir = f'/dipoles/{deta_str}-rc/'
		amps = ['Qu', 'Qd', 'Qs', 'GT', 'G2', 'I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5']

		if self.corrected_evo:
			self.deta = 0.08 # only one supported right now
			polar_indir = f'/dipoles/d08-rc-corrected/'
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

					self.basis_dipoles[iamp][jamp][jbasis] = load(f'{self.current_dir}{polar_indir}pre-cook-{jamp}-{jbasis}-{iamp}.dat')


		#--load unpolarized dipole amplitude
		# mv parameters from 0902.1112
		unpolar_file = '/dipoles/narr_ymin4.61_ymax14.91_AAMS09.dat'
		unpolar_input_file = self.current_dir + unpolar_file
		self.normN = 0.5*32.77*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		# mv parameters from 2311.10491
		# unpolar_file = 'narr_ymin4.61_ymax14.91_CKM24.dat'
		# unpolar_input_file = self.current_dir + '/dipoles/' + unpolar_file
		# self.normN = 13.9*(1/0.3894)  # value of b integral in mb (converted to GeV^-2)

		self.ndipole = np.loadtxt(unpolar_input_file)

		print('--> loaded unpol. amp. data from', unpolar_file)
		print('--> loaded pol. amp. data from', polar_indir)

		# grid values for dipoles
		n_psteps = len(self.basis_dipoles['Qu']['Qu']['eta'])
		self.s10_values = np.arange(0.0, self.deta*(n_psteps), self.deta)

		n_usteps = len(self.ndipole[0])
		self.dlogr = 0.01 # from evolution code
		self.logr_values = np.arange(-13.8, round(np.log(1/self.lambdaIR), 2), self.dlogr)


	def load_params(self, fit_type='dis', **options):

		if fit_type == 'dis':
			params_file = '/dipoles/replica_params_dis.csv'
			mom_params_file = '/dipoles/moment_params_dis_oam3_range10.csv'

		elif fit_type == 'pp':
			params_file = '/dipoles/replica_params_pp.csv'
			mom_params_file = '/dipoles/moment_params_pp_oam3_range10.csv'

		elif fit_type == 'ones':
			params_file = '/dipoles/replica_params_ones.csv'
			mom_params_file = '/dipoles/moment_params_ones.csv'

		elif fit_type == 'one_basis':
			params_file = '/dipoles/replica_params_one_basis.csv'
			mom_params_file = '/dipoles/moment_params_zeros.csv'

		if options.get('moments', '') == 'random':
			mom_params_file = '/dipoles/random_moment_params.csv'

		fdf = pd.read_csv(self.current_dir + params_file)
		header = ['nrep'] + [f'{ia}{ib}' for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2'] for ib in ['eta', 's10', '1']]
		fdf = fdf.dropna(axis=1, how='all')
		fdf.columns = header
		self.params = fdf
		self.mom_params = pd.read_csv(self.current_dir + mom_params_file)

		print('--> loaded params from', params_file)
		print('--> loaded random moment params from', mom_params_file)


		# if self.corrected_evo:
		# 	params_file = 'replica_params_dis_corrected.csv'
		# 	fdf = pd.read_csv(self.current_dir + f'/dipoles/{params_file}')

		# 	header = ['nrep']
		# 	for ia in ['Qu', 'Qd', 'Qs', 'um1', 'dm1', 'sm1', 'GT', 'G2', 'QTu', 'QTd', 'QTs']:
		# 		for ib in ['eta', 's10', '1']:
		# 			if ia in ['GT', 'QTu', 'QTd', 'QTs'] and ib in ['eta', '1']: continue
		# 			header.append(f'{ia}{ib}')



	def set_params(self, nreplica=1):


		sdf = self.params[self.params['nrep'] == nreplica]
		assert len(sdf) == 1, 'Error: more than 1 replica selected...'
		print('--> loaded replica', nreplica)

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
			if iamp in ['GT', 'I3T']: continue

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
		# if self.IR_params[0] != None:
			# amp_values = regulate_IR(amp_values, r, self.IR_params)

		ir_term = np.ones_like(r)
		if self.IR_params[0] == 'gauss':
			ir_term = np.exp(-(r**2)*self.IR_params[1])
		elif self.IR_params[0] == 'skin':
			ir_term = 1.0/(1 + np.exp(self.IR_params[1]*((r/self.IR_params[2]) - 1)))
		elif self.IR_params[0] == 'exp':
			ir_term = np.exp(-r*self.IR_params[1])


		# Compute the Riemann sum for each set of indices
		for i_a, i_b, i_c, i_d in indices_array:
			c_term = (pT*r)**i_c
			d_term = (Qeff*r)**i_d
			jv_term = jv(i_a, pT*r)
			kv_term = kv(i_b, Qeff*r)

			# Perform the sum
			integrand = measure*c_term*d_term*jv_term*kv_term*amp_values*ir_term

			# print(integrand)
			total_sum = np.sum(integrand)
			results.append(total_sum)

		if len(indices_array) > 1: return results
		else: return results[0]




	def get_coeff(self, flavor, kvar):

		Q, x, z = kvar.Q, kvar.x, kvar.z
		prefactor = self.alpha_em*(self.Nc**2)*(Q**2)
		w2 = (Q**2)*((1/x) - 1)

		if 'TT_unpolar' in flavor: prefactor *= self.Zfsq*(4*(z**2)*((1-z)**2)*((z)**2 + (1-z)**2))/((2*np.pi)**4)
		elif 'LL_unpolar' in flavor: prefactor *= self.Zfsq*(8*(z**3)*((1-z)**3))/((2*np.pi)**4)
		elif 'TmT_unpolar' in flavor: prefactor *= self.Zfsq*(-8*(z**3)*((1-z)**3))/((2*np.pi)**4)
		elif 'TT' in flavor: prefactor *= -(z*(1-z))/(2*(np.pi**4)*w2)
		elif 'LT' in flavor: prefactor *= -((z*(1-z))**1.5)/(np.sqrt(2)*(np.pi**4)*w2)

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
			b_TT -= (z**2 + (1-z)**2)*N_11*(self.Zusq*Qu_21_10 + self.Zdsq*Qd_21_10 + self.Zssq*Qs_21_10)
			b_TT -= (z**2 + (1-z)**2)*N_11*self.Zfsq*(2*G2_21_10 + I4_21_10 + I4_10_01 - I5_11 + I5_10_01 + I5_01_10)
			return -1*prefactor*(1-(2*z))*b_TT

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

			# c_LT = (z**2 + (1-z)**2)*N_11*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			c_LT = (z**2 + (1-z)**2)*N_00*(self.Zusq*Qu_11 + self.Zdsq*Qd_11 + self.Zssq*Qs_11)
			c_LT -= N_00*(self.Zusq*I3u_11 + self.Zdsq*I3d_11 + self.Zssq*I3s_11)
			c_LT += 0.5*((1- 2*z)**2)*N_11*(self.Zusq*Qu_00 + self.Zdsq*Qd_00 + self.Zssq*Qs_00)
			c_LT += ((1- 2*z)**2)*N_00*self.Zfsq*(G2_01_10 - 3*G2_11 - I4_10_10 - 2*I4_11 + I4_01_10 - I4_11_20 - I4_10_01 + I4_00_11 + I5_11 - I5_11_20 - I5_10_01 + I5_00_11)
			c_LT += ((1- 2*z)**2)*N_11*self.Zfsq*(G2_10_10 + I4_11_11 + I4_00_20 + I5_11_11 - I5_10_10 + I5_00_20)

			return prefactor*c_LT

		elif flavor == 'A_TT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return prefactor*(N_11**2)

		elif flavor == 'B_TT_unpolar':
			N_11, N_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'N')
			# return prefactor*2*(z-0.5)*(2*(N_11**2) - N_11*N_21_10)
			return prefactor*2*(z-0.5)*N_11*(N_11 - N_21_10)

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
			N_11, N_21_10 = self.fourier_bessel(kvar, [[1,1,0,0], [2,1,1,0]], 'N')
			# return prefactor*(z-0.5)*(-2*(N_11**2) + N_11*N_01_10)
			return prefactor*2*(z-0.5)*N_11*(N_11 - N_21_10)

		elif flavor == 'C_TmT_unpolar':
			N_11 = self.fourier_bessel(kvar, [[1,1,0,0]], 'N')
			return -2*prefactor*(z-0.5)*(N_11**2)

		else:
			print('requested coefficient', flavor, 'does not exist')



	# returns numerator or denominator of asymmetry in pb or fb (differential in Q^2, y (or x), \phi, p_T, z, t, \phi_p, \phi_\Delta)
	def get_xsec(self, kinematics, diff='dy', kind='num', weight='1'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta
		phi_Dp, phi_kp = kinematics.phi_Dp, kinematics.phi_kp

		if kind == 'num':

			prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2))
			if diff == 'dx': prefactor *= (0.25*pT*y)/(x*z*(1-z))
			elif diff == 'dy': prefactor *= (0.25*pT)/(z*(1-z))
			else: raise ValueError('diff should be dx or dy')

			tt_term = (2-y) * self.get_coeff('A_TT', kinematics)
			tt_term += (2-y) * (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT', kinematics)
			lt_term = np.sqrt(2-2*y) * np.cos(phi_kp)*self.get_coeff('A_LT', kinematics)
			lt_term += np.sqrt(2-2*y) * (delta/pT)*np.cos(phi_Dp)*np.cos(phi_kp)*self.get_coeff('B_LT', kinematics)
			lt_term += np.sqrt(2-2*y) * (delta/pT)*np.sin(phi_Dp)*np.sin(phi_kp)*self.get_coeff('C_LT', kinematics)
			xsec = prefactor*(tt_term + lt_term)
			xsec *= 0.3894*(10**12) # convert to fb
			# xsec *= 0.3894*(10**9) # convert to pb
			return xsec

		elif kind == 'den':

			prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2)*y)
			if diff == 'dx': prefactor *= (0.25*pT*y)/(x*z*(1-z))
			elif diff == 'dy': prefactor *= (0.25*pT)/(z*(1-z))
			else: raise ValueError('diff should be dx or dy')

			tt_term =  (1 + (1-y)**2) * self.get_coeff('A_TT_unpolar', kinematics)
			tt_term += (1 + (1-y)**2) * (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_TT_unpolar', kinematics)
			tmt_term = -2*(1-y)* np.cos(2*phi_kp)*self.get_coeff('A_TmT_unpolar', kinematics)
			tmt_term += -2*(1-y)* (delta/pT)*np.cos(phi_Dp)*np.cos(2*phi_kp)*self.get_coeff('B_TmT_unpolar', kinematics)
			tmt_term += -2*(1-y)* (delta/pT)*np.sin(phi_Dp)*np.sin(2*phi_kp)*self.get_coeff('C_TmT_unpolar', kinematics)
			ll_term = 4*(1-y)* self.get_coeff('A_LL_unpolar', kinematics)
			ll_term += 4*(1-y)* (delta/pT)*np.cos(phi_Dp)*self.get_coeff('B_LL_unpolar', kinematics)
			xsec = prefactor*(tt_term + tmt_term + ll_term)
			xsec *= 0.3894*(10**12) # convert to fb
			# xsec *= 0.3894*(10**9) # convert to pb
			return xsec


	# returns numerator or denominator of asymmetry in pb or fb integrated over azimuthal angles (differential in Q^2, y (or x), p_T, z, t)
	def get_angle_integrated_xsec(self, kinematics, weight='1', diff='dy', kind='num'):

		Q, x, y, z, pT, delta = kinematics.Q, kinematics.x, kinematics.y, kinematics.z, kinematics.pT, kinematics.delta

		if kind == 'num':

			prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2))
			if diff == 'dx': prefactor *= (0.25*pT*y)/(x*z*(1-z))
			elif diff == 'dy': prefactor *= (0.25*pT)/(z*(1-z))
			else: raise ValueError('diff should be dx or dy')

			if weight == '1':
				tt_term = (2-y) * self.get_coeff('A_TT', kinematics)
				lt_term = 0
				prefactor *= 8*(np.pi**3)

			elif weight == 'cos(phi_Dp)':
				tt_term = (2-y) * (delta/pT) * self.get_coeff('B_TT', kinematics)
				lt_term = 0
				prefactor *= 4*(np.pi**3)

			elif weight == 'cos(phi_Dp)cos(phi_kp)' or weight == 'cos(phi_kp)cos(phi_Dp)':
				tt_term = 0
				lt_term = np.sqrt(2-2*y) * (delta/pT) * self.get_coeff('B_LT', kinematics)
				prefactor *= 2*(np.pi**3)

			elif weight == 'sin(phi_Dp)sin(phi_kp)' or weight == 'sin(phi_kp)sin(phi_Dp)':
				tt_term = 0
				lt_term = np.sqrt(2-2*y) * (delta/pT) * self.get_coeff('C_LT', kinematics)
				prefactor *= 2*(np.pi**3)

			elif weight == 'cos(phi_kp)':
				tt_term = 0
				lt_term = np.sqrt(2-2*y) * self.get_coeff('A_LT', kinematics)
				prefactor *= 4*(np.pi**3)

			else:
				raise ValueError(f'weight {weight} not recognized')

			xsec = prefactor*(tt_term + lt_term)
			xsec *= 0.3894*(10**12) # convert to fb
			# xsec *= 0.3894*(10**9) # convert to pb
			return xsec

		elif kind == 'den':

			prefactor = self.alpha_em/(4*(np.pi**2)*(Q**2)*y)
			if diff == 'dx': prefactor *= (0.25*pT*y)/(x*z*(1-z))
			elif diff == 'dy': prefactor *= (0.25*pT)/(z*(1-z))
			else: raise ValueError('diff should be dx or dy')

			if weight == '1':
				tt_term =  (1 + (1-y)**2) * self.get_coeff('A_TT_unpolar', kinematics)
				tmt_term = 0
				ll_term = 4*(1-y)* self.get_coeff('A_LL_unpolar', kinematics)
				prefactor *= 8*(np.pi**3)

			elif weight == 'cos(phi_Dp)':
				tt_term = (1 + (1-y)**2) * (delta/pT) * self.get_coeff('B_TT_unpolar', kinematics)
				tmt_term = 0
				ll_term = 4*(1-y)* (delta/pT)* self.get_coeff('B_LL_unpolar', kinematics)
				prefactor *= 4*(np.pi**3)

			elif weight == 'cos(2*phi_kp)':
				tt_term = 0
				tmt_term = -2*(1-y) * self.get_coeff('A_TmT_unpolar', kinematics)
				ll_term = 0
				prefactor *= 4*(np.pi**3)

			elif weight == 'cos(phi_Dp)cos(2*phi_kp)' or weight == 'cos(2*phi_kp)cos(phi_Dp)':
				tt_term = 0
				tmt_term = -2*(1-y)* (delta/pT) * self.get_coeff('B_TmT_unpolar', kinematics)
				ll_term = 0
				prefactor *= 2*(np.pi**3)

			elif weight == 'sin(phi_Dp)sin(2*phi_kp)' or weight == 'sin(2*phi_kp)sin(phi_Dp)':
				tt_term = 0
				tmt_term = -2*(1-y)* (delta/pT) * self.get_coeff('C_TmT_unpolar', kinematics)
				ll_term = 0
				prefactor *= 2*(np.pi**3)

			else:
				raise ValueError(f'weight {weight} not recognized')

			xsec = prefactor*(tt_term + tmt_term + ll_term)
			xsec *= 0.3894*(10**12) # convert to fb
			# xsec *= 0.3894*(10**9) # convert to pb
			return xsec



	def get_integrated_xsec(self, pT_values, s, phase_space, **options):

		###### setting up parameters 
		method = options.get('method', 'gauss-legendre')
		r0 = options.get('r0', 2.0)
		x0 = options.get('x0', 0.01)
		kind = options.get('kind', 'num')
		points = options.get('points', 10)
		weight = options.get('weight', '1')

		# error checking
		assert method in ['gauss-legendre', 'riemann', 'mc'], f'Error: method = {method} not recognized'
		assert kind in ['num', 'den'], 'Error: kind must be "num" or "den"'
		#############################

		# kinematics = Kinematics(pT=pT, s=s)
		kinematics = Kinematics(s=s)
		integrated = {var: isinstance(space, list) for var, space in phase_space.items()}
		integrate_over_angles = isinstance(phase_space['phi_Dp'], list)

		# choose right function to use
		if integrate_over_angles:
			xsec_func = self.get_angle_integrated_xsec

		else:
			kinematics.phi_Dp = phase_space['phi_Dp']
			kinematics.phi_kp = phase_space['phi_kp']
			xsec_func = self.get_xsec


		results = []
		for pT in pT_values:

			kinematics.pT = pT

			# integrate using one of 3 methods: gaussian quadrature, riemann sums, or monte-carlo
			if method == 'gauss-legendre':

				# Gauss–Legendre nodes, weights on [-1,1]
				nodes, weights = leggauss(points)

				if integrated['z']: 
					z_values, z_w = map_to_interval(nodes, weights, *phase_space['z'])
				else:
					z_values, z_w = [phase_space['z']], [1]

				if integrated['y']:
					y_values, y_w = map_to_interval(nodes, weights, *phase_space['y'])
				else:
					y_values, y_w = [phase_space['y']], [1]

				if integrated['t']:
					t_values, t_w = map_to_interval(nodes, weights, *phase_space['t'])
				else:
					t_values, t_w = [phase_space['t']], [1]

				result = 0.0

				for i, y in enumerate(y_values):
					kinematics.y = y

					for k, z in enumerate(z_values):
						kinematics.z = z

						if integrated['Q2']:
							# Q2_min = max((r0**2)/(z*(1-z)), phase_space['Q2'][0])
							# Q2_max = min(x0 * s * y, phase_space['Q2'][1])
							Q2_min = (r0**2)/(z*(1-z))
							Q2_max = x0 * s * y
							if Q2_max < Q2_min: continue
							Q2_values, Q2_w = map_to_interval(nodes, weights, Q2_min, Q2_max)
						else:
							Q2_values, Q2_w = [phase_space['Q2']], [1]

						for j, Q2 in enumerate(Q2_values):
							kinematics.Q = np.sqrt(Q2)
							x = Q2 / (s * y)
							kinematics.x = x

							# if x > 0.01: continue
							# if np.sqrt(Q2) * np.sqrt(z * (1 - z)) < r0: continue

							for l, t in enumerate(t_values):
								kinematics.delta = np.sqrt(t)

								weight_factor = y_w[i] * Q2_w[j] * z_w[k] * t_w[l]

								result += weight_factor * xsec_func(kinematics, weight=weight, diff='dy', kind=kind)


			elif method == 'riemann':

				if integrated['z']: 
					z_values = np.linspace(*phase_space['z'], points)
					dz = z_values[1]-z_values[0]
				else:
					z_values, dz = [phase_space['z']], 1

				if integrated['y']:
					y_values = np.linspace(*phase_space['y'], points)
					dy = y_values[1]-y_values[0]
				else:
					y_values, dy = [phase_space['y']], 1

				if integrated['t']:
					t_values = np.linspace(*phase_space['t'], points)
					dt = t_values[1]-t_values[0]
				else:
					t_values, dt = [phase_space['t']], 1

				result = 0.0

				for i, y in enumerate(y_values):
					kinematics.y = y

					if integrated['Q2']:
						Q2_max = min(x0 * s * y, phase_space['Q2'][1])
						if Q2_max < phase_space['Q2'][0]: continue
						Q2_values = np.linspace(phase_space['Q2'][0], Q2_max, points)
						dQ2 = Q2_values[1]-Q2_values[0]
					else:
						Q2_values, dQ2 = [phase_space['Q2']], 1

					for j, Q2 in enumerate(Q2_values):
						kinematics.Q = np.sqrt(Q2)
						x = Q2 / (s * y)
						kinematics.x = x

						if x > 0.01: continue

						for k, z in enumerate(z_values):
							if np.sqrt(Q2) * np.sqrt(z * (1 - z)) < r0: continue
							kinematics.z = z

							for l, t in enumerate(t_values):
								kinematics.delta = np.sqrt(t)

								weight_factor = dy * dQ2 * dz * dt
								result += weight_factor * xsec_func(kinematics, weight=weight, diff='dy', kind=kind)

			elif method == 'mc':
			
				rng = np.random.default_rng()
				ran_sum = 0
				for i in range(points):
					if integrated['y']:
						kinematics.y = rng.uniform(*phase_space['y'])
					else:
						kinematics.y = phase_space['y']

					if integrated['t']:
						kinematics.delta = np.sqrt(rng.uniform(*phase_space['t']))
					else:
						kinematics.delta = np.sqrt(phase_space['t'])

					if integrated['z']:
						kinematics.z = rng.uniform(*phase_space['z'])
					else:
						kinematics.z = phase_space['z']

					if integrated['Q2']:
						kinematics.Q = np.sqrt(rng.uniform(*phase_space['Q2']))
					else:
						kinematics.Q = np.sqrt(phase_space['Q2'])

					kinematics.x = (kinematics.Q**2)/(s*kinematics.y)

					if kinematics.x > 0.01: continue
					if np.sqrt((kinematics.Q**2)*kinematics.z*(1-kinematics.z)) < r0: continue

					ran_sum += xsec_func(kinematics, weight=weight, diff='dy', kind=kind)

				box_volume = 1
				for var in ['y', 't', 'Q2', 'z']:
					if integrated[var]: box_volume *= phase_space[var][1]-phase_space[var][0]

				result = ran_sum*(1/points)*box_volume


			results.append(result) 
		return np.array(results)



	########## functions to calculate helicity objects below


	def alpha_s(self, running, s0, s):
		Nc=3.
		Nf=3.
		beta2 = (11*Nc-2*Nf)/(12*np.pi)
		if running==False: return 0.3
		elif running==True: return (np.sqrt(Nc/(2*np.pi))/beta2)*(1/(s0+s))



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
	test_kins.s = 95**2
	test_kins.delta = 0.2
	test_kins.phi_Dp = 0
	test_kins.phi_kp = 0
	test_kins.pT = 10.0
	test_kins.y = (test_kins.Q**2)/(test_kins.s*test_kins.x)

	print(test_kins.y)
	print('pT:', test_kins.pT)
	print('root s', np.sqrt(test_kins.s))

	space = {
		'y' : [0.05, 0.95],
		'z' : [0.2, 0.5],
		'Q2' : [16, 100],
		# 't' : [0.01, 0.04],
		't' : 0.04,
		'phi_Dp' : [0, 2*np.pi],
		'phi_kp' : [0, 2*np.pi]
	}

	dj = DIJET(1, constrained_moments=True)
	dj.load_params('replica_params_pp.csv')
	dj.set_params(4)

	test_den = dj.get_integrated_xsec([test_kins.pT], test_kins.s, space, weight='1', points=7, kind='den')
	test_num = dj.get_integrated_xsec([test_kins.pT], test_kins.s, space, weight='1', points=7, kind='num')

	print(test_num, test_den, test_num/test_den)
	







