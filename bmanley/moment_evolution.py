import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import time

# here lambda is a cutoff, also includes Qt corrections from 2406.11647 and It amplitude from 2410.21260
# this one has running coupling

def alpha_s(coupling, s):
	Nc = 3.0
	Nf = 3.0
	beta2 = (11*Nc-2*Nf)/(12*np.pi)
	s0 = 1.0 # log ratio of IR cutoff to \Lambda_{QCD}
	
	if coupling == 'fixed': return 0.25
	elif coupling == 'running': return (np.sqrt(Nc/(2*np.pi))/beta2)*(1/(s0+s))


def get_IC(ip = 'a', nsteps=-1):
	if ip == 'a': return np.ones((nsteps+1, 1))*np.arange(nsteps+1)
	elif ip == 'b': return np.arange(nsteps+1)[:, None]*np.ones(nsteps+1)
	elif ip == 'c': return np.ones((nsteps + 1, nsteps + 1))
		

def evolve(deta, eta_max, Nf=3.0, ICs='ones', coupling='fixed', s0=1.0):

	Nc = 3.0
	Nr = 0.25*(1.0/Nc)

	nsteps = round(eta_max/deta)
	d2 = deta**2
	cp = coupling
	
	G = np.zeros((nsteps + 1, nsteps + 1))
	Qu = np.zeros((nsteps + 1, nsteps + 1))
	Qd = np.zeros((nsteps + 1, nsteps + 1))
	Qs = np.zeros((nsteps + 1, nsteps + 1))
	G2 = np.zeros((nsteps + 1, nsteps + 1))
	Qt = np.zeros((nsteps + 1, nsteps + 1))
	GmT = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	GmBu = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	GmBd = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	GmBs = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm2 = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))

	I3t = np.zeros((nsteps + 1, nsteps + 1))
	I3u = np.zeros((nsteps + 1, nsteps + 1))
	I3d = np.zeros((nsteps + 1, nsteps + 1))
	I3s = np.zeros((nsteps + 1, nsteps + 1))
	I4 = np.zeros((nsteps + 1, nsteps + 1))
	I5 = np.zeros((nsteps + 1, nsteps + 1))
	It = np.zeros((nsteps + 1, nsteps + 1))
	Gm3t = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3u = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3d = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3s = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm4 = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm5 = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))

	if ICs == 'ones':
		G += 1
		Qu += 1
		Qd += 1
		Qs += 1
		G2 += 1
		Qt += 1
		I3t += 1
		I3u += 1
		I3d += 1
		I3s += 1
		I4 += 1
		I5 += 1
		It += 1

	# basis amp
	elif isinstance(ICs, dict):
		if len(ICs) > 1: raise ValueError('Basis amp IC can only take entry at a time')
		
		bamp = list(ICs.keys())[0]
		bvalue = list(ICs.values())[0]

		if bamp == 'G': G = get_IC(bvalue, nsteps)
		elif bamp == 'Qu': Qu = get_IC(bvalue, nsteps)
		elif bamp == 'Qd': Qd = get_IC(bvalue, nsteps)
		elif bamp == 'Qs': Qs = get_IC(bvalue, nsteps)
		elif bamp == 'G2': G2 = get_IC(bvalue, nsteps)
		elif bamp == 'Qt': Qt = get_IC(bvalue, nsteps)
		elif bamp == 'I3t': I3t = get_IC(bvalue, nsteps)
		elif bamp == 'I3u': I3u = get_IC(bvalue, nsteps)
		elif bamp == 'I3d': I3d = get_IC(bvalue, nsteps)
		elif bamp == 'I3s': I3s = get_IC(bvalue, nsteps)
		elif bamp == 'I4': I4 = get_IC(bvalue, nsteps)
		elif bamp == 'I5': I5 = get_IC(bvalue, nsteps)
		elif bamp == 'It': It = get_IC(bvalue, nsteps)

	else: raise ValueError(f'Initial coniditon ({ICs}) not recognized')
 
	# print(f'--> eta_max = {eta_max}, deta = {deta}, Nf = {Nf}')
	print(f'--> coupling: {coupling}')
	print(f'--> amplitudes initialized to {ICs}')
		
	for j in range(1, nsteps + 1):
		for i in range(j):
			
			Qu[i, j] = Qu[i, j-1]
			Qu[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qu[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qu[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:j] + 2*G2[:i, j-i:j]))
			Qu[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qu[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBu[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))

			Qd[i, j] = Qd[i, j-1]
			Qd[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qd[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qd[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qd[:i, j-i:j] + 2*G2[:i, j-i:j]))
			Qd[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qd[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBd[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))

			Qs[i, j] = Qs[i, j-1]
			Qs[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qs[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qs[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qs[:i, j-i:j] + 2*G2[:i, j-i:j]))
			Qs[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qs[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBs[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))
			
			GmBu[i, i, j] = Qu[i, j]
			for k in range(i+1, j+1):
				GmBu[i, k, j] = GmBu[i, k-1, j-1]
				GmBu[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qu[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBu[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBu[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qu[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))
			
			GmBd[i, i, j] = Qd[i, j]
			for k in range(i+1, j+1):
				GmBd[i, k, j] = GmBd[i, k-1, j-1]
				GmBd[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qd[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBd[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBd[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qd[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))

			GmBs[i, i, j] = Qs[i, j]
			for k in range(i+1, j+1):
				GmBs[i, k, j] = GmBs[i, k-1, j-1]
				GmBs[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qs[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBs[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBs[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qs[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))

			G[i, j] = G[i, j-1]
			G[i, j] -= Nr*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:i] + Qd[:i, j-i:i] + Qs[:i, j-i:i] + 2*Nf*G2[:i, j-i:i]))
			G[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(3*G[i:j-1, j-1] + 2*G2[i:j-1, j-1] - 2*Nr*Nf*Qt[i:j-1, j-1]))
			G[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(GmT[i, i:j-1, j-1] + (2-2*Nr*Nf)*Gm2[i, i:j-1, j-1] - Nr*(GmBu[i, i:j-1, j-1] + GmBd[i, i:j-1, j-1] + GmBs[i, i:j-1, j-1])))
			
			GmT[i, i, j] = G[i, j]
			for k in range(i+1, j+1):
				GmT[i, k, j] = GmT[i, k-1, j-1]
				ii = max(i, k-1)
				GmT[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(GmT[i, ii:j-1, j-1] + (2-2*Nr*Nf)*Gm2[i, ii:j-1, j-1] - Nr*(GmBu[i, ii:j-1, j-1] + GmBd[i, ii:j-1, j-1] + GmBs[i, ii:j-1, j-1]) + 3*G[ii:j-1, j-1] + 2*G2[ii:j-1, j-1] - 2*Nr*Nf*Qt[ii:j-1, j-1]))
			
			G2[i, j] = G2[i, j-1]
			G2[i, j] += 2*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(G[:i, j-i:j] + 2*G2[:i, j-i:j]))

			Gm2[i, i, j] = G2[i, j]
			Gm2[i, i+1:j+1, j] = Gm2[i, i:j, j-1]

			Qt[i, j] = Qt[i, j-1]
			Qt[i, j] -= d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:j] + Qd[:i, j-i:j] + Qs[:i, j-i:j] + 2*G2[:i, j-i:j]))
			  
			I3u[i, j] = I3u[i, j-1]
			I3u[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3u[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3u[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))

			I3d[i, j] = I3d[i, j-1]
			I3d[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3d[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3d[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))

			I3s[i, j] = I3s[i, j-1]
			I3s[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3s[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3s[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))
			
			Gm3u[i, i, j] = I3u[i, j]
			for k in range(i+1, j+1):
				Gm3u[i, k, j] = Gm3u[i, k-1, j-1]
				ii = max(i, k-1)
				Gm3u[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3u[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))

			Gm3d[i, i, j] = I3d[i, j]
			for k in range(i+1, j+1):
				Gm3d[i, k, j] = Gm3d[i, k-1, j-1]
				ii = max(i, k-1)
				Gm3d[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3d[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))

			Gm3s[i, i, j] = I3s[i, j]
			for k in range(i+1, j+1):
				Gm3s[i, k, j] = Gm3s[i, k-1, j-1]
				ii = max(i, k-1)
				Gm3s[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3s[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))
			
			I3t[i, j] = I3t[i, j-1]
			I3t[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*Gm3t[i, i:j-1, j-1] - 2*Nr*(Gm3u[i, i:j-1, j-1] + Gm3d[i, i:j-1, j-1] + Gm3s[i, i:j-1, j-1])+ (4*Nf*Nr - 4)*Gm4[i, i:j-1, j-1] + (2 - 2*Nr*Nf)*Gm5[i, i:j-1, j-1] + (2*Nr*Nf - 2)*Gm2[i, i:j-1, j-1]))
			I3t[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(-2*Nr*(I3u[:i, j-i:j] + I3d[:i, j-i:j] + I3s[:i, j-i:j]) + 4*I3t[:i, j-i:j] + (4*Nr*Nf-4)*I4[:i, j-i:j] + (2 - 2*Nr*Nf)*I5[:i, j-i:j] - 4*G[:i, j-i:j] + (2*Nr - 6)*G2[:i, j-i:j]))
			
			Gm3t[i, i, j] = I3t[i, j]
			for k in range(i+1, j+1):
				Gm3t[i, k, j] = Gm3t[i, k-1, j-1]
				Gm3t[i, k, j] += d2*np.sum(deta*alpha_s(cp,np.arange(ii, j-1))*(Gm3t[i, ii:j-1, j-1] - Nr*(Gm3u[i, ii:j-1, j-1] + Gm3d[i, ii:j-1, j-1] + Gm3s[i, ii:j-1, j-1]) + (2*Nr*Nf - 2)*Gm4[i, ii:j-1, j-1] + (1 - Nr*Nf)*Gm5[i, ii:j-1, j-1] + (Nr*Nf - 1)*Gm2[i, ii:j-1, j-1]))

			I4[i, j] = I4[i, j-1]
			I4[i, j] += 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] + G2[:i, j-i:j]))
			
			Gm4[i, i, j] = I4[i, j]
			Gm4[i, i+1:j+1, j] = Gm4[i, i:j, j-1]
			
			I5[i, j] = I5[i, j-1]
			I5[i, j] += 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(-2*I3t[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j] + 2*G[:i, j-i:j] + 3*G2[:i, j-i:j]))
			
			Gm5[i, i, j] = I5[i, j]
			Gm5[i, i+1:j+1, j] = Gm5[i, i:j, j-1]

			It[i, j] = It[i, j-1]
			It[i, j] -= 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:j] + Qd[:i, j-i:j] + Qs[:i, j-i:j] + 3*G2[:i, j-i:j] - I3u[:i, j-i:j] - I3d[:i, j-i:j] - I3s[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j]))
				
		if(np.mod(j, round(nsteps/10)) == 0):
			now = datetime.now()
			curr_time = now.strftime('%H:%M.%S')
			print(str(round((j/round(nsteps))*100,1))+'% done ('+curr_time+')')

	return [Qu, Qd, Qs, G, G2, Qt, I3u, I3d, I3s, I3t, I4, I5, It]





if __name__ == '__main__':

	output_dir = '/Users/brandonmanley/Desktop/PhD/moment_evolution/evolved_dipoles/largeNc&Nf/'
	ICs_tag = 'basis'

	eta_max = 15
	deta = 0.05
	Nf = 3

	if round(eta_max/deta) > 800: raise ValueError('nsteps > 800 and will probably crash your computer')

	amps = {0:'Qu', 1:'Qd', 2:'Qs', 3:'G', 4:'G2', 5:'Qt', 6:'I3u', 7:'I3d', 8:'I3s', 9:'I3t', 10:'I4', 11:'I5', 12:'It'}

	for iamp, amp_name in amps.items():
		for ibasis in ['a', 'b', 'c']:
			
			print('--> nsteps:', round(eta_max/deta)) 
			deta_str = 'd' + str(deta)[2:] + '_'
			amp_strs = {0:'Q', 1:'G', 2:'G2', 3:'Qt', 4:'I3', 5:'I3t', 6:'I4', 7:'I5', 8:'It'}
			print('-> Staring (d, eta_max, Nf)=', deta, eta_max, Nf)

			IC = {amp_name: ibasis}
			output = evolve(deta, eta_max, Nf = Nf, ICs = IC, coupling='running')

			# save output
			if not os.path.exists(output_dir+deta_str+ICs_tag): os.mkdir(output_dir+deta_str+ICs_tag)
			for jamp in range(len(amps)): np.savetxt(output_dir+deta_str+ICs_tag+'/'+deta_str+'NcNf'+str(Nf)+'_'+amp_name+ibasis+'_'+amps[jamp]+'.dat', output[jamp])
			
			print('--> wrote out amplitudes to', output_dir)
			output = 0











	