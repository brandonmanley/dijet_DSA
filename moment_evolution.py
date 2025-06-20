import numpy as np
from datetime import datetime
import os
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
	
	# initialize amplitudes
	G = np.zeros((nsteps + 1, nsteps + 1))
	Qu = np.zeros((nsteps + 1, nsteps + 1))
	Qd = np.zeros((nsteps + 1, nsteps + 1))
	Qs = np.zeros((nsteps + 1, nsteps + 1))
	G2 = np.zeros((nsteps + 1, nsteps + 1))
	Qtu = np.zeros((nsteps + 1, nsteps + 1))
	Qtd = np.zeros((nsteps + 1, nsteps + 1))
	Qts = np.zeros((nsteps + 1, nsteps + 1))
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
	Itu = np.zeros((nsteps + 1, nsteps + 1))
	Itd = np.zeros((nsteps + 1, nsteps + 1))
	Its = np.zeros((nsteps + 1, nsteps + 1))
	Gm3t = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3u = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3d = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm3s = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm4 = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))
	Gm5 = np.zeros((nsteps + 1, nsteps + 1, nsteps + 1))

	if len(ICs) > 2: raise ValueError('Basis amp IC can only take entry at a time')
	bamp, bvalue = ICs

	if bamp == 'G': G = get_IC(bvalue, nsteps)
	elif bamp == 'Qu': Qu = get_IC(bvalue, nsteps)
	elif bamp == 'Qd': Qd = get_IC(bvalue, nsteps)
	elif bamp == 'Qs': Qs = get_IC(bvalue, nsteps)
	elif bamp == 'G2': G2 = get_IC(bvalue, nsteps)
	elif bamp == 'Qtu': Qtu = get_IC(bvalue, nsteps)
	elif bamp == 'Qtd': Qtd = get_IC(bvalue, nsteps)
	elif bamp == 'Qts': Qts = get_IC(bvalue, nsteps)
	elif bamp == 'I3t': I3t = get_IC(bvalue, nsteps)
	elif bamp == 'I3u': I3u = get_IC(bvalue, nsteps)
	elif bamp == 'I3d': I3d = get_IC(bvalue, nsteps)
	elif bamp == 'I3s': I3s = get_IC(bvalue, nsteps)
	elif bamp == 'I4': I4 = get_IC(bvalue, nsteps)
	elif bamp == 'I5': I5 = get_IC(bvalue, nsteps)
	elif bamp == 'Itu': Itu = get_IC(bvalue, nsteps)
	elif bamp == 'Itd': Itd = get_IC(bvalue, nsteps)
	elif bamp == 'Its': Its = get_IC(bvalue, nsteps)
	else: raise ValueError(f'Unrecognized amp {bamp}')

	# intialize initial conditions
	G0 = np.copy(G)
	Qu0 = np.copy(Qu)
	Qd0 = np.copy(Qd)
	Qs0 = np.copy(Qs)
	G20 = np.copy(G2)
	Qtu0 = np.copy(Qtu)
	Qtd0 = np.copy(Qtd)
	Qts0 = np.copy(Qts)
	I3t0 = np.copy(I3t)
	I3u0 = np.copy(I3u)
	I3d0 = np.copy(I3d)
	I3s0 = np.copy(I3s)
	I40 = np.copy(I4)
	I50 = np.copy(I5)
	Itu0 = np.copy(Itu)
	Itd0 = np.copy(Itd)
	Its0 = np.copy(Its)

 
	print(f'--> coupling: {coupling}')
	print(f'--> amplitudes initialized to {ICs}')
		

	for j in range(nsteps + 1):
		for i in range(j):
			
			Qu[i, j] = Qu[i, j-1] + Qu0[i, j] - Qu0[i, j-1]
			Qu[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qu[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qu[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qu[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBu[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))
			Qu[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i-1))[:, np.newaxis]*(Qu[:i-1, j-i-1:j] + 2*G2[:i-1, j-i-1:j]))

			Qd[i, j] = Qd[i, j-1] + Qd0[i, j] - Qd0[i, j-1]
			Qd[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qd[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qd[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qd[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBd[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))
			Qd[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i-1))[:, np.newaxis]*(Qd[:i-1, j-i-1:j] + 2*G2[:i-1, j-i-1:j]))

			Qs[i, j] = Qs[i, j-1] + Qs0[i, j] - Qs0[i, j-1]
			Qs[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(Qs[i:j-1, j-1] + 2*G2[i:j-1, j-1]))
			Qs[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*G[i:j-1,j-1] + Qs[i:j-1,j-1] + 2*G2[i:j-1,j-1] + 2*GmT[i, i:j-1, j-1] - GmBs[i, i:j-1, j-1] + 2*Gm2[i, i:j-1, j-1]))
			Qs[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i-1))[:, np.newaxis]*(Qs[:i-1, j-i-1:j] + 2*G2[:i-1, j-i-1:j]))

			GmBu[i, i, j] = Qu[i, j]
			for k in range(i+1, j+1):
				GmBu[i, k, j] = GmBu[i, k-1, j-1] + Qu0[i, j] - Qu0[i, j-1]
				GmBu[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qu[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBu[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBu[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qu[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))
			
			GmBd[i, i, j] = Qd[i, j]
			for k in range(i+1, j+1):
				GmBd[i, k, j] = GmBd[i, k-1, j-1] + Qd0[i, j] - Qd0[i, j-1]
				GmBd[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qd[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBd[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBd[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qd[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))

			GmBs[i, i, j] = Qs[i, j]
			for k in range(i+1, j+1):
				GmBs[i, k, j] = GmBs[i, k-1, j-1] + Qs0[i, j] - Qs0[i, j-1]
				GmBs[i, k, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(k-1, j-1))*(Qs[k-1:j-1, j-1] + 2*G2[k-1:j-1, j-1]))
				ii = max(i, k-1)
				GmBs[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*GmT[i, ii:j-1, j-1] - GmBs[i, ii:j-1, j-1] + 2*Gm2[i, ii:j-1, j-1] + 2*G[ii:j-1, j-1] + Qs[ii:j-1, j-1] + 2*G2[ii:j-1, j-1]))

			G[i, j] = G[i, j-1] + G0[i, j] - G0[i, j-1]
			G[i, j] -= Nr*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:i] + Qd[:i, j-i:i] + Qs[:i, j-i:i] + 2*Nf*G2[:i, j-i:i]))
			G[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(3*G[i:j-1, j-1] + 2*G2[i:j-1, j-1] - 2*Nr*(Qtu[i:j-1, j-1] + Qtd[i:j-1, j-1] + Qts[i:j-1, j-1])))
			G[i, j] += d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(GmT[i, i:j-1, j-1] + (2-2*Nr*Nf)*Gm2[i, i:j-1, j-1] - Nr*(GmBu[i, i:j-1, j-1] + GmBd[i, i:j-1, j-1] + GmBs[i, i:j-1, j-1])))
			
			GmT[i, i, j] = G[i, j]
			for k in range(i+1, j+1):
				GmT[i, k, j] = GmT[i, k-1, j-1] + G0[i, j] - G0[i, j-1]
				ii = max(i, k-1)
				GmT[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(GmT[i, ii:j-1, j-1] + (2-2*Nr*Nf)*Gm2[i, ii:j-1, j-1] - Nr*(GmBu[i, ii:j-1, j-1] + GmBd[i, ii:j-1, j-1] + GmBs[i, ii:j-1, j-1]) + 3*G[ii:j-1, j-1] + 2*G2[ii:j-1, j-1] - 2*Nr*(Qtu[ii:j-1, j-1] + Qtd[ii:j-1, j-1] + Qts[ii:j-1, j-1])))
			
			G2[i, j] = G2[i, j-1] + G20[i, j] - G20[i, j-1]
			G2[i, j] += 2*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(G[:i, j-i:j] + 2*G2[:i, j-i:j]))

			Gm2[i, i, j] = G2[i, j] 
			Gm2[i, i+1:j+1, j] = Gm2[i, i:j, j-1] + (G20[i, j] - G20[i, j-1])

			Qtu[i, j] = Qtu[i, j-1] + Qtu0[i, j] - Qtu0[i, j-1]
			Qtu[i, j] -= d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:j] + 2*G2[:i, j-i:j]))

			Qtd[i, j] = Qtd[i, j-1] + Qtd0[i, j] - Qtd0[i, j-1]
			Qtd[i, j] -= d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qd[:i, j-i:j] + 2*G2[:i, j-i:j]))

			Qts[i, j] = Qts[i, j-1] + Qts0[i, j] - Qts0[i, j-1]
			Qts[i, j] -= d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(Qs[:i, j-i:j] + 2*G2[:i, j-i:j]))

			I3u[i, j] = I3u[i, j-1] + I3u0[i, j] - I3u0[i, j-1]
			I3u[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3u[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3u[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))

			I3d[i, j] = I3d[i, j-1] + I3d0[i, j] - I3d0[i, j-1]
			I3d[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3d[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3d[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))

			I3s[i, j] = I3s[i, j-1] + I3s0[i, j] - I3s0[i, j-1]
			I3s[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(4*Gm3t[i, i:j-1, j-1] - 2*Gm3s[i, i:j-1, j-1] - 4*Gm4[i, i:j-1, j-1] + 2*Gm5[i, i:j-1, j-1] - 2*Gm2[i, i:j-1, j-1]))
			I3s[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(4*I3t[:i, j-i:j] - 4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] - 4*G[:i, j-i:j] - 6*G2[:i, j-i:j]))
			
			Gm3u[i, i, j] = I3u[i, j]
			for k in range(i+1, j+1):
				Gm3u[i, k, j] = Gm3u[i, k-1, j-1] + I3u0[i, j] - I3u0[i, j-1]
				ii = max(i, k-1)
				Gm3u[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3u[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))

			Gm3d[i, i, j] = I3d[i, j]
			for k in range(i+1, j+1):
				Gm3d[i, k, j] = Gm3d[i, k-1, j-1] + I3d0[i, j] - I3d0[i, j-1]
				ii = max(i, k-1)
				Gm3d[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3d[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))

			Gm3s[i, i, j] = I3s[i, j]
			for k in range(i+1, j+1):
				Gm3s[i, k, j] = Gm3s[i, k-1, j-1] + I3s0[i, j] - I3s0[i, j-1]
				ii = max(i, k-1)
				Gm3s[i, k, j] += d2*np.sum(alpha_s(cp,deta*np.arange(ii, j-1))*(2*Gm3t[i, ii:j-1, j-1] - Gm3s[i, ii:j-1, j-1] - 2*Gm4[i, ii:j-1, j-1] + Gm5[i, ii:j-1, j-1] - Gm2[i, ii:j-1, j-1]))
			
			I3t[i, j] = I3t[i, j-1] + I3t0[i, j] - I3t0[i, j-1]
			I3t[i, j] += 0.5*d2*np.sum(alpha_s(cp,deta*np.arange(i, j-1))*(2*Gm3t[i, i:j-1, j-1] - 2*Nr*(Gm3u[i, i:j-1, j-1] + Gm3d[i, i:j-1, j-1] + Gm3s[i, i:j-1, j-1])+ (4*Nf*Nr - 4)*Gm4[i, i:j-1, j-1] + (2 - 2*Nr*Nf)*Gm5[i, i:j-1, j-1] + (2*Nr*Nf - 2)*Gm2[i, i:j-1, j-1]))
			I3t[i, j] += 0.5*d2*np.trace(alpha_s(cp,deta*np.arange(i))[:, np.newaxis]*(-2*Nr*(I3u[:i, j-i:j] + I3d[:i, j-i:j] + I3s[:i, j-i:j]) + 4*I3t[:i, j-i:j] + (4*Nr*Nf-4)*I4[:i, j-i:j] + (2 - 2*Nr*Nf)*I5[:i, j-i:j] - 4*G[:i, j-i:j] + (2*Nr - 6)*G2[:i, j-i:j]))
			
			Gm3t[i, i, j] = I3t[i, j]
			for k in range(i+1, j+1):
				Gm3t[i, k, j] = Gm3t[i, k-1, j-1] + I3t0[i, j] - I3t0[i, j-1]
				Gm3t[i, k, j] += d2*np.sum(deta*alpha_s(cp,np.arange(ii, j-1))*(Gm3t[i, ii:j-1, j-1] - Nr*(Gm3u[i, ii:j-1, j-1] + Gm3d[i, ii:j-1, j-1] + Gm3s[i, ii:j-1, j-1]) + (2*Nr*Nf - 2)*Gm4[i, ii:j-1, j-1] + (1 - Nr*Nf)*Gm5[i, ii:j-1, j-1] + (Nr*Nf - 1)*Gm2[i, ii:j-1, j-1]))

			I4[i, j] = I4[i, j-1] + I40[i, j] - I40[i, j-1]
			I4[i, j] += 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(4*I4[:i, j-i:j] + 2*I5[:i, j-i:j] + G2[:i, j-i:j]))
			
			Gm4[i, i, j] = I4[i, j]
			Gm4[i, i+1:j+1, j] = Gm4[i, i:j, j-1] + (I40[i, j] - I40[i, j-1])
			
			I5[i, j] = I5[i, j-1] + I50[i, j] - I50[i, j-1]
			I5[i, j] += 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(-2*I3t[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j] + 2*G[:i, j-i:j] + 3*G2[:i, j-i:j]))
			
			Gm5[i, i, j] = I5[i, j] 
			Gm5[i, i+1:j+1, j] = Gm5[i, i:j, j-1] + (I50[i, j] - I50[i, j-1])

			Itu[i, j] = Itu[i, j-1] + Itu0[i, j] - Itu0[i, j-1]
			Itu[i, j] -= 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(Qu[:i, j-i:j] + 3*G2[:i, j-i:j] - I3u[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j]))

			Itd[i, j] = Itd[i, j-1] + Itd0[i, j] - Itd0[i, j-1]
			Itd[i, j] -= 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(Qd[:i, j-i:j] + 3*G2[:i, j-i:j] - I3d[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j]))

			Its[i, j] = Its[i, j-1] + Its0[i, j] - Its0[i, j-1]
			Its[i, j] -= 0.5*d2*np.trace(deta*alpha_s(cp,np.arange(i))[:, np.newaxis]*(Qs[:i, j-i:j] + 3*G2[:i, j-i:j] - I3s[:i, j-i:j] + 2*I4[:i, j-i:j] - I5[:i, j-i:j]))


		if(np.mod(j, round(nsteps/10)) == 0):
			curr_time = datetime.now().strftime('%H:%M.%S')
			print(str(round((j/round(nsteps))*100,1))+'% done ('+curr_time+')')

	return [Qu, Qd, Qs, G, G2, Qtu, Qtd, Qts, I3u, I3d, I3s, I3t, I4, I5, Itu, Itd, Its]




if __name__ == '__main__':

	output_dir = os.getcwd() + '/evolved_dipoles/'

	eta_max = 15
	deta = 0.5
	nsteps = round(eta_max/deta)
	Nf = 3

	if nsteps > 800: raise ValueError('nsteps > 800 and will probably crash your computer')

	amps = ['Qu', 'Qd', 'Qs', 'G', 'G2', 'Qtu', 'Qtd', 'Qts', 
			'I3u', 'I3d', 'I3s', 'I3t', 'I4', 'I5', 'Itu', 'Itd', 'Its']

	for iamp, amp_name in enumerate(amps):
		for ibasis in ['a', 'b', 'c']:
			
			print('--> nsteps=', nsteps)
			print('--> Staring (d, eta_max, Nf)=', deta, eta_max, Nf)

			IC = [amp_name,  ibasis]
			output = evolve(deta, eta_max, Nf = Nf, ICs = IC, coupling='running')

			# save output
			deta_str = 'd' + str(deta)[2:]
			if not os.path.exists(output_dir+deta_str): 
				os.makedirs(output_dir+deta_str)
			for jamp in range(len(amps)): np.savetxt(output_dir+deta_str+'/'+deta_str+'_NcNf'+str(Nf)+'_'+amp_name+ibasis+'_'+amps[jamp]+'.dat', output[jamp])
			
			print('--> wrote out amplitudes to', output_dir)
			output = 0











	