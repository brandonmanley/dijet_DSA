import sys
import os
import numpy as np
import copy
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


def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)


def G0func(s10,eta,var): #function that returns the possible Born level initial conditions
	if var=='s10': return s10
	if var=='eta': return eta
	if var=='1': return 1.
	if var=='0': return 0.


def get_Qu0(deta,ds,eta_max,s_max,G,var):#initial condtion for Q_u
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G=='Qu':
		var2=var
	else:
		var2='0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_GamBu0(deta,ds,eta_max,s_max,G,var):#initial conditon for Gamma Bar_u
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G=='Qu':
		var2=var
	else:
		var2='0'
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_Qd0(deta,ds,eta_max,s_max,G,var):#initial conditon for Q_d
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G=='Qd':
		var2=var
	else:
		var2='0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0



def get_QTu0(deta, ds, eta_max, s_max, G, var): # initial conditon for QTu
	nsteps = int(eta_max/deta)
	I = np.arange(nsteps +1)
	s10_arr = I*ds
	eta_arr = I*deta
	G0 = np.zeros((s10_arr.size,eta_arr.size))
	
	if G == 'QTu': var2 = var
	else: var2 = '0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j] = G0func(s10_arr[i], eta_arr[j], var2)
	
	return G0



def get_QTd0(deta, ds, eta_max, s_max, G, var): # initial conditon for QTd
	nsteps = int(eta_max/deta)
	I = np.arange(nsteps +1)
	s10_arr = I*ds
	eta_arr = I*deta
	G0 = np.zeros((s10_arr.size,eta_arr.size))
	
	if G == 'QTd': var2 = var
	else: var2 = '0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j] = G0func(s10_arr[i], eta_arr[j], var2)
	
	return G0



def get_QTs0(deta, ds, eta_max, s_max, G, var): # initial conditon for QTu
	nsteps = int(eta_max/deta)
	I = np.arange(nsteps +1)
	s10_arr = I*ds
	eta_arr = I*deta
	G0 = np.zeros((s10_arr.size,eta_arr.size))
	
	if G == 'QTs': var2 = var
	else: var2 = '0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j] = G0func(s10_arr[i], eta_arr[j], var2)
	
	return G0


def get_GamBd0(deta,ds,eta_max,s_max,G,var):#initial conditon for Gamma Bar_d
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps + 1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G=='Qd':
		var2=var
	else:
		var2='0'
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_Qs0(deta,ds,eta_max,s_max,G,var):#initial conditon for Q_s
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G=='Qs':
		var2=var
	else:
		var2='0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_GamBs0(deta,ds,eta_max,s_max,G,var):#initial conditon for Gamma Bar_s
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G=='Qs':
		var2=var
	else:
		var2='0'
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_GT0(deta,ds,eta_max,s_max,G,var):#initial conditon for G~
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G=='GT':
		var2=var
	else:
		var2='0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_GamT0(deta,ds,eta_max,s_max,G,var):#initial conditon for Gamma~
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G=='GT':
		var2=var
	else:
		var2='0'
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_G20(deta,ds,eta_max,s_max,G,var):#initial conditon for G_2
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G=='G2':
		var2=var
	else:
		var2='0'
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def get_Gam20(deta,ds,eta_max,s_max,G,var):#initial conditon for Gamma_2
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G=='G2':
		var2=var
	else:
		var2='0'
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


# initial conditions for moments (can be used for all dipoles)
def get_ics_dipole(deta, ds, eta_max, s_max, G, inG, var): 
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	eta_arr=I*deta
	
	G0=np.zeros((s10_arr.size,eta_arr.size))
	if G==inG:
		var2=var
	else:
		return G0
			
	for i in range(0,s10_arr.size):
		for j in range(0,eta_arr.size):
			G0[i,j]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


# initial conditions for moments (can be used for all neighbors)
def get_ics_neighbor(deta,ds,eta_max,s_max,G,inG,var):
	nsteps=int(eta_max/deta)
	I=np.arange(nsteps +1)
	s10_arr=I*ds
	s20_arr=I*ds
	eta_arr=I*deta

	G0=np.zeros((s10_arr.size,s20_arr.size,eta_arr.size))
	
	if G==inG:
		var2=var
	else:
		return G0
		
	for i in range(0,s10_arr.size):
		for j in range(0,s10_arr.size):
			for k in range(0,eta_arr.size):
				G0[i,j,k]=G0func(s10_arr[i],eta_arr[j],var2)
	
	return G0


def alpha_s(running, s0, s):
	Nc=3.
	Nf=3.
	beta2 = (11*Nc-2*Nf)/(12*np.pi) 
	if running==False:
		return 0.3
	elif running ==True:
		return (np.sqrt(Nc/(2*np.pi))/beta2)*(1/(s0+s))
		


def Mevo_corrected(deta,ds,eta_max,s_max,eta0,G,var,running=False, s0=1., ckernel=1.0):#Evolution algorithm
	
	eta0_index = int(np.ceil(eta0/deta))
	Nc=3.0 #Number ofcolors
	Nf=3.0 #Number of flavours
	Nr = 0.25*(Nf/Nc)
	
	#initialization
	Qu0   = get_Qu0(deta,ds,eta_max,s_max,G,var)
	GamBu = get_GamBu0(deta,ds,eta_max,s_max,G,var)
	Qu    = np.copy(Qu0)
	Qd0   = get_Qd0(deta,ds,eta_max,s_max,G,var)
	GamBd = get_GamBd0(deta,ds,eta_max,s_max,G,var)
	Qd    = np.copy(Qd0)
	Qs0   = get_Qs0(deta,ds,eta_max,s_max,G,var)
	GamBs = get_GamBs0(deta,ds,eta_max,s_max,G,var)
	Qs   = np.copy(Qs0)
	GT0   = get_GT0(deta,ds,eta_max,s_max,G,var)
	GamT = get_GamT0(deta,ds,eta_max,s_max,G,var)
	GT    = np.copy(GT0)
	G20   = get_G20(deta,ds,eta_max,s_max,G,var)
	Gam2 = get_Gam20(deta,ds,eta_max,s_max,G,var)
	G2    = np.copy(G20)
	QTu0   = get_QTu0(deta,ds,eta_max,s_max,G,var)
	QTu    = np.copy(QTu0)
	QTd0   = get_QTd0(deta,ds,eta_max,s_max,G,var)
	QTd    = np.copy(QTd0)
	QTs0   = get_QTs0(deta,ds,eta_max,s_max,G,var)
	QTs    = np.copy(QTs0)
	
	I3u0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I3u', var)
	I3d0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I3d', var)
	I3s0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I3s', var)
	I3T0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I3T', var)
	I40 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I4', var)
	I50 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'I5', var)
	ITu0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'ITu', var)
	ITd0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'ITd', var)
	ITs0 = get_ics_dipole(deta, ds, eta_max, s_max, G, 'ITs', var)

	I3u = np.copy(I3u0)
	I3d = np.copy(I3d0)
	I3s = np.copy(I3s0)
	I3T = np.copy(I3T0)
	I4 = np.copy(I40)
	I5 = np.copy(I50)
	ITu = np.copy(ITu0)
	ITd = np.copy(ITd0)
	ITs = np.copy(ITs0)

	Gam3u = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I3u', var)
	Gam3d = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I3d', var)
	Gam3s = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I3s', var)
	Gam3T = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I3T', var)
	Gam4 = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I4', var)
	Gam5 = get_ics_neighbor(deta, ds, eta_max, s_max, G, 'I5', var)
		
	#ckernel=1.0
	for j in range(eta0_index, Qu.shape[0]):
		for i in range(0,j-1-eta0_index +1): #range(1,np.int(np.ceil((j*deta-eta0)/deta)) +1):
			
			#Q sums
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum = ipsum + alpha_s(running, s0, ip*ds)*(1.5*Qu[ip,j-1]-GamBu[i,ip,j-1]+2*GT[ip,j-1]+2*GamT[i,ip,j-1]+3*G2[ip,j-1]+2*Gam2[i,ip,j-1])
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum = jpsum + alpha_s(running, s0, (i+jp-j+1)*ds)*(Qu[i+jp-j+1,jp]+2*G2[i+jp-j+1,jp])
			Qu[i,j] = Qu[i,j-1]+Qu0[i,j]-Qu0[i,j-1]+(deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(1.5*Qd[ip,j-1]-GamBd[i,ip,j-1]+2*GT[ip,j-1]+2*GamT[i,ip,j-1]+3*G2[ip,j-1]+2*Gam2[i,ip,j-1])            
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum = jpsum + alpha_s(running, s0, (i+jp-j+1)*ds)*(Qd[i+jp-j+1,jp]+2*G2[i+jp-j+1,jp])
			Qd[i,j] = Qd[i,j-1]+Qd0[i,j]-Qd0[i,j-1]+(deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(1.5*Qs[ip,j-1]-GamBs[i,ip,j-1]+2*GT[ip,j-1]+2*GamT[i,ip,j-1]+3*G2[ip,j-1]+2*Gam2[i,ip,j-1])            
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum = jpsum + alpha_s(running,  s0, (i+jp-j+1)*ds)*(Qs[i+jp-j+1,jp]+2*G2[i+jp-j+1,jp])
			Qs[i,j] = Qs[i,j-1]+Qs0[i,j]-Qs0[i,j-1]+(deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			#GT sums
			ipsum=0
			for ip in range(i,j-2-eta0_index+ 1):
				ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(3*GT[ip,j-1] + GamT[i,ip,j-1] - (1./(4.*Nc))*(GamBu[i,ip,j-1]+GamBd[i,ip,j-1]+GamBs[i,ip,j-1]) + 2*G2[ip,j-1] + (2-Nf/(2*Nc))*Gam2[i,ip,j-1] - (1./(4.*Nc))*(QTu[ip,j-1] + QTd[ip,j-1] + QTs[ip,j-1]))
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum = jpsum + alpha_s(running,  s0, (i+jp-j+1)*ds)*(Qu[i+jp-j+1,jp] + Qd[i+jp-j+1,jp] + Qs[i+jp-j+1,jp] + 2*Nf*G2[i+jp-j+1,jp])
			GT[i,j] = GT[i,j-1] + GT0[i,j] - GT0[i,j-1] + (deta**2)*ipsum -(1./(4.*Nc))*(deta**2)*jpsum
			
			#G2 sum
			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum = jpsum + alpha_s(running,  s0, (i+jp-j+1)*ds)*(GT[i+jp-j+1,jp]+2*G2[i+jp-j+1,jp])
			G2[i,j] = G2[i,j-1] + G20[i,j] - G20[i,j-1] + 2*(deta**2)*jpsum
			
			
			# I3 sums
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3u[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running, s0, (i+jp-j+1)*ds)*(4*I3T[i+jp-j+1,jp] - 4*I4[i+jp-j+1,jp] + 2*I5[i+jp-j+1,jp] - 4*GT[i+jp-j+1,jp] - 6*G2[i+jp-j+1,jp])
			I3u[i,j] = I3u[i,j-1]+I3u0[i,j]-I3u0[i,j-1] + (deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3d[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running, s0, (i+jp-j+1)*ds)*(4*I3T[i+jp-j+1,jp] - 4*I4[i+jp-j+1,jp] + 2*I5[i+jp-j+1,jp] - 4*GT[i+jp-j+1,jp] - 6*G2[i+jp-j+1,jp])
			I3d[i,j] = I3d[i,j-1]+I3d0[i,j]-I3d0[i,j-1]+(deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			ipsum=0
			for ip in range(i,j-2-eta0_index +1):
				ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3s[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running, s0, (i+jp-j+1)*ds)*(4*I3T[i+jp-j+1,jp] - 4*I4[i+jp-j+1,jp] + 2*I5[i+jp-j+1,jp] - 4*GT[i+jp-j+1,jp] - 6*G2[i+jp-j+1,jp])
			I3s[i,j] = I3s[i,j-1]+I3s0[i,j]-I3s0[i,j-1]+(deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			
			# I3T sums
			ipsum=0
			for ip in range(i,j-2-eta0_index+ 1):
				ipsum += alpha_s(running,  s0, ip*ds)*(Gam3T[i,ip,j-1] - (0.25/Nc)*(Gam3u[i,ip,j-1] + Gam3d[i,ip,j-1] + Gam3s[i,ip,j-1]) + (2*Nr - 2)*Gam4[i,ip,j-1] + (1.0 - Nr)*Gam5[i,ip,j-1] + (Nr -1)*Gam2[i,ip,j-1])
			jpsum=0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(-(0.5/Nc)*(I3u[i+jp-j+1,jp] + I3d[i+jp-j+1,jp] + I3s[i+jp-j+1,jp]) + 4*I3T[i+jp-j+1,jp] + (4*Nr-4)*I4[i+jp-j+1,jp] + (2-2*Nr)*I5[i+jp-j+1,jp] - 4*GT[i+jp-j+1,jp] + (2*Nr - 6)*G2[i+jp-j+1,jp])
			I3T[i,j] = I3T[i,j-1] + I3T0[i,j] - I3T0[i,j-1] + (deta**2)*ipsum + 0.5*(deta**2)*jpsum
			
			
			# I4 sum
			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(4*I4[i+jp-j+1,jp] + 2*I5[i+jp-j+1,jp] - 2*GT[i+jp-j+1,jp] - 3*G2[i+jp-j+1,jp])
			I4[i,j] = I4[i,j-1] + I40[i,j] - I40[i,j-1] + 0.5*(deta**2)*jpsum
			
			# I5 sum
			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(-2*I3T[i+jp-j+1,jp] + 2*I4[i+jp-j+1,jp] - I5[i+jp-j+1,jp] + 4*GT[i+jp-j+1,jp] + 7*G2[i+jp-j+1,jp])
			I5[i,j] = I5[i,j-1] + I50[i,j] - I50[i,j-1] + 0.5*(deta**2)*jpsum

			# IT sums
			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(Qu[i+jp-j+1,jp] + 3*G2[i+jp-j+1,jp] - I3u[i+jp-j+1,jp] + 2*I4[i+jp-j+1,jp] - I5[i+jp-j+1,jp])
			ITu[i,j] = ITu[i,j-1] + ITu0[i,j] - ITu0[i,j-1] - 0.5*(deta**2)*jpsum

			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(Qd[i+jp-j+1,jp] + 3*G2[i+jp-j+1,jp] - I3d[i+jp-j+1,jp] + 2*I4[i+jp-j+1,jp] - I5[i+jp-j+1,jp])
			ITd[i,j] = ITd[i,j-1] + ITd0[i,j] - ITd0[i,j-1] - 0.5*(deta**2)*jpsum


			jpsum = 0
			for jp in range(j-1-i,j-2 +1):
				jpsum += alpha_s(running,  s0, (i+jp-j+1)*ds)*(Qs[i+jp-j+1,jp] + 3*G2[i+jp-j+1,jp] - I3s[i+jp-j+1,jp] + 2*I4[i+jp-j+1,jp] - I5[i+jp-j+1,jp])
			ITs[i,j] = ITs[i,j-1] + ITs0[i,j] - ITs0[i,j-1] - 0.5*(deta**2)*jpsum


			
			GamBu[i,i,j]=Qu[i,j]
			GamBd[i,i,j]=Qd[i,j]
			GamBs[i,i,j]=Qs[i,j]
			GamT[i,i,j]=GT[i,j]
			Gam2[i,i,j]=G2[i,j]
			
			Gam3u[i,i,j]=I3u[i,j]
			Gam3d[i,i,j]=I3d[i,j]
			Gam3s[i,i,j]=I3s[i,j]
			Gam4[i,i,j]=I4[i,j]
			Gam5[i,i,j]=I5[i,j]
			
			for k in range(i+1,j-eta0_index +1):
				#Gamma Bar sums
				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(1.5*Qu[ip,j-1]-GamBu[i,ip,j-1] + 2*GT[ip,j-1] + 2*GamT[i,ip,j-1] + 3*G2[ip,j-1] + 2*Gam2[i,ip,j-1])
				GamBu[i,k,j]=GamBu[i,k-1,j-1]+Qu0[i,j]-Qu0[i,j-1]+(deta**2)*ipsum
				
				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum = ipsum + alpha_s(running, s0, ip*ds)*(1.5*Qd[ip,j-1]-GamBd[i,ip,j-1] + 2*GT[ip,j-1] + 2*GamT[i,ip,j-1] + 3*G2[ip,j-1] + 2*Gam2[i,ip,j-1])
				GamBd[i,k,j]=GamBd[i,k-1,j-1]+Qd0[i,j]-Qd0[i,j-1]+(deta**2)*ipsum
				
				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(1.5*Qs[ip,j-1]-GamBs[i,ip,j-1] + 2*GT[ip,j-1] + 2*GamT[i,ip,j-1] + 3*G2[ip,j-1] + 2*Gam2[i,ip,j-1])
				GamBs[i,k,j]=GamBs[i,k-1,j-1]+Qs0[i,j]-Qs0[i,j-1]+(deta**2)*ipsum
				
				#Gamma tilde sums
				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum = ipsum + alpha_s(running,  s0, ip*ds)*(3*GT[ip,j-1]+GamT[i,ip,j-1]+2*G2[ip,j-1]+(2-Nf/(2*Nc))*Gam2[i,ip,j-1]-(1./(4*Nc))*(GamBu[i,ip,j-1]+GamBd[i,ip,j-1]+GamBs[i,ip,j-1]) -(1./(4*Nc))*(QTu[ip,j-1]+QTd[ip,j-1]+QTs[ip,j-1]))
				GamT[i,k,j]=GamT[i,k-1,j-1]+GT0[i,j]-GT0[i,j-1]+(deta**2)*ipsum
				
				#Gamma2
				Gam2[i,k,j]=Gam2[i,k-1,j-1]+G20[i,j]-G20[i,j-1]
				
		
				# Gamma3 sums
				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3u[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
				Gam3u[i,k,j] = Gam3u[i,k-1,j-1]+I3u0[i,j]-I3u0[i,j-1]+(deta**2)*ipsum

				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3d[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
				Gam3d[i,k,j] = Gam3d[i,k-1,j-1]+I3d0[i,j]-I3d0[i,j-1]+(deta**2)*ipsum

				ipsum=0
				for ip in range(k-1,j-2-eta0_index +1):
					ipsum += alpha_s(running, s0, ip*ds)*(2*Gam3T[i,ip,j-1] - Gam3s[i,ip,j-1] - 2*Gam4[i,ip,j-1] + Gam5[i,ip,j-1] - Gam2[i,ip,j-1])
				Gam3s[i,k,j] = Gam3s[i,k-1,j-1]+I3s0[i,j]-I3s0[i,j-1]+(deta**2)*ipsum
				
				# Gamma3T sums
				ipsum=0
				for ip in range(k-1,j-2-eta0_index+ 1):
					ipsum += alpha_s(running,  s0, ip*ds)*(Gam3T[i,ip,j-1] - (0.25/Nc)*(Gam3u[i,ip,j-1] + Gam3d[i,ip,j-1] + Gam3s[i,ip,j-1]) + (2*Nr - 2)*Gam4[i,ip,j-1] + (1-Nr)*Gam5[i,ip,j-1] + (Nr -1)*Gam2[i,ip,j-1])
				Gam3T[i,k,j] = Gam3T[i,k-1,j-1] + I3T0[i,j] - I3T0[i,j-1] + (deta**2)*ipsum
				
				# Gamma4,5
				Gam4[i,k,j] = Gam4[i,k-1,j-1]+I40[i,j]-I40[i,j-1]
				Gam5[i,k,j] = Gam5[i,k-1,j-1]+I50[i,j]-I50[i,j-1]
				
			
			
	output={}
	output['Qu'] = Qu
	output['Qd'] = Qd
	output['Qs'] = Qs
	output['GT'] = GT
	output['G2'] = G2
	output['QTu'] = QTu
	output['QTd'] = QTd
	output['QTs'] = QTs

	output['I3u'] = I3u
	output['I3d'] = I3d
	output['I3s'] = I3s
	output['I3T'] = I3T
	output['I4'] = I4
	output['I5'] = I5
	output['ITu'] = ITu
	output['ITd'] = ITd
	output['ITs'] = ITs
	
	return output


def precook_Mevo_corrected(wdir, deta, eta_max, x0, eta0, s0, running, ckernel=1):
	Dipoles = ['Qu','Qd','Qs','GT','G2','QTu','QTd','QTs'] #List of Dipoles
	Moments = ['I3u', 'I3d', 'I3s', 'I3T', 'I4', 'I5', 'ITu', 'ITd', 'ITs'] # list of moments
	ICs = ['eta','s10','1'] # list of initial conditions
	
	for G in Dipoles+Moments:
		for var in ICs:
			precook = Mevo_corrected(deta,deta,eta_max,eta_max,eta0,G,var, running, s0, ckernel=ckernel) #be careful when messing with
			for component in Dipoles+Moments:
				checkdir('%s/pre-cooked_corrected_moments/pre-cooked-x0_%f-deta_%f-etamax_%f-s0_%f-running_%s-ckernel_%f' % (wdir, x0, deta,eta_max, s0, running, ckernel))
				save(precook[component],'%s/pre-cooked_corrected_moments/pre-cooked-x0_%f-deta_%f-etamax_%f-s0_%f-running_%s-ckernel_%f/pre-cook-%s-%s-%s.dat' % (wdir, x0, deta, eta_max, s0, running, ckernel, G, var, component))
				print('---> cooked '+'%s/pre-cooked_corrected_moments/pre-cooked-x0_%f-deta_%f-etamax_%f-s0_%f-running_%s-ckernel_%f/pre-cook-%s-%s-%s.dat' % (wdir, x0, deta, eta_max, s0, running, ckernel, G, var, component))
		 


if __name__ == '__main__':

	wdir = 'dipoles'
	Nc = 3.0
	deta = 0.025
	eta_max = 15.0
	x0 = 0.1
	eta0 = np.sqrt(Nc/(2*np.pi))*np.log(1/x0)
	s0 = 1.0
	running = True


	precook_Mevo_corrected(wdir, deta, eta_max, x0, eta0, s0, running)














  

