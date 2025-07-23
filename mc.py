import numpy as np
import dijet
import json

def get_particles(kins):

	# choose type of dipole frame here
	P_plus = np.sqrt(0.5)*np.sqrt(kins.s)
	q_minus = kins.y*np.sqrt(0.5)*np.sqrt(kins.s)

	# pick \phi_k = 0 since it does not matter
	phi_k = 0

	# jet kinematics
	p1_perp = np.sqrt((kins.z**2)*kins.delta**2  + kins.pT**2 + 2*kins.z*kins.pT*kins.delta*np.cos(kins.phi_Dp))
	p1_minus = kins.z*q_minus
	p1_plus = (p1_perp**2)/(2*p1_minus)
	p1_E = np.sqrt(0.5)*(p1_plus + p1_minus)
	p1_x = kins.z * kins.delta * np.cos(kins.phi_Dp - kins.phi_kp + phi_k) +  kins.pT * np.cos(phi_k - kins.phi_kp)
	p1_y = kins.z * kins.delta * np.sin(kins.phi_Dp - kins.phi_kp + phi_k) +  kins.pT * np.sin(phi_k - kins.phi_kp)
	p1_z = np.sqrt(0.5)*(p1_plus - p1_minus)

	p2_perp = np.sqrt(((1-kins.z)**2)*kins.delta**2  + kins.pT**2 - 2*(1-kins.z)*kins.pT*kins.delta*np.cos(kins.phi_Dp))
	p2_minus = (1-kins.z)*q_minus
	p2_plus = (p2_perp**2)/(2*p2_minus)
	p2_E = np.sqrt(0.5)*(p2_plus + p2_minus)
	p2_x = (1-kins.z) * kins.delta * np.cos(kins.phi_Dp - kins.phi_kp + phi_k) -  kins.pT * np.cos(phi_k - kins.phi_kp)
	p2_y = (1-kins.z) * kins.delta * np.sin(kins.phi_Dp - kins.phi_kp + phi_k) -  kins.pT * np.sin(phi_k - kins.phi_kp)
	p2_z = np.sqrt(0.5)*(p2_plus - p2_minus)

	# proton kinematics
	P_E = np.sqrt(0.5)*P_plus
	P_x = 0
	P_y = 0 
	P_z = np.sqrt(0.5)*P_plus

	Pp_plus = P_plus
	Pp_perp = kins.delta
	Pp_minus = (kins.delta**2)/(2*Pp_plus)
	Pp_E = np.sqrt(0.5)*(Pp_plus + Pp_minus)
	Pp_x = - Pp_perp * np.cos(kins.phi_Dp - kins.phi_kp + phi_k)
	Pp_y = - Pp_perp * np.sin(kins.phi_Dp - kins.phi_kp + phi_k)
	Pp_z = np.sqrt(0.5)*(Pp_plus - Pp_minus)

	# electron kinematics
	k_plus = ((kins.Q**2)*(1-kins.y))/(2*q_minus*kins.y) 
	k_minus = q_minus / kins.y
	k_perp = np.sqrt(((kins.Q**2)*(1-kins.y))/(kins.y**2))
	k_E = np.sqrt(0.5)*(k_plus + k_minus)
	k_x = k_perp * np.cos(phi_k)
	k_y = k_perp * np.sin(phi_k)
	k_z = np.sqrt(0.5)*(k_plus - k_minus)

	kp_plus = ((kins.Q**2)*(1-kins.y))/(2*kins.y*q_minus*(1-kins.y))
	kp_minus = (q_minus*(1-kins.y)) / kins.y
	kp_E = np.sqrt(0.5)*(kp_plus + kp_minus)
	kp_x = k_x
	kp_y = k_y
	kp_z = np.sqrt(0.5)*(kp_plus - kp_minus)


	# print('inc. p M^2', P_E**2 - P_x**2 - P_y**2 - P_z**2)
	# print('out. p M^2', Pp_E**2 - Pp_x**2 - Pp_y**2 - Pp_z**2, 2*Pp_plus*Pp_minus - Pp_perp**2)
	# print('inc. e M^2', k_E**2 - k_x**2 - k_y**2 - k_z**2, 2*k_plus*k_minus - k_perp**2)
	# print('out. e M^2', kp_E**2 - kp_x**2 - kp_y**2 - kp_z**2, 2*kp_plus*kp_minus - k_perp**2)
	# print('out. j1 M^2', p1_E**2 - p1_x**2 - p1_y**2 - p1_z**2, 2*p1_plus*p1_minus - p1_perp**2)
	# print('out. j2 M^2', p2_E**2 - p2_x**2 - p2_y**2 - p2_z**2, 2*p2_plus*p2_minus - p2_perp**2)


	event = [
		{'status':1, 'id':2, 'E':p1_E, 'px':p1_x, 'py':p1_y, 'pz':p1_z},
		{'status':1, 'id':-2, 'E':p2_E, 'px':p2_x, 'py':p2_y, 'pz':p2_z},
		{'status':1, 'id':11, 'E':k_E, 'px':k_x, 'py':k_y, 'pz':k_z},
		{'status':1, 'id':2212, 'E':Pp_E, 'px':Pp_x, 'py':Pp_y, 'pz':Pp_z},
		{'status':-1, 'id':11, 'E':kp_E, 'px':kp_x, 'py':kp_y, 'pz':kp_z},
		{'status':-1, 'id':2212, 'E':P_E, 'px':P_x, 'py':P_y, 'pz':P_z}
	]

	return event



if __name__ == '__main__':	

	# produce a sample of events to save to a numpy file
	y_range = [0.05, 0.095]
	t_range = [0.05, 0.1]
	z_range = [0.2, 0.8]
	Q2_range = [1, 100]
	pT_range = [0, 10]
	roots = 90
	Qbar0 = 0.5

	nevents = 10
	rng = np.random.default_rng()

	events = []
	event_id = 0
	while(len(events) < nevents):

		kins = dijet.Kinematics(s=roots**2)
		kins.y = rng.uniform(*y_range)
		kins.delta = np.sqrt(rng.uniform(*t_range))
		kins.z = rng.uniform(*z_range)
		kins.Q = np.sqrt(rng.uniform(*Q2_range))
		kins.pT = rng.uniform(*pT_range)
		kins.x = (kins.Q**2)/(kins.s*kins.y)
		kins.phi_Dp = rng.uniform(0, 2*np.pi)
		kins.phi_kp = rng.uniform(0, 2*np.pi)

		if kins.x > 0.01: continue
		if np.sqrt((kins.Q**2)*kins.z*(1-kins.z)) < Qbar0: continue

		event_id += 1
		particles = get_particles(kins)
		event = {
		    "event_id": event_id,
		    "particles": particles
		}
		events.append(event)

	# fname = 'data/dijet_mc_test.json'
	# with open(fname, 'w') as f:
	# 	json.dump(events, f, indent=2)

	# print('saved mc file w/', nevents, 'events in', fname)



	




