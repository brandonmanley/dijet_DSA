import numpy as np
import matplotlib.pyplot as plt

# setup plotting
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = 25  
plt.rcParams["xtick.labelsize"] = 20  
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 18  
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams['axes.xmargin'] = 0


def get_data(files):

	data = [np.load(f, allow_pickle=True).item() for f in files]

	targets = ['p', 'd', 'h']
	corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	# compute nuclear target replicas
	for i in range(len(data)):
		for tar in targets:
			if tar == 'd':
				p,n = 1,1
				omega_d = 0.07
				Pp = 1-1.5*omega_d
				Pn = Pp
		
			elif tar == 'h':
				p,n = 2,1
				pS,pD,pSp = 0.9,0.1,0.02
				Pp = -(4./3.)*(pD-pSp)
				Pn = pS - (1./3.)*(pD-pSp)
		
			elif tar == 'p': continue
		
			data[i][tar] = []
			for irep in range(len(data[i]['p'])):
				tar_rep = {}
				for corr in corrs:
					p_rep = data[i]['p'][irep][corr]
					n_rep = data[i]['n'][irep][corr]

					tar_rep[corr] = (Pp*p_rep + Pn*n_rep)/(p+n)
					
				data[i][tar].append(tar_rep)


	# compute confidence intervals for target replicas
	confid = 68
	bands = [{} for i in range(len(data))]
	for i in range(len(data)):
		for tar in targets:
			bands[i][tar] = {}
			for obj in ['lower', 'upper', 'mean']:
				bands[i][tar][obj] = {}
		
			for corr in corrs:
				corr_reps = [rep[corr] for rep in data[i][tar]]
				bands[i][tar]['lower'][corr] = np.percentile(corr_reps, 0.5*(100-confid), axis=0)
				bands[i][tar]['upper'][corr] = np.percentile(corr_reps, 100 - 0.5*(100-confid), axis=0)
				bands[i][tar]['mean'][corr] = np.mean(corr_reps, axis=0)


	lumi = 100
	errors = [{} for i in range(len(data))]
	for i in range(len(data)):
		for tar in targets:
			errors[i][tar] = {}

			if tar == 'd':
				p,n = 1,1
				omega_d = 0.07
				Pp = 1-1.5*omega_d
				Pn = Pp

			elif tar == 'h':
				p,n = 2,1
				pS,pD,pSp = 0.9,0.1,0.02
				Pp = -(4./3.)*(pD-pSp)
				Pn = pS - (1./3.)*(pD-pSp)

			if tar in ['d', 'h']:
				nuc_fac = ((Pp**2) + (Pn**2))/((p+n)**2)
			else:
				nuc_fac = 1

			errors[i][tar]['all'] = np.sqrt(nuc_fac/(2*lumi*np.array(data[i]['p'][0]['denom'])))
			errors[i][tar]['corr'] = np.sqrt((2*nuc_fac)/(2*lumi*np.array(data[i]['p'][0]['denom'])))

	return {'data': data, 'bands': bands, 'errors': errors}




def plot_z_variation(data, **options):

	asp_ratio = 3.5/3
	psize = 4
	nrows, ncols = 4, len(data['data'])
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row', sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#6257ff', '#FF6961', '#51c46f']
	corrs = ['<1>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	zs = [dat['space']['z'] for dat in data['data']]
	pT_values = data['data'][0]['pT values']

	for iz, z in enumerate(zs):
		for icorr, corr in enumerate(corrs):

			if plot_bands:
				if plot_mean: ax[icorr, iz].plot(pT_values, data['bands'][iz][target]['mean'][corr], color='black')
				ax[icorr, iz].fill_between(pT_values, data['bands'][iz][target]['lower'][corr], data['bands'][iz][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.5)
			
			else:
				for irep, rep in enumerate(data['data'][iz][target]):
					ax[icorr, iz].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])

			if plot_stat_errors:
				if '1' in corr: err = 'all'
				else: err = 'corr'
				ax[icorr, iz].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][iz][target][err][1:-1].shape), yerr=np.sqrt(10)*data['errors'][iz][target][err][1:-1], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr, iz].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][iz][target][err][1:-1].shape), yerr=data['errors'][iz][target][err][1:-1], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	# tar_lims = {'p': [[-0.04, 0.04], [-0.01, 0.01], 
	for iz, z in enumerate(zs): 

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[0, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[0, 0].set_ylabel(r'$ d^3 \langle 1 \rangle / d p_{\perp} dt dz$', size=22)
		ax[0, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[0, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		   
		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[1, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[1, 0].set_ylabel(r'$ d^3 \langle \cos \phi_{{\Delta p}} \rangle / d p_{\perp} dt dz $', size=22)
		ax[1, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[1, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		
		# ax[itar, 1].axhline(y=0, color='lightgray', linestyle='--')
		# ax[2, -1].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[2, 0].set_ylabel(r'$ d^3  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle / d p_{\perp} dt dz $', size=22)
		ax[2,iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[1, iz].legend(frameon=False)
		# ax[2, iz].set_ylim([-0.01, 0.01])

		# ax[2,iz].axhline(y=0, color='lightgray', linestyle='--')
		ax[3, iz].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[3, 0].set_ylabel(r'$ d^3 \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle / d p_{\perp} dt dz $', size=22)
		ax[3, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[3,iz].legend(frameon=False)
		# ax[3,iz].set_ylim([-0.005, 0.005])


		for i in range(nrows):
			ax[i, iz].text(
				0.1, 0.88, fr'$z={z}$',
				transform=ax[i, iz].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				# color=colors[iz],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)
			ax[i, iz].text(
				0.1, 0.1, tar_labels[tar_col[target]],
				transform=ax[i, iz].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				color=colors[tar_col[target]],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)

	# ax[0, 2].text(
	# 	0.68, 0.05, info_text, 
	# 	transform=ax[0, 2].transAxes,
	# 	ha='left', va='bottom', 
	# 	fontsize=16, wrap=True, 
	# 	bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	# )

	plt.tight_layout()
	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.subplots_adjust(wspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_zvar_band.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_zvar_lines.pdf', dpi=400, bbox_inches="tight")





def plot_target_variation(data, **options):

	asp_ratio = 3.5/3
	psize = 4
	nrows, ncols = 5, 3
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row', sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	plot_mean = options.get('plot_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	targets = ['p', 'd', 'h']
	colors = ['#6257ff', '#FF6961', '#51c46f']
	corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	pT_values = data['data'][0]['pT values']

	for it, tar in enumerate(targets):
		for icorr, corr in enumerate(corrs):

			if plot_bands:
				if plot_mean: ax[icorr, it].plot(pT_values, data['bands'][0][tar]['mean'][corr], color='black')
				ax[icorr, it].fill_between(pT_values, data['bands'][0][tar]['lower'][corr], data['bands'][0][tar]['upper'][corr], color=colors[tar_col[tar]], alpha=0.5)
			
			else:
				for irep, rep in enumerate(data['data'][0][tar]):
					ax[icorr, it].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[tar]])

			if plot_stat_errors:
				if '1' in corr: err = 'all'
				else: err = 'corr'
				ax[icorr, it].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][0][tar][err][1:-1].shape), yerr=np.sqrt(10)*data['errors'][0][tar][err][1:-1], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr, it].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][0][tar][err][1:-1].shape), yerr=data['errors'][0][tar][err][1:-1], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	# tar_lims = {'p': [[-0.04, 0.04], [-0.01, 0.01], 
	for it, tar in enumerate(targets):

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[0, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[0, 0].set_ylabel(r'$ d^2 \langle 1 \rangle / d p_{\perp} dt$', size=22)
		ax[0, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[0, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()

		ax[1, 0].set_ylabel(r'$ d^2 \langle \cos \phi_{{k p}} \rangle / d p_{\perp} dt $', size=22)
		ax[1, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		   
		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[1, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[2, 0].set_ylabel(r'$ d^2 \langle \cos \phi_{{\Delta p}} \rangle / d p_{\perp} dt $', size=22)
		ax[2, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[1, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		
		# ax[itar, 1].axhline(y=0, color='lightgray', linestyle='--')
		# ax[2, -1].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[3, 0].set_ylabel(r'$ d^2  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle / d p_{\perp} dt $', size=22)
		ax[3, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[1, iz].legend(frameon=False)
		# ax[2, iz].set_ylim([-0.01, 0.01])

		# ax[2,iz].axhline(y=0, color='lightgray', linestyle='--')
		ax[4, it].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[4, 0].set_ylabel(r'$ d^2 \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle / d p_{\perp} dt $', size=22)
		ax[4, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[3,iz].legend(frameon=False)
		# ax[3,iz].set_ylim([-0.005, 0.005])


		for i in range(nrows):
			ax[i, it].text(
				0.1, 0.1, tar_labels[tar_col[tar]],
				transform=ax[i, it].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				color=colors[tar_col[tar]],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)

	# ax[0, 2].text(
	# 	0.68, 0.05, info_text, 
	# 	transform=ax[0, 2].transAxes,
	# 	ha='left', va='bottom', 
	# 	fontsize=16, wrap=True, 
	# 	bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	# )

	plt.tight_layout()
	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.subplots_adjust(wspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_targetvar_band.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_targetvar_lines.pdf', dpi=400, bbox_inches="tight")







def plot_t_variation(data, **options):

	asp_ratio = 3.5/3
	psize = 4
	nrows, ncols = 4, len(data['data'])
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row', sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#6257ff', '#FF6961', '#51c46f']
	corrs = ['<1>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	zs = [dat['space']['z'] for dat in data['data']]
	pT_values = data['data'][0]['pT values']

	for iz, z in enumerate(zs):
		for icorr, corr in enumerate(corrs):

			if plot_bands:
				if plot_mean: ax[icorr, iz].plot(pT_values, data['bands'][iz][target]['mean'][corr], color='black')
				ax[icorr, iz].fill_between(pT_values, data['bands'][iz][target]['lower'][corr], data['bands'][iz][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.5)
			
			else:
				for irep, rep in enumerate(data['data'][iz][target]):
					ax[icorr, iz].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])

			if plot_stat_errors:
				if '1' in corr: err = 'all'
				else: err = 'corr'
				ax[icorr, iz].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][iz][target][err][1:-1].shape), yerr=np.sqrt(10)*data['errors'][iz][target][err][1:-1], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr, iz].errorbar(
						pT_values[1:-1], np.zeros(data['errors'][iz][target][err][1:-1].shape), yerr=data['errors'][iz][target][err][1:-1], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	# tar_lims = {'p': [[-0.04, 0.04], [-0.01, 0.01], 
	for iz, z in enumerate(zs): 

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[0, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[0, 0].set_ylabel(r'$ d \langle 1 \rangle / d p_{\perp}$', size=22)
		ax[0, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[0, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		   
		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[1, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[1, 0].set_ylabel(r'$ d \langle \cos \phi_{{\Delta p}} \rangle / d p_{\perp}  $', size=22)
		ax[1, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[1, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		
		# ax[itar, 1].axhline(y=0, color='lightgray', linestyle='--')
		# ax[2, -1].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[2, 0].set_ylabel(r'$ d  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle / d p_{\perp}  $', size=22)
		ax[2,iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[1, iz].legend(frameon=False)
		# ax[2, iz].set_ylim([-0.01, 0.01])

		# ax[2,iz].axhline(y=0, color='lightgray', linestyle='--')
		ax[3, iz].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[3, 0].set_ylabel(r'$ d \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle / d p_{\perp}  $', size=22)
		ax[3, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[3,iz].legend(frameon=False)
		# ax[3,iz].set_ylim([-0.005, 0.005])

		if iz == 0: tlabel = r'$t=0.1\,\, \mathrm{GeV}^2$'
		else: tlabel = r'$t \in [0.05, 0.2]\,\, \mathrm{GeV}^2$'
		for i in range(nrows):
			ax[i, iz].text(
				0.1, 0.88, tlabel,
				transform=ax[i, iz].transAxes,
				ha='left', va='bottom', 
				fontsize=20, wrap=True, 
				# color=colors[iz],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)
			ax[i, iz].text(
				0.1, 0.1, tar_labels[tar_col[target]],
				transform=ax[i, iz].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				color=colors[tar_col[target]],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)

	# ax[0, 2].text(
	# 	0.68, 0.05, info_text, 
	# 	transform=ax[0, 2].transAxes,
	# 	ha='left', va='bottom', 
	# 	fontsize=16, wrap=True, 
	# 	bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	# )

	plt.tight_layout()
	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.subplots_adjust(wspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_tvar_band.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_tvar_lines.pdf', dpi=400, bbox_inches="tight")





def plot_harmonics(data, **options):

	asp_ratio = 6/3
	psize = 3
	nrows, ncols = 5, 1
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#6257ff', '#FF6961', '#51c46f']
	corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	pT_values = data['data'][0]['pT values']

	for icorr, corr in enumerate(corrs):

		if plot_bands:
			if plot_mean: ax[icorr].plot(pT_values, data['bands'][0][target]['mean'][corr], color='black')
			ax[icorr].fill_between(pT_values, data['bands'][0][target]['lower'][corr], data['bands'][0][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.5)
		
		else:
			for irep, rep in enumerate(data['data'][0][target]):
				ax[icorr].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])

		if plot_stat_errors:
			if '1' in corr: err = 'all'
			else: err = 'corr'
			ax[icorr].errorbar(
					pT_values[1:-1], np.zeros(data['errors'][0][target][err][1:-1].shape), yerr=np.sqrt(10)*data['errors'][0][target][err][1:-1], fmt='o',
					capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
					label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
			)
			ax[icorr].errorbar(
					pT_values[1:-1], np.zeros(data['errors'][0][target][err][1:-1].shape), yerr=data['errors'][0][target][err][1:-1], fmt='o',
					capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
					label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
			)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	for icorr, corr in enumerate(corrs): 
		ax[icorr].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[icorr].axhline(y=0, color='lightgray', linestyle='--')

		ax[icorr].text(
				0.1, 0.1, tar_labels[tar_col[target]],
				transform=ax[icorr].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				color=colors[tar_col[target]],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
		)


	ax[0].set_ylabel(r'$ d \langle 1 \rangle / d p_{\perp}$', size=20)
	# ax[0].set_ylim([-0.04, 0.04])

	ax[1].set_ylabel(r'$ d \langle \cos \phi_{{k p}} \rangle / d p_{\perp}  $', size=20)
	# ax[1].set_ylim([-0.04, 0.04])
		   
	ax[2].set_ylabel(r'$ d \langle \cos \phi_{{\Delta p}} \rangle / d p_{\perp}  $', size=20)
	# ax[2].set_ylim([-0.04, 0.04])
		
	ax[3].set_ylabel(r'$ d  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle / d p_{\perp}  $', size=20)
	# ax[3].set_ylim([-0.01, 0.01])

	ax[4].set_xlabel(r'$p_{\perp}$ [GeV]')
	ax[4].set_ylabel(r'$ d \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle / d p_{\perp}  $', size=20)
	# ax[4].set_ylim([-0.005, 0.005])

	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_band.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_lines.pdf', dpi=400, bbox_inches="tight")








