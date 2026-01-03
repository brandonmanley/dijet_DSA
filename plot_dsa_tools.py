import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

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

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#FF6961','#6257ff', '#51c46f']
	# corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']
	corrs = ['<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	asp_ratio = 4.5/3
	psize = 3
	nrows, ncols = len(corrs), len(data['data'])
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row', sharex='col')


	zs = [dat['space']['z'] for dat in data['data']]
	pT_values = data['data'][0]['pT values']

	for iz, z in enumerate(zs):
		for icorr, corr in enumerate(corrs):

			if plot_bands:
				if plot_mean: ax[icorr, iz].plot(pT_values, data['bands'][iz][target]['mean'][corr], color='black')
				ax[icorr, iz].fill_between(pT_values, data['bands'][iz][target]['lower'][corr], data['bands'][iz][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.6)
			
			else:
				for irep, rep in enumerate(data['data'][iz][target]):
					ax[icorr, iz].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])


			if plot_stat_errors:
				if '1' in corr: err = 'all'
				else: err = 'corr'
				if errors_at_mean: error_points = data['bands'][iz][target]['mean'][corr]
				else: error_points = np.zeros(data['errors'][iz][target][err].shape)

				ax[icorr, iz].errorbar(
						pT_values[1:-1:2], error_points[1:-1:2], yerr=np.sqrt(10)*data['errors'][iz][target][err][1:-1:2], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (1 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr, iz].errorbar(
						pT_values[1:-1:2], error_points[1:-1:2], yerr=data['errors'][iz][target][err][1:-1:2], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (100 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	# tar_lims = {'p': [[-0.04, 0.04], [-0.01, 0.01], 
	for iz, z in enumerate(zs): 

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[0, 0].set_ylabel(r'$ \frac{ d^3 }{ d p_{\perp} dt dz } \langle 1 \rangle $', size=20)
		# ax[0, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[1, 0].set_ylabel(r'$ \frac{ d^3 }{ d p_{\perp} dt dz } \langle \cos \phi_{{k p}} \rangle  $', size=20)
		# ax[1, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
		# ax[1, iz].set_ylim([-0.04, 0.04])
		   
		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		ax[0, 0].set_ylabel(r'$ \frac{ d^3 }{ d p_{\perp} dt dz }  \langle \cos \phi_{{\Delta p}} \rangle  $', size=20)
		ax[0, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

		ax[1, 0].set_ylabel(r'$ \frac{ d^3 }{ d p_{\perp} dt dz }  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle  $', size=20)
		ax[1, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

		ax[2, iz].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[2, 0].set_ylabel(r'$\frac{ d^3 }{ d p_{\perp} dt dz } \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle$', size=20)
		ax[2, iz].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

		# ax[0, 0].set_ylabel(r'$ \frac{ d^2 }{ d p_{\perp} dt } \mathrm{or} \frac{ d^2 }{ d p_{\perp} dt}\,\, \langle \cos \phi_{{\Delta p}} \rangle  $', size=20)
		# ax[1, 0].set_ylabel(r'$ \frac{ d^2 }{ d p_{\perp} dt }  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle  $', size=20)
		# ax[2, 0].set_ylabel(r'$\frac{ d^2 }{ d p_{\perp} dt } \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle$', size=20)

		# ax[0, iz].set_ylim([-0.0025, 0.0025])
		# ax[1, iz].set_ylim([-0.0025, 0.0025])
		# ax[2, iz].set_ylim([-0.0025, 0.0025])

		for i in range(nrows):
			ax[i, iz].text(
				0.05, 0.8, fr'$z={z}$',
				transform=ax[i, iz].transAxes,
				ha='left', va='bottom', 
				fontsize=30, wrap=True, 
				# color=colors[iz],
				# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			)
			# ax[i, iz].text(
			# 	0.1, 0.1, tar_labels[tar_col[target]],
			# 	transform=ax[i, iz].transAxes,
			# 	ha='left', va='bottom', 
			# 	fontsize=30, wrap=True, 
			# 	color=colors[tar_col[target]],
			# 	# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			# )

	ax[0, -1].legend(frameon=False, fontsize=15)

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

	asp_ratio = 4.5/3
	psize = 3
	nrows, ncols = 5, 3
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row', sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	targets = ['p', 'd', 'h']
	colors = ['#FF6961','#6257ff', '#51c46f']
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
				if errors_at_mean: error_points = data['bands'][0][tar]['mean'][corr]
				else: error_points = np.zeros(data['errors'][0][tar][err].shape)

				ax[icorr, it].errorbar(
						pT_values[1:-1], error_points[1:-1], yerr=np.sqrt(10)*data['errors'][0][tar][err][1:-1], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr, it].errorbar(
						pT_values[1:-1], error_points[1:-1], yerr=data['errors'][0][tar][err][1:-1], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	# tar_lims = {'p': [[-0.04, 0.04], [-0.01, 0.01], 
	for it, tar in enumerate(targets):

		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[0, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[0, 0].set_ylabel(r'$ \frac{ d^2 }{d p_{\perp} dt} \langle 1 \rangle $', size=22)
		ax[0, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[0, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()

		ax[1, 0].set_ylabel(r'$ \frac{ d^2 }{d p_{\perp} dt} \langle \cos \phi_{{k p}} \rangle  $', size=22)
		ax[1, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		   
		# ax[itar, 0].axhline(y=0, color='lightgray', linestyle='--')
		# ax[1, nrows-1].set_xlabel(r'$p_\perp$ [GeV]')
		ax[2, 0].set_ylabel(r'$ \frac{ d^2 }{d p_{\perp} dt} \langle \cos \phi_{{\Delta p}} \rangle  $', size=22)
		ax[2, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[1, iz].set_ylim([-0.04, 0.04])
		# ax[0,iz].semilogy()
		
		# ax[itar, 1].axhline(y=0, color='lightgray', linestyle='--')
		# ax[2, -1].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[3, 0].set_ylabel(r'$ \frac{ d^2 }{d p_{\perp} dt}  \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle $', size=22)
		ax[3, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[1, iz].legend(frameon=False)
		# ax[2, iz].set_ylim([-0.01, 0.01])

		# ax[2,iz].axhline(y=0, color='lightgray', linestyle='--')
		ax[4, it].set_xlabel(r'$p_{\perp}$ [GeV]')
		ax[4, 0].set_ylabel(r'$ \frac{ d^2 }{d p_{\perp} dt} \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle  $', size=22)
		ax[4, it].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=True, top=False)
		# ax[3,iz].legend(frameon=False)
		# ax[3,iz].set_ylim([-0.005, 0.005])


		for i in range(nrows):
			ax[i, it].text(
				0.1, 0.1, tar_labels[tar_col[tar]],
				transform=ax[i, it].transAxes,
				ha='left', va='bottom', 
				fontsize=20, wrap=True, 
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
	nrows, ncols = 1, len(data['data'])
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharey='row')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)
	harmonic = options.get('harmonic', '<cos(phi_Dp)>')

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#51c46f', '#FF6961','#6257ff']



	for iset in range(len(data['data'])):
		pT_values = data['data'][iset]['pT values']

		if plot_bands:
			if plot_mean: ax[iset].plot(pT_values, data['bands'][iset][target]['mean'][harmonic], color='black')
			ax[iset].fill_between(pT_values, data['bands'][iset][target]['lower'][harmonic], data['bands'][iset][target]['upper'][harmonic], color=colors[tar_col[target]], alpha=0.6)
			
		else:
			for irep, rep in enumerate(data['data'][iset][target]):
				ax[iset].plot(pT_values, np.array(rep[harmonic]), alpha=0.3, color=colors[tar_col[target]])

		if plot_stat_errors:
			if '1' in harmonic: err = 'all'
			else: err = 'corr'
			if errors_at_mean: error_points = data['bands'][iset][target]['mean'][harmonic]
			else: error_points = np.zeros(data['errors'][iset][target][err].shape)

			if iset == 0: step = 1
			else: step=4
			ax[iset].errorbar(
						pT_values[1:-1:step], error_points[1:-1:step], yerr=np.sqrt(10)*data['errors'][iset][target][err][1:-1:step], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
			)
			ax[iset].errorbar(
						pT_values[1:-1:step], error_points[1:-1:step], yerr=data['errors'][iset][target][err][1:-1:step], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
			)



	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	for iset in range(len(data['data'])):
		ax[iset].set_xlabel(r'$p_\perp$ [GeV]')
		ax[iset].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

	# tlabel = r'$t=0.1\,\, \mathrm{GeV}^2$'
	tlabel = r'$z=0.2$'
	ax[1].text(
		0.1, 0.8, tlabel,
		transform=ax[1].transAxes,
		ha='left', va='bottom', 
		fontsize=18, wrap=True, 
		# color=colors[iz],
		# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	)

	# tlabel = r'$t \in [0.05, 0.1]\,\, \mathrm{GeV}^2$'
	tlabel = r'$z \in [0.2, 0.5]$'
	ax[0].text(
		0.1, 0.8, tlabel,
		transform=ax[0].transAxes,
		ha='left', va='bottom', 
		fontsize=18, wrap=True, 
		# color=colors[iz],
		# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	)

	for i in range(2):
		ax[i].text(
			0.1, 0.1, r'$\langle \cos \phi_{\Delta p} \rangle$', 
			transform=ax[i].transAxes,
			ha='left', va='bottom', 
			fontsize=18, wrap=True, 
		)

	ax[0].set_ylim([-0.03, 0.03])

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.05)
	plt.show()

	fname = options.get('fname', None)
	if save_plots:
		if fname == None: raise ValueError('Error: please specify file name')
		fig.savefig(f'plots/{fname}', dpi=400, bbox_inches="tight")





def plot_harmonics(data, **options):

	asp_ratio = 8/3
	psize = 2.75
	nrows, ncols = 3, 2

	fig = plt.figure(figsize=(asp_ratio*psize*ncols, psize*nrows))
	gs = GridSpec(2, 6, figure=fig)
	ax1 = fig.add_subplot(gs[0, 1:3])
	ax2 = fig.add_subplot(gs[0, 3:5])
	ax3 = fig.add_subplot(gs[1, 0:2])
	ax4 = fig.add_subplot(gs[1, 2:4])
	ax5 = fig.add_subplot(gs[1, 4:6])
	ax = [ax1, ax2, ax3, ax4, ax5]

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)
	smooth_plot = options.get('smooth_plot', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#6257ff',  '#FF6961', '#51c46f']
	color = options.get('colors', colors)
	labels = options.get('labels', None)

	corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']
	pT_values = data['data'][0]['pT values']

	for idata in range(len(data['bands'])):
		if idata > 0 and plot_stat_errors: continue
		if len(data['bands']) > 1: 
			band_color = colors[idata]
			label = labels[idata]
		else: 
			band_color = colors[tar_col[target]]
			label = None
		

		for icorr, corr in enumerate(corrs):

			if plot_bands:
				if plot_mean: ax[icorr].plot(pT_values, data['bands'][idata][target]['mean'][corr], color=band_color)

				if smooth_plot: 
					y_lower = data['bands'][idata][target]['lower'][corr] 
					y_upper = data['bands'][idata][target]['upper'][corr]
					f_lower = interp1d(pT_values, y_lower, kind='cubic')
					f_upper = interp1d(pT_values, y_upper, kind='cubic')
					fine_pT_values = np.linspace(pT_values[0], pT_values[-1], 200)
					lower_band_values = f_lower(fine_pT_values)
					upper_band_values = f_upper(fine_pT_values)
					x_values = fine_pT_values
				else:
					lower_band_values = data['bands'][idata][target]['lower'][corr]
					upper_band_values = data['bands'][idata][target]['upper'][corr]
					x_values = pT_values

				ax[icorr].fill_between(x_values, lower_band_values, upper_band_values, color=band_color, alpha=1.0, label=label)
			
			else:
				# split colors by percentile
				border = 33
				low_pTs = np.array([rep[corr][2] for rep in data['data'][idata][target]])
				# print(pT_values[2])
				p25 = np.percentile(low_pTs, border)
				p75 = np.percentile(low_pTs, 100 - border)

				lower_idx  = np.where(low_pTs <= p25)[0]
				middle_idx = np.where((low_pTs > p25) & (low_pTs < p75))[0]
				upper_idx  = np.where(low_pTs >= p75)[0]

				for irep, rep in enumerate(data['data'][idata][target]):
					# if irep != 1: continue
					if irep in lower_idx: color = 'blue'
					elif irep in middle_idx: color = 'red'
					elif irep in upper_idx: color = 'green'
					else: color = band_color
					color = band_color
					ax[icorr].plot(pT_values, np.array(rep[corr]), alpha=0.2, color=color)

			if plot_stat_errors:
				if '1' in corr: err = 'all'
				else: err = 'corr'

				if errors_at_mean: error_points = data['bands'][0][target]['mean'][corr]
				else: error_points = np.zeros(data['errors'][0][target][err].shape)

				ax[icorr].errorbar(
						pT_values[1:-1:2], error_points[1:-1:2], yerr=np.sqrt(10)*data['errors'][0][target][err][1:-1:2], fmt='o',
						capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, linewidth=0,
						label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
				)
				ax[icorr].errorbar(
						pT_values[1:-1:2], error_points[1:-1:2], yerr=data['errors'][0][target][err][1:-1:2], fmt='o',
						capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, linewidth=0, 
						label=rf'Stat. error (100 $\mathrm{{fb}}^{{-1}}$)'
				)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	for icorr, corr in enumerate(corrs): 
		for icol in range(ncols):
			ax[icorr].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
			# ax[icorr].axhline(y=0, color='lightgray', linestyle='--')

			# ax[irow, icol].text(
			# 		0.1, 0.1, tar_labels[tar_col[target]],
			# 		transform=ax[irow,icol].transAxes,
			# 		ha='left', va='bottom', 
			# 		fontsize=30, wrap=True, 
			# 		color=colors[tar_col[target]],
			# 		# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			# )

	ax[0].legend(frameon=False, fontsize=15, loc='upper left')

	fsize=20
	ax[0].text(
		0.1, 0.6, r'$\langle 1 \rangle $',
		transform=ax[0].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[1].text(
		0.1, 0.6, r'$\langle \cos \phi_{{k p}} \rangle $',
		transform=ax[1].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[2].text(
		0.1, 0.8, r'$\langle \cos \phi_{{\Delta p}} \rangle $',
		transform=ax[2].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[3].text(
		0.1, 0.8, r'$\langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle $',
		transform=ax[3].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[4].text(
		0.1, 0.8, r'$\langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}} \rangle $',
		transform=ax[4].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	info = r'''
	$\sqrt{s} = 40\, \mathrm{GeV}$
	$Q^2 \in [1, 100] \, \mathrm{GeV}^2$
	$y \in [0.05, 0.95]$
	$z \in [0.2, 0.5]$
	$t \in [0.05, 0.1]\, \mathrm{GeV}^2$
	'''
	# ax[0].text(
	# 		0.05, -0.02, info,
	# 		transform=ax[0].transAxes,
	# 		ha='left', va='bottom', 
	# 		fontsize=12, wrap=True, 
	# 		color='black',
	# 		# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	# )

	# ax[0,0].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle 1 \rangle $', size=20)
	# ax[1,0].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{k p}} \rangle  $', size=20)
	# ax[0,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{\Delta p}} \rangle  $', size=20)
	# ax[1,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle   $', size=20)
	# ax[2,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle   $', size=20)
	for icorr in range(5): ax[icorr].set_xlabel(r'$p_{\perp}$ [GeV]')

	ax[0].set_ylim([-0.25, 1])
	ax[1].set_ylim([-0.2, 0.045])
	ax[2].set_ylim([-0.045, 0.045])
	ax[3].set_ylim([-0.025, 0.025])
	ax[4].set_ylim([-0.025, 0.025])

	plt.tight_layout()
	# plt.subplots_adjust(hspace=0)
	plt.show()

	fname = options.get('fname', None)
	if save_plots:
		if fname == None: raise ValueError('Error: please specify file name')
		fig.savefig(f'plots/{fname}', dpi=400, bbox_inches="tight")





def plot_oam_harmonics(data, **options):

	asp_ratio = 6/3
	psize = 3
	nrows, ncols = 3, 1
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#FF6961','#6257ff', '#51c46f']
	corrs = ['<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	pT_values = data['data'][0]['pT values']

	for icorr, corr in enumerate(corrs):

		if plot_bands:
			if plot_mean: ax[icorr].plot(pT_values, data['bands'][0][target]['mean'][corr], color=colors[tar_col[target]])
			ax[icorr].fill_between(pT_values, data['bands'][0][target]['lower'][corr], data['bands'][0][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.6)
		
		else:
			for irep, rep in enumerate(data['data'][0][target]):
				ax[icorr].plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])

		if plot_stat_errors:
			if '1' in corr: err = 'all'
			else: err = 'corr'

			if errors_at_mean: error_points = data['bands'][0][target]['mean'][corr]
			else: error_points = np.zeros(data['errors'][0][target][err].shape)

			ax[icorr].errorbar(
					pT_values[1:-1:2], error_points[1:-1:2], yerr=np.sqrt(10)*data['errors'][0][target][err][1:-1:2], fmt='o',
					capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, linewidth=0,
					label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
			)
			ax[icorr].errorbar(
					pT_values[1:-1:2], error_points[1:-1:2], yerr=data['errors'][0][target][err][1:-1:2], fmt='o',
					capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, linewidth=0, 
					label=rf'Stat. error (100 $\mathrm{{fb}}^{{-1}}$)'
			)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	for icorr, corr in enumerate(corrs): 

		ax[icorr].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
			# ax[icorr].axhline(y=0, color='lightgray', linestyle='--')

			# ax[irow, icol].text(
			# 		0.1, 0.1, tar_labels[tar_col[target]],
			# 		transform=ax[irow,icol].transAxes,
			# 		ha='left', va='bottom', 
			# 		fontsize=30, wrap=True, 
			# 		color=colors[tar_col[target]],
			# 		# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
			# )

	ax[0].legend(frameon=False, fontsize=15)

	ax[0].set_ylabel(r'$ \frac{d^2 }{d p_{\perp} dt} \langle \cos \phi_{{\Delta p}} \rangle  $', size=20)
	ax[1].set_ylabel(r'$ \frac{d^2 }{d p_{\perp} dt } \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle   $', size=20)
	ax[2].set_ylabel(r'$ \frac{d^2 }{d p_{\perp} dt} \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle   $', size=20)
	ax[2].set_xlabel(r'$p_{\perp}$ [GeV]')

	ax[0].set_ylim([-0.022, 0.022])
	ax[1].set_ylim([-0.008, 0.008])
	ax[2].set_ylim([-0.006, 0.006])

	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_oam_band.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_oam_lines.pdf', dpi=400, bbox_inches="tight")




def plot_one_harmonic(data, **options):

	asp_ratio = 4/3
	psize = 5
	nrows, ncols = 1, 1
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows), sharex='col')

	save_plots = options.get('save_plots', False)
	plot_bands = options.get('plot_bands', True)
	plot_stat_errors = options.get('plot_stat_errors', True)
	target = options.get('target', 'p')
	plot_mean = options.get('plot_mean', False)
	errors_at_mean = options.get('errors_at_mean', False)

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#51c46f', '#6257ff', '#FF6961']
	corr = '<cos(phi_Dp)>'

	pT_values = data['data'][0]['pT values']

	if plot_bands:
		if plot_mean: ax.plot(pT_values, data['bands'][0][target]['mean'][corr], color='#6257ff')
		ax.fill_between(pT_values, data['bands'][0][target]['lower'][corr], data['bands'][0][target]['upper'][corr], color=colors[tar_col[target]], alpha=0.5)
	
	else:
		for irep, rep in enumerate(data['data'][0][target]):
			if irep > 50: continue
			ax.plot(pT_values, np.array(rep[corr]), alpha=0.3, color=colors[tar_col[target]])

	if plot_stat_errors:
		if '1' in corr: err = 'all'
		else: err = 'corr'

		if errors_at_mean: error_points = data['bands'][0][target]['mean'][corr]
		else: error_points = np.zeros(data['errors'][0][target][err].shape)

		ax.errorbar(
				pT_values[1:-1:2], error_points[1:-1:2], yerr=np.sqrt(10)*data['errors'][0][target][err][1:-1:2], fmt='o',
				capsize=3, elinewidth=1, capthick=1, color='gray', markersize=0, 
				label=rf'Stat. error (10 $\mathrm{{fb}}^{{-1}}$)'
		)
		ax.errorbar(
				pT_values[1:-1:2], error_points[1:-1:2], yerr=data['errors'][0][target][err][1:-1:2], fmt='o',
				capsize=3, elinewidth=1.5, capthick=1.5, color='black', markersize=0, 
				label=rf'Stat. error (100 $\mathrm{{fb}}^{{-1}}$)'
		)


	tar_labels = [r'$p$', r'$d$', r'$^3\mathrm{He}$']

	ax.tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)
	# ax[icorr].axhline(y=0, color='lightgray', linestyle='--')

	info = r'''
	$\sqrt{s} = 40\, \mathrm{GeV}$
	$t=0.1\, \mathrm{GeV}^2$
	$Q^2 \in [1, 100] \, \mathrm{GeV}^2$
	$y \in [0.05, 0.95]$
	$z \in [0.2, 0.5]$
	'''
	ax.text(
			0.05, -0.02, info,
			transform=ax.transAxes,
			ha='left', va='bottom', 
			fontsize=15, wrap=True, 
			color='black',
			# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	)

	ax.text(
		0.1, 0.65, r'$ \langle \cos \phi_{{\Delta p}} \rangle $',
		transform=ax.transAxes,
		ha='left', va='bottom', 
		fontsize=25, wrap=True, 
		color='black'
	)
	ax.set_xlabel(r'$ p_\perp\,\, [\mathrm{GeV}] $', size=20)
	ax.legend(frameon=False, fontsize=15)

	ax.set_ylim([-0.04, 0.04])

	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.show()

	if save_plots:
		if plot_bands:
			fig.savefig('plots/dsa_band_cos.pdf', dpi=400, bbox_inches="tight")
		else:
			fig.savefig('plots/dsa_lines_cos.pdf', dpi=400, bbox_inches="tight")






def plot_p_dist(data, **options):

	save_plots = options.get('save_plots', False)
	target = options.get('target', 'p')
	ipT = options.get('ipT', 4)
	labels = options.get('labels', [])

	asp_ratio = 5.5/3
	psize = 3
	nrows, ncols = 3,2
	fig, ax = plt.subplots(nrows, ncols, figsize=(asp_ratio*psize*ncols, psize*nrows))

	tar_col = {'p':0, 'd':1, 'h':2}
	colors = ['#51c46f', '#FF6961','#6257ff']
	corrs = ['<1>', '<cos(phi_kp)>', '<cos(phi_Dp)>', '<cos(phi_Dp)cos(phi_kp)>', '<sin(phi_Dp)sin(phi_kp)>']

	bins = {}
	nbins = 15
	for icorr, corr in enumerate(corrs):
		if icorr == 0: bins[corr] = np.linspace(-0.1, 0.1, nbins)
		elif icorr == 1: bins[corr] = np.linspace(-0.01, 0.01, nbins)
		elif icorr == 2: bins[corr] = np.linspace(-0.01, 0.01, nbins)
		elif icorr == 3: bins[corr] = np.linspace(-0.0025, 0.0025, nbins)
		elif icorr == 4: bins[corr] = np.linspace(-0.005, 0.005, nbins)

	for i in range(len(data['data'])):
		for icorr, corr in enumerate(corrs):
			if icorr > 1: icol,irow = 1,icorr-2
			else: icol,irow = 0,icorr

			corr_data = [data['data'][i][target][irep][corr][ipT] for irep in range(len(data['data'][i][target]))]
			ax[irow,icol].hist(corr_data, label=labels[i], color=colors[i], bins=bins[corr], 
					  			alpha=0.5, density=True, 
								edgecolor='gray', 
                                linewidth=0.5)


	for icorr, corr in enumerate(corrs): 
		if icorr > 1: icol,irow = 1,icorr-2
		else: icol,irow = 0,icorr
		for icol in range(ncols):
			if irow == 2 and icol == 0: continue 
			ax[irow, icol].tick_params(axis="both", direction="in", length=5, width=1, which='both', right=False, top=False)

	ax[0,0].legend(frameon=False, fontsize=15, loc='upper right')

	fsize=20
	ax[0, 0].text(
		0.1, 0.8, r'$\langle 1 \rangle$',
		transform=ax[0,0].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[1, 0].text(
		0.1, 0.8, r'$\langle \cos \phi_{{k p}} \rangle $',
		transform=ax[1,0].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[0, 1].text(
		0.1, 0.8, r'$\langle \cos \phi_{{\Delta p}} \rangle $',
		transform=ax[0,1].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[1, 1].text(
		0.1, 0.8, r'$\langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle $',
		transform=ax[1,1].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	ax[2, 1].text(
		0.1, 0.8, r'$\langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}} \rangle $',
		transform=ax[2,1].transAxes,
		ha='left', va='bottom', 
		fontsize=fsize, wrap=True, 
		color='black'
	)

	pT_value = data["data"][0]["pT values"][ipT]

	info = rf'$p_{{\perp}} = {pT_value} \, \mathrm{{GeV}}$'
	ax[0,0].text(
			0.05, 0.7, info,
			transform=ax[0,0].transAxes,
			ha='left', va='bottom', 
			fontsize=15, wrap=True, 
			color='black',
			# bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black')
	)

	# ax[0,0].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle 1 \rangle $', size=20)
	# ax[1,0].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{k p}} \rangle  $', size=20)
	# ax[0,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{\Delta p}} \rangle  $', size=20)
	# ax[1,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \cos \phi_{{\Delta p}}  \cos \phi_{{k p}} \rangle   $', size=20)
	# ax[2,1].set_ylabel(r'$ \frac{d}{d p_{\perp}} \langle \sin \phi_{{\Delta p}} \sin \phi_{{k p}}  \rangle   $', size=20)
	# ax[1,0].set_xlabel(r'$p_{\perp}$ [GeV]')
	# ax[2,1].set_xlabel(r'$p_{\perp}$ [GeV]')

	ax[2,0].axis('off')
	ax[1,0].get_xaxis().set_visible(True)
	ax[1,0].tick_params(labelbottom=True)

	# ax[0,0].set_ylim([-0.25, 0.25])
	# ax[1,0].set_ylim([-0.045, 0.045])
	# ax[0,1].set_ylim([-0.045, 0.045])
	# ax[1,1].set_ylim([-0.025, 0.025])
	# ax[2,1].set_ylim([-0.025, 0.025])

	# plt.tight_layout()
	# plt.subplots_adjust(hspace=0)
	plt.show()

	if save_plots:
		fig.savefig('plots/dsa_band_hists.pdf', dpi=400, bbox_inches="tight")