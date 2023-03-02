import xarray as xr
import numpy as np
import glob
import os
import pdb
import sys
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'monospace'})

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import select_MWR_channels, Gband_double_side_band_average, compute_retrieval_statistics
from import_data import import_mirac_level1b_daterange, import_hatpro_level1b_daterange

import matplotlib.pyplot as plt


def post_process_pamtra(DS):

	"""
	Post process PAMTRA output by removing obsolete dimensions, selecting the 
	needed angle, average over polarizations, perform double side band averaging.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset that will be post processed.
	"""

	# remove obsolete dimensions and select the right frequencies:
	DS = DS.isel(grid_y=0, angles=-1, outlevel=0)
	DS['tb'] = DS.tb.mean(axis=-1)

	# double side band averaging:
	tb, freq_sb = Gband_double_side_band_average(DS.tb.values, DS.tb.frequency.values)
	DS = DS.sel(frequency=freq_sb)
	DS['tb'] = xr.DataArray(tb, dims=['grid_x', 'frequency'], coords=DS.tb.coords)

	return DS


def load_mwr_data(
	path_data,
	set_dict,
	radiometer):

	"""
	Load radiometer data (i.e., as they are stored on the cologne servers in daily subfolders)
	and generate an xarray dataset out of the loaded data (stored in a dictionary). 

	Parameters:
	-----------
	path_data : str
		String indicating the super directory of the MWR data (i.e., 
		/data/obs/campaigns/WALSEMA/atm/hatpro/l1/). Contains subdirectories
		indicating the current year, month and day (i.e., ./2022/07/21/).
	set_dict : dict
		Dictionary with various settings and auxiliary information.
	radiometer : str
		String indicating which radiometer data is addressed. Options:
		'hatpro', 'mirac-p'
	"""

	# load data:
	if radiometer == 'hatpro':
		data_dict = import_hatpro_level1b_daterange(path_data, set_dict['date_start'], set_dict['date_end'], 
													vers=set_dict['mwr_version'][radiometer])

	elif radiometer == 'mirac-p':
		data_dict = import_mirac_level1b_daterange(path_data, set_dict['date_start'], set_dict['date_end'], 
													vers=set_dict['mwr_version'][radiometer])


	# compose microwave radiometer data dictionary into xarray dataset:
	DS = xr.Dataset({'tb': (['time', 'frequency'], data_dict['tb'],
							{'units': "K"})},
					coords=	{'time': (['time'], data_dict['time'].astype("datetime64[s]"),
										{'units': "seconds since 1970-01-01 00:00:00 UTC"}),
							'frequency': (['frequency'], data_dict['freq_sb'],
										{'units': "GHz"})})

	return data_dict, DS


def merge_sonde_and_mwr(
	data_dict,
	DS,
	launch_time,
	set_dict,
	radiometer):

	"""
	Find overlapping radiometer and radiosonde launch times. Then compute the average and
	standard deviation of the observed brightness temperatures over that overlapping time span.

	Parameters:
	-----------
	data_dict : dict
		Data dictionary containing brightness temperature (tb), freuency (freq_sb) and time
		information.
	DS : xarray dataset
		As data_dict, but composed as xarray dataset according to the function load_mwr_data.
	launch_time : array of floats
		Radiosonde launch time array in seconds since 1970-01-01 00:00:00 UTC.
	set_dict : dict
		Dictionary with various settings and auxiliary information.
	radiometer : str
		String indicating which radiometer data is addressed. Options:
		'hatpro', 'mirac-p'
	"""

	# find temporal overlap:
	idx_overlap = np.array([np.where((data_dict['time'] >= lt) & 
							(data_dict['time'] <= lt + set_dict['window']))[0] for lt in launch_time])


	# compute mean and standard deviation of TB data for each radiosonde that overlaped with MWR data:
	tb_mean = np.zeros((len(launch_time), len(DS.frequency)))
	tb_std = np.zeros((len(launch_time), len(DS.frequency)))
	for k, idx_o in enumerate(idx_overlap):
		tb_mean[k,:] = np.nanmean(DS.tb.values[idx_o,:], axis=0)
		tb_std[k,:] = np.nanstd(DS.tb.values[idx_o,:], axis=0)


	# put into xarray dataset:
	DS_overlap = xr.Dataset({'tb_mean': 	(['time', 'frequency'], tb_mean,
									{'units': "K"}),
							'tb_std': 	(['time', 'frequency'], tb_std,
									{'units': "K"})},
							coords=	{'time': (['time'], launch_time.astype("datetime64[s]"),
												{'units': "seconds since 1970-01-01 00:00:00 UTC"}),
									'frequency': (['frequency'], DS.frequency.values,
												{'units': "GHz"})})

	return DS_overlap


"""
	Brief script for a comparison between simulated (PAMTRA) and observed
	(HATPRO or MiRAC-P) TBs. Comparison i.e., via scatter plot.
	- load simulated and obs TB data
	- modify simulated data to have the instrument characteristics of obs TBs
	- reduce obs TB data time grid to radiosonde launch times
	- visualize
"""


# paths:
path_data = {'mirac-p': "/data/obs/campaigns/WALSEMA/atm/mirac-p/l1/",
			'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l1/",
			'sim': "/net/blanc/awalbroe/Data/WALSEMA/radiosondes/fwd_sim_radiosondes/"}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/fwd_sim_radiosondes/"


# settings:
set_dict = {'date_start': "2022-06-28", 
			'date_end': "2022-08-12",
			'hatpro_obs': True,		# if True, HATPRO obs will be compared
			'mirac-p_obs': True,	# if True, MiRAC-P obs will be compared
			'frq_label': "",		# frequency label indicating which channels are selected,
									# here, it depends on hatpro_obs and mirac-p_obs
			'mwr_version': {'hatpro': 'v00',		# indicates the version of the radiometer data
							'mirac-p': 'v01'},
			'window': 900.0,		# time window in seconds (time over which to average observed TBs
			'plot_single': False,		# each frequency in its own plot, no subplots
			'plot_all_in_one': True,	# opposite of plot_single: plotting all freqs in one large figure
			'save_figures': True}


# initialize:
DS_dict = dict()		# will contain the xarray datasets

# load simulated data:
files = sorted(glob.glob(path_data['sim'] + "*_pamtra.nc"))
DS_dict['sim'] = xr.open_mfdataset(files, combine='nested', concat_dim='grid_x', preprocess=post_process_pamtra)

# rename some dimensions and variables because I aligned different times along grid_x:
DS_dict['sim'] = DS_dict['sim'].rename_dims({'grid_x': 'time'})
DS_dict['sim'] = DS_dict['sim'].rename({'grid_x': 'time'})
DS_dict['sim'] = DS_dict['sim'].assign_coords({'time': DS_dict['sim'].datatime})


# - load obs data
# - reduce TB time grid to radiosonde launch time:launch time + 15 minutes and compute mean and std dev of TBs:
# - compute mean and std dev
mwr_dict = {'hatpro': dict(),
			'mirac-p': dict()}
DS_overlap_dict = dict()
launch_time = DS_dict['sim'].time.values.astype("datetime64[s]").astype(np.float64)

if set_dict['hatpro_obs']:
	mwr_dict['hatpro'], DS_dict['hatpro'] = load_mwr_data(path_data['hatpro'], set_dict, 'hatpro')

	# radiometer and radiosonde overlap and compute mean and std. of TB over that overlap time
	DS_overlap_dict['hatpro'] = merge_sonde_and_mwr(mwr_dict['hatpro'], DS_dict['hatpro'], launch_time, set_dict, 'hatpro')

if set_dict['mirac-p_obs']:
	mwr_dict['mirac-p'], DS_dict['mirac-p'] = load_mwr_data(path_data['mirac-p'], set_dict, 'mirac-p')
	DS_dict['mirac-p'] = DS_dict['mirac-p'].isel(frequency=(DS_dict['mirac-p'].frequency < 300.0))
							
	# radiometer and radiosonde overlap and compute mean and std. of TB over that overlap time
	DS_overlap_dict['mirac-p'] = merge_sonde_and_mwr(mwr_dict['mirac-p'], DS_dict['mirac-p'], launch_time, set_dict, 'mirac-p')


# put it back into the dataset and merge HATPRO and MiRAC-P if needed:
if set_dict['hatpro_obs'] and not set_dict['mirac-p_obs']:
	DS_dict['obs_merged'] = xr.Dataset({'tb_mean': (['time', 'frequency'], DS_overlap_dict['hatpro'].tb_mean.values),
									'tb_std': (['time', 'frequency'], DS_overlap_dict['hatpro'].tb_std.values)},
									coords= {'time': launch_time.astype("datetime64[s]"),
											'frequency': DS_dict['hatpro'].frequency})

	# set frequency label:
	set_dict['frq_label'] += "K+V"


elif set_dict['mirac-p_obs'] and not set_dict['hatpro_obs']:
	DS_dict['obs_merged'] = xr.Dataset({'tb_mean': 	(['time', 'frequency'], DS_overlap_dict['mirac-p'].tb_mean.values,
													{'units': "K"}),
									'tb_std': 		(['time', 'frequency'], DS_overlap_dict['mirac-p'].tb_std.values,
													{'units': "K"})},
									coords= {'time': launch_time.astype("datetime64[s]"),
											'frequency': DS_dict['mirac-p'].frequency})

	# set frequency label:
	set_dict['frq_label'] += "G+243/340"

elif set_dict['hatpro_obs'] and set_dict['mirac-p_obs']:
	DS_dict['obs_merged'] = xr.merge([DS_overlap_dict['hatpro'], DS_overlap_dict['mirac-p']])

	# set frequency label:
	set_dict['frq_label'] += "K+V+G+243/340"

else:
	raise ValueError("No radiometer data??")


# select simulated tb frequencies according to available observed TBs:
frq_idx = select_MWR_channels(DS_dict['sim'].tb, DS_dict['sim'].frequency, set_dict['frq_label'], return_idx=2)
DS_dict['sim'] = DS_dict['sim'].isel(frequency=frq_idx)


assert np.all(DS_dict['obs_merged'].frequency.values == DS_dict['sim'].frequency.values)


# visualize:
fs = 16
fs_small = fs - 2
fs_dwarf = fs_small -2
c_H = (0.6,0.0,0.05)

# axis limits
ax_lim_DS = xr.DataArray(np.asarray([[10.0, 60.0],		# in K
									[10.0, 60.0],
									[10.0, 60.0],
									[10.0, 60.0],
									[10.0, 60.0],
									[10.0, 60.0],
									[10.0, 60.0],
									[100.0, 130.0],		# V band
									[130.0, 160.0],
									[230.0, 260.0],
									[265.0, 290.0],
									[265.0, 290.0],
									[265.0, 290.0],
									[265.0, 290.0],
									[265.0, 290.0],		# G band
									[260.0, 295.0],
									[260.0, 295.0],
									[260.0, 295.0],
									[240.0, 290.0],
									[190.0, 280.0],
									[125.0, 250.0]]),
					dims=['frequency', 'lim'], 
					coords={'frequency': np.array([	22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000,
									183.910, 184.810, 185.810, 186.810, 188.310, 190.810,
									243.000])})


if set_dict['plot_all_in_one']:

	fs = 12
	fs_small = fs - 2
	fs_dwarf = fs_small - 2
	fs_midget = fs_dwarf - 4

	f1, a1 = plt.subplots(ncols=6, nrows=4, figsize=(14.5,10), constrained_layout=True)

	a1 = a1.flatten()


	for k, freq in enumerate(DS_dict['sim'].frequency.values):

		# compute retrieval statistics:
		ret_stat_dict = compute_retrieval_statistics(DS_dict['obs_merged'].tb_mean.values[:,k],
														DS_dict['sim'].tb.values[:,k],
														compute_stddev=True)

		ax_lim = ax_lim_DS.sel(frequency=f"{freq:.2f}")

		# plotting:
		a1[k].errorbar(DS_dict['obs_merged'].tb_mean.values[:,k], DS_dict['sim'].tb.values[:,k], 
						xerr=DS_dict['obs_merged'].tb_std.values[:,k],
						ecolor=c_H, elinewidth=1.4, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.2) ##, label=f'{freq:.2f}$\,$GHz')


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_hatson = np.argwhere(~np.isnan(DS_dict['obs_merged'].tb_mean.values[:,k]) &
							~np.isnan(DS_dict['sim'].tb.values[:,k])).flatten()
		y_fit = DS_dict['sim'].tb.values[:,k][nonnan_hatson]
		x_fit = DS_dict['obs_merged'].tb_mean.values[:,k][nonnan_hatson]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = a1[k].plot(ax_lim, a*ax_lim + b, color=c_H, linewidth=1.2) ##, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		a1[k].plot(ax_lim, ax_lim, color=(0,0,0), linewidth=1.2, alpha=0.5) ##, label="Theoretical perfect fit")


		# add statistics:
		mean_both = np.nanmean(np.concatenate((DS_dict['sim'].tb.values[:,k], DS_dict['obs_merged'].tb_mean.values[:,k]), axis=0))
		a1[k].text(0.99, 0.01, f"N = {ret_stat_dict['N']} \nMean = {mean_both:.2f} \nbias = {ret_stat_dict['bias']:.2f} \n" +
				f"rmse = {ret_stat_dict['rmse']:.2f} \nstd. = {ret_stat_dict['stddev']:.2f} \nR = {ret_stat_dict['R']:.3f}",
				ha='right', va='bottom', transform=a1[k].transAxes, fontsize=fs_dwarf)

		# add subplot label:
		a1[k].text(0.02, 0.98, f"{freq:.2f} GHz", ha='left', va='top', transform=a1[k].transAxes, fontsize=fs_small)


		if k == 0:
			ds_fit = a1[k].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=1.2, label="Best fit")

			# plot a line for orientation which would represent a perfect fit:
			a1[k].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")
			
			# legend:
			leg_pos = np.asarray(a1[len(DS_dict['sim'].frequency.values)].get_position())
			leg_handles, leg_labels = a1[k].get_legend_handles_labels()
			f1.legend(handles=leg_handles, labels=leg_labels, loc='lower left',
						bbox_to_anchor=(leg_pos[0,0],leg_pos[0,1]), fontsize=fs,
						framealpha=0.5)


		# set axis limits and aspect ratio:
		a1[k].set_xlim(ax_lim)
		a1[k].set_ylim(ax_lim)
		a1[k].set_aspect('equal')

		# set ticks and tick labels and parameters:
		a1[k].tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1[k].minorticks_on()
		a1[k].grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		if k%6 == 0:
			a1[k].set_ylabel("TB$_{\mathrm{sim}}$ (K)", fontsize=fs)

		if k >= 18:
			a1[k].set_xlabel("TB$_{\mathrm{obs}}$ (K)", fontsize=fs)

	# remaining axes removed or used for legend:
	for k in range(len(DS_dict['sim'].frequency.values), len(a1)):
		a1[k].axis('off')
	
	f1.suptitle("WALSEMA simulated vs. observed TBs", fontsize=fs)

	if set_dict['save_figures']:
		plotname = "WALSEMA_mirac-p_radiosonde_pamtra_tb_scatterplot_all_freqs"
		f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
	else:
		plt.show()
		# pdb.set_trace()



if set_dict['plot_single']:
	# loop over frequencies:
	for k, freq in enumerate(DS_dict['sim'].frequency.values):
		# compute retrieval statistics:
		ret_stat_dict = compute_retrieval_statistics(DS_dict['obs_merged'].tb_mean.values[:,k],
														DS_dict['sim'].tb.values[:,k],
														compute_stddev=True)



		f1 = plt.figure(figsize=(9,9))
		a1 = plt.axes()

		ax_lim = ax_lim_DS.sel(frequency=f"{freq:.2f}")

		# plotting:
		a1.errorbar(DS_dict['obs_merged'].tb_mean.values[:,k], DS_dict['sim'].tb.values[:,k], 
							xerr=DS_dict['obs_merged'].tb_std.values[:,k],
							ecolor=c_H, elinewidth=1.4, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
							linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.2, label=f'{freq:.2f}$\,$GHz')

		for kk in range(len(DS_dict['sim'].time)): 
			a1.annotate(f"{kk}", (DS_dict['obs_merged'].tb_mean.values[kk,k], DS_dict['sim'].tb.values[kk,k]),
						xytext=(0.0, 1.5), xycoords='data', textcoords='offset points', annotation_clip=True, 
						ha='center', va='bottom', fontsize=fs_dwarf)


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_hatson = np.argwhere(~np.isnan(DS_dict['obs_merged'].tb_mean.values[:,k]) &
							~np.isnan(DS_dict['sim'].tb.values[:,k])).flatten()
		y_fit = DS_dict['sim'].tb.values[:,k][nonnan_hatson]
		x_fit = DS_dict['obs_merged'].tb_mean.values[:,k][nonnan_hatson]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = a1.plot(ax_lim, a*ax_lim + b, color=c_H, linewidth=1.2, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		a1.plot(ax_lim, ax_lim, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")


		# add statistics:
		mean_both = np.nanmean(np.concatenate((DS_dict['sim'].tb.values[:,k], DS_dict['obs_merged'].tb_mean.values[:,k]), axis=0))
		a1.text(0.99, 0.01, f"N = {ret_stat_dict['N']} \nMean = {mean_both:.2f} \nbias = {ret_stat_dict['bias']:.2f} \n" +
				f"rmse = {ret_stat_dict['rmse']:.2f} \nstd. = {ret_stat_dict['stddev']:.2f} \nR = {ret_stat_dict['R']:.3f}",
				ha='right', va='bottom', transform=a1.transAxes, fontsize=fs_small)

		# legend:
		leg_handles, leg_labels = a1.get_legend_handles_labels()
		a1.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.05, 1.00), fontsize=fs,
					framealpha=0.5)


		# set axis limits and aspect ratio:
		a1.set_xlim(ax_lim)
		a1.set_ylim(ax_lim)
		a1.set_aspect('equal')

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("TB$_{\mathrm{obs}}$ (K)", fontsize=fs)
		a1.set_ylabel("TB$_{\mathrm{sim}}$ (K)", fontsize=fs)
		a1.set_title(f"WALSEMA simulated vs. observed TBs, {freq:.2f}$\,$GHz", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"WALSEMA_mirac-p_radiosonde_pamtra_tb_scatterplot_{k}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			# pdb.set_trace()