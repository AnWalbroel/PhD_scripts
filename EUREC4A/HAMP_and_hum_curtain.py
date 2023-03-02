import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pdb
import glob
import os

import sys
import matplotlib as mpl
sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')
from data_tools import select_MWR_channels, numpydatetime64_to_epochtime, Gband_double_side_band_average
from met_tools import compute_IWV, e_sat, compute_LWC_from_Z
from my_classes_eurec4a import *


def HALO_DS_sim_preprocess(DS):
	"""
	Preprocessing the PAMTRA output file dataset before concatenation.
	Removing undesired dimensions (average over polarisation, use nadir only,
	remove y dimension). Additionally, it adds another variable (DataArray)
	containing the datatime in sec since epochtime.

	Parameters:
	-----------
	ds : xarray dataset
		Dataset of the PAMTRA output.
	"""

	# Nadir only ("0"),
	# Average over polarisation (".mean(axis=-1)")
	# Remove redundant dimensions: ("0,0")
	DS['tb'] = DS.tb[:,0,0,0,:,:].mean(axis=-1)

	# And add a variable giving the datatime in seconds since 1970-01-01 00:00:00 UTC
	# along dimension grid_x:
	DS['time'] = xr.DataArray(numpydatetime64_to_epochtime(DS.datatime[:,0].values),
								dims=['grid_x'])

	return DS


def plot_Z_LWC(HAMP_radar, path_plots, time_found, which_date):

	"""
	Quick plotting routine of Z-LWC relations.
	"""

	fs = 15
	fs_small = fs-2

	fig1 = plt.figure(figsize=(6,12))

	ax_nd = plt.subplot2grid((2,1), (0,0))
	ax_log = plt.subplot2grid((2,1), (1,0))			# log scale

	ax_nd.plot(HAMP_radar.LWC_ND, HAMP_radar.dBZ[time_found['ND'],:], linestyle='none',
				marker='.', markersize=2, color=(0,0,0))

	ax_nd.plot(HAMP_radar.LWC_LD, HAMP_radar.dBZ[time_found['LD'],:], linestyle='none',
				marker='.', markersize=2, color=(0.85,0.77,0.12))

	ax_nd.plot(HAMP_radar.LWC_HD, HAMP_radar.dBZ[time_found['HD'],:], linestyle='none',
				marker='.', markersize=2, color=(0.7,0,0))

	# dummies for legend:
	ax_nd.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0,0,0), label='no_drizzle')
	ax_nd.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0.85,0.77,0.12), label='light_drizzle')
	ax_nd.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0.7,0,0), label='heavy_drizzle')

	lh, ll = ax_nd.get_legend_handles_labels()
	ax_nd.legend(handles=lh, labels=ll, loc='lower right', fontsize=fs_small-2)

	ax_nd.minorticks_on()
	ax_nd.grid(which='both', axis='both')

	ax_nd.set_xlabel("LWC ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs, labelpad=0.5)
	ax_nd.set_ylabel("Radar refl. factor (dBZ)", fontsize=fs, labelpad=0.5)
	ax_nd.set_title("Z-LWC relation", fontsize=fs, pad=0.2)


	ax_log.plot(HAMP_radar.LWC_ND, HAMP_radar.dBZ[time_found['ND'],:], linestyle='none',
				marker='.', markersize=2, color=(0,0,0))

	ax_log.plot(HAMP_radar.LWC_LD, HAMP_radar.dBZ[time_found['LD'],:], linestyle='none',
				marker='.', markersize=2, color=(0.85,0.77,0.12))

	ax_log.plot(HAMP_radar.LWC_HD, HAMP_radar.dBZ[time_found['HD'],:], linestyle='none',
				marker='.', markersize=2, color=(0.7,0,0))

	# dummies for legend:
	ax_log.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0,0,0), label='no_drizzle')
	ax_log.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0.85,0.77,0.12), label='light_drizzle')
	ax_log.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none',
				marker='.', markersize=2, color=(0.7,0,0), label='heavy_drizzle')

	lh, ll = ax_log.get_legend_handles_labels()
	ax_log.legend(handles=lh, labels=ll, loc='lower right', fontsize=fs_small-2)

	ax_log.set_xscale('log')

	ax_log.minorticks_on()
	ax_log.grid(which='both', axis='both')

	ax_log.set_xlabel("$\mathrm{log}_{10}(\mathrm{LWC}, \mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs, labelpad=0.5)
	ax_log.set_ylabel("Radar refl. factor (dBZ)", fontsize=fs, labelpad=0.5)

	fig1.savefig(path_plots + f"Z-LWC_relation_HAMP_cloud_radar_EUREC4A_{which_date}.png", dpi=400)
	

"""
	This script visualises HAMP MWR brightness temperatures (TBs) together with
	relative and absolute humidity (time x height) from dropsondes. Additionally,
	HAMP cloud radar reflectivity is plotted.

	- Load radiometer, dropsonde and cloud radar data from HALO
	- Select wanted TBs
	- Interpolate dropsonde to HAMP time to have a 'curtain'
	- Plot
"""


# Settings:
save_figures = True			# if True, figures will be saved
mwr_band = 'G'				# select MWR band: options: 'K', 'V', 'W', 'F', 'G' 
							# and combinations, i.e.: 'K+V+G*
add_DS_TBs = False			# if True, simulated TBs from dropsonde measurements are included
compute_LWC = True			# if True, LWC will be computed using Z-LWC relations


dates = {'all': ['20200119',
				# '20200122', # no radar
				'20200124',
				'20200126',
				'20200128',
				'20200130',
				'20200131',
				'20200202',
				'20200205',
				'20200207',
				'20200209',
				'20200211',
				'20200213',
				#'20200215',
				'20200218'],
		'ice': 	[
				'20200119', 
				'20200209'
				],
		'liq':	[
				'20200124',
				'20200202'
				]}

dates = dates['all']		# default selection


# Paths:
unified_version = 'v0.9'
dropsonde_version = 'joanne_level_3'

path_unified = f"/data/obs/campaigns/eurec4a/HALO/unified/{unified_version}/"
path_dropsonde = "/data/obs/campaigns/eurec4a/HALO/JOANNE/"
path_dropsonde_TBs = "/net/blanc/awalbroe/Data/EUREC4A/forward_sim_dropsondes/pam_out_sonde/"
path_plots = "/net/blanc/awalbroe/Plots/EUREC4A/HAMP_and_hum_curtain/"


# check if plot path exists:
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(plot_path_dir)


for hh, which_date in enumerate(['20200202']):
	print(which_date)


	# Import HALO HAMP, lidar, and cloud mask data:
	HAMP_MWR = MWR(path_unified, version=unified_version, which_date=which_date, cut_low_altitude=False)
	HAMP_radar = radar(path_unified, version=unified_version, which_date=which_date, cut_low_altitude=False)

	HALO_dropsondes = dropsondes(path_dropsonde, version=dropsonde_version, which_date=which_date, save_time_height_matrix=False)
	HALO_dropsondes.ntime = len(HALO_dropsondes.launch_time)
	HALO_dropsondes.nheight = len(HALO_dropsondes.height)

	# Select MWR channels:
	HAMP_MWR.TB, HAMP_MWR.freq = select_MWR_channels(HAMP_MWR.TB, HAMP_MWR.freq, band=mwr_band, return_idx=0)


	# Compute IWV:
	HALO_dropsondes.iwv = np.asarray([compute_IWV(HALO_dropsondes.rho_v[k,:], 
										HALO_dropsondes.height) for k in range(HALO_dropsondes.ntime)])


	# Interpolate dropsonde data to have a full time x height matrix over the whole day:
	ds_vars = list(HALO_dropsondes.__dict__.keys())

	# loop through time height variables:
	ds_vars_th = [vars for vars in ds_vars if (vars not in ['ntime', 'nheight']) and (HALO_dropsondes.__dict__[vars].shape == 
											(HALO_dropsondes.ntime, HALO_dropsondes.nheight))]
	for var in ds_vars_th:

		HALO_dropsondes.__dict__[var + "_ip"] = np.full((len(HAMP_MWR.time), HALO_dropsondes.nheight), np.nan)

		# loop through all height levels and interpolate:
		for k in range(HALO_dropsondes.nheight):

			# only use those launch_times as anchor points if that launch doesn't show a nan:
			time_idx = np.where(~np.isnan(HALO_dropsondes.__dict__[var][:,k]))[0]
			if len(time_idx) > 0:
				HALO_dropsondes.__dict__[var + "_ip"][:,k] = np.interp(HAMP_MWR.time, HALO_dropsondes.launch_time[time_idx], 
																		HALO_dropsondes.__dict__[var][time_idx,k],
																		left=np.nan, right=np.nan)

		# replace uninterpolated variables:
		HALO_dropsondes.__dict__[var] = HALO_dropsondes.__dict__[var + "_ip"]
		del HALO_dropsondes.__dict__[var + "_ip"]


	# check if time axes are identical:
	assert np.all(HAMP_MWR.time == HAMP_radar.time)


	# Import simulated TBs from dropsondes if desired:
	if add_DS_TBs:
		all_sim_files = sorted(glob.glob(path_dropsonde_TBs + "*.nc"))
		HALO_DS_sim = xr.open_mfdataset(all_sim_files, concat_dim='grid_x', combine='nested', preprocess=HALO_DS_sim_preprocess)

		# find correct time indices and remove useless dimensions; apply double side average and select G band channels:
		time_idx = np.where((HALO_DS_sim.time.values >= HAMP_MWR.time[0]) & (HALO_DS_sim.time.values <= HAMP_MWR.time[-1]))[0]
		HALO_DS_sim = HALO_DS_sim.isel(grid_y=0, outlevel=0, angles=0, grid_x=time_idx)
		tb_dsba, frequency_dsba = Gband_double_side_band_average(HALO_DS_sim.tb.values, HALO_DS_sim.frequency.values)
		tb_dsba, frequency_dsba = select_MWR_channels(tb_dsba, frequency_dsba, band=mwr_band, return_idx=0)
		tb_dsba = tb_dsba[:,:-1]			# remove last G band freq because this doesn't appear in MWR data
		frequency_dsba = frequency_dsba[:-1]
		HALO_DS_sim = HALO_DS_sim.isel(frequency=np.arange(len(frequency_dsba)))
		HALO_DS_sim = HALO_DS_sim.assign_coords({'frequency': frequency_dsba})
		HALO_DS_sim = HALO_DS_sim.assign_coords({'grid_x': np.arange(len(time_idx))})		# must be done because of xarray logic
		HALO_DS_sim['tb'] = xr.DataArray(tb_dsba, dims=['grid_x', 'frequency'])

		# datetime object for plotting:
		HALO_DS_sim['time_dt'] = xr.DataArray(np.array([dt.datetime.utcfromtimestamp(ttt) for ttt in HALO_DS_sim.time.values]), 
												dims=['grid_x'])


	if compute_LWC:
		# filter cloud types:
		time_found = dict()

		# heavy drizzle (HD):
		# >10 indices with dBZ > 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask_HD = HAMP_radar.dBZ > 15
		time_found['HD'] = np.where(np.count_nonzero(search_mask_HD[:,height_idx], axis=1) > 10)[0]
		HAMP_radar.LWC_HD = compute_LWC_from_Z(HAMP_radar.Z[time_found['HD'],:], 'heavy_drizzle')

		# light drizzle (LD):
		# >10 indices with dBZ >= -15 but <= 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask_LD = ((HAMP_radar.dBZ >= -15) & (HAMP_radar.dBZ <= 15))
		time_found['LD'] = np.where(np.count_nonzero(search_mask_LD[:, height_idx], axis=1) > 10)[0]
		HAMP_radar.LWC_LD = compute_LWC_from_Z(HAMP_radar.Z[time_found['LD'],:], 'light_drizzle')

		# no drizzle (ND):
		# no drizzle threshold of -15 was chosen because of Fig. 1 and 6 in Khain et al. 2008,
		# Fig. 9 of Fox and Illingworth 1997, an because of Liu et al. 2008.
		# And dBZ must be < -38 in the lowest 300 m
		height_idx = np.where(HAMP_radar.height < 300)[0]
		search_mask1 = HAMP_radar.dBZ >= -15
		search_mask2 = HAMP_radar.dBZ[:,height_idx] >= -38
		time_found['ND'] = np.where((np.count_nonzero(search_mask1, axis=1) == 0) &
								(np.count_nonzero(search_mask2, axis=1) == 0))[0]

		HAMP_radar.LWC_ND = compute_LWC_from_Z(HAMP_radar.Z[time_found['ND'],:], 'no_drizzle', 
												algorithm='Fox_and_Illingworth_1997_ii')

		pdb.set_trace()
		# If wanted, you may generate some Z-LWC plots. If you don't want it, comment it out:
		# plot_Z_LWC(HAMP_radar, path_plots, time_found, which_date)
		# pdb.set_trace()


	# Plot:
	fs = 15
	fs_small = fs-2

	time_dt = np.array([dt.datetime.utcfromtimestamp(ttt) for ttt in HAMP_MWR.time])
	launch_time_dt = np.array([dt.datetime.utcfromtimestamp(ttt) for ttt in HALO_dropsondes.launch_time])
	y_lim_height = [0, 10500]
	x_lim = [time_dt[0], time_dt[-1]]

	mpl.rcParams.update({'xtick.labelsize': fs_small, 'ytick.labelsize': fs_small, 'axes.grid.which': 'major', 
							'axes.grid': True, 'axes.grid.axis': 'both', 'grid.color': (0.5,0.5,0.5),
							'axes.labelsize': fs})
	dt_fmt_major = mdates.DateFormatter("%H:%M")

	fig1 = plt.figure(figsize=(16,18))

	ax_cr = plt.subplot2grid((5,1), (0,0))			# cloud radar plot
	ax_q = plt.subplot2grid((5,1), (1,0))			# spec hum plot
	ax_rh = plt.subplot2grid((5,1), (2,0))			# rel hum plot
	ax_pw = plt.subplot2grid((5,1), (3,0))			# IWV plot
	ax_tb = plt.subplot2grid((5,1), (4,0))			# TB plot

	
	# Radar refl plot: Mesh coordinates:
	xv, yv = np.meshgrid(HAMP_radar.height, time_dt)
	cmap = mpl.cm.turbo
	bounds = np.arange(-40,50,5)
	normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)

	radar_plot = ax_cr.pcolormesh(yv, xv, HAMP_radar.dBZ, cmap=cmap, norm=normalize, shading='flat')

	# Colorbar:
	cb1 = fig1.colorbar(mappable=radar_plot, ax=ax_cr, boundaries=bounds, use_gridspec=False, spacing='proportional',
						extend='both', fraction=0.075, orientation='vertical', pad=0)
	cb1.set_label(label="dBZ", fontsize=fs_small)
	cb1.ax.tick_params(labelsize=fs_small-2)

	ax_cr.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


	# Spec hum. plot: in g kg^-1
	q_levels = np.arange(0,15.01,0.1)
	xv, yv = np.meshgrid(HALO_dropsondes.height, time_dt)
	q_cmap = mpl.cm.gist_earth
	q_curtain = ax_q.contourf(yv, xv, 1000*HALO_dropsondes.q, levels=q_levels, 
								cmap=q_cmap)

	# mark sonde drops:
	ax_q.plot(launch_time_dt, np.full(launch_time_dt.shape, (y_lim_height[1] - y_lim_height[0])*0.98 + y_lim_height[0]), 
				linestyle='none', color=(0,0,0), marker='v', markersize=5)

	# Colorbar:
	cb2 = fig1.colorbar(mappable=q_curtain, ax=ax_q, use_gridspec=False, extend='max',
						orientation='vertical', fraction=0.075, pad=0)
	cb2.set_label(label="q ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs_small)
	cb2.ax.tick_params(labelsize=fs_small-2)


	# RH plot: in %
	rh_levels = np.arange(0,100.001,2.5)
	rh_cmap = mpl.cm.gist_earth
	rh_curtain = ax_rh.contourf(yv, xv, 100*HALO_dropsondes.rh, levels=rh_levels,
									cmap=rh_cmap)

	# mark sonde drops:
	ax_rh.plot(launch_time_dt, np.full(launch_time_dt.shape, (y_lim_height[1] - y_lim_height[0])*0.98 + y_lim_height[0]), 
				linestyle='none', color=(0,0,0), marker='v', markersize=5)

	# Colorbar:
	cb3 = fig1.colorbar(mappable=rh_curtain, ax=ax_rh, use_gridspec=False, extend='neither',
						orientation='vertical', fraction=0.075, pad=0)
	cb3.set_label(label="RH (%)", fontsize=fs_small)
	cb3.ax.tick_params(labelsize=fs_small-2)


	# Plot IWV in mm:
	ax_pw.plot(launch_time_dt, HALO_dropsondes.iwv, color=(0,0,0), linewidth=1.3, marker='.', markersize=4)

	ax_pw.minorticks_on()
	ax_pw.grid(which='both', axis='y', color=(0.5,0.5,0.5), alpha=0.5)


	# Plot TBs:

	if add_DS_TBs:

		n_freq = len(HAMP_MWR.freq)
		tb_cmap = mpl.cm.get_cmap('tab10', n_freq)
		for kk, freq in enumerate(HAMP_MWR.freq):
			ax_tb.plot(time_dt, HAMP_MWR.TB[:,kk], color=tb_cmap(kk), linewidth=1.2,
						alpha=0.65)

			# dummy:
			ax_tb.plot([np.nan, np.nan], [np.nan, np.nan], color=tb_cmap(kk), linewidth=1.2,
								label=f"{freq:.2f}")

		n_freq_sim = len(HALO_DS_sim.frequency.values)
		for kk, freq in enumerate(HALO_DS_sim.frequency.values):
			ax_tb.plot(HALO_DS_sim.time_dt.values, HALO_DS_sim.tb.values[:,kk], color=tb_cmap(kk),
						marker='x', markersize=6, linestyle='none')
		# dummy:
		ax_tb.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), marker='x', markersize=6, 
					linestyle='none', label="Dropsonde, PAMTRA")

	else:
		n_freq = len(HAMP_MWR.freq)
		tb_cmap = mpl.cm.get_cmap('tab10', n_freq)
		for kk, freq in enumerate(HAMP_MWR.freq):
			ax_tb.plot(time_dt, HAMP_MWR.TB[:,kk], color=tb_cmap(kk), linewidth=1.2,
						marker='.', markersize=1.0)

			# dummy:
			ax_tb.plot([np.nan, np.nan], [np.nan, np.nan], color=tb_cmap(kk), linewidth=1.2,
								label=f"{freq:.2f}")


	# Legend:
	lh, ll = ax_tb.get_legend_handles_labels()
	ax_tb.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(1.02,1.05), fontsize=fs_small-2,
					title="Frequency (GHz)", title_fontsize=fs_small-2)
		

	# Set axis labels and titles:
	ax_cr.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
	ax_cr.set_title("HAMP cloud radar reflectivity", fontsize=fs, pad=0.1)

	ax_q.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
	ax_q.set_title("Dropsonde specific humidity q", fontsize=fs, pad=0.1)

	ax_rh.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
	ax_rh.set_title("Dropsonde relative humidity RH", fontsize=fs, pad=0.1)

	ax_pw.set_ylabel("IWV (mm)", fontsize=fs, labelpad=1.0)
	ax_pw.set_title("Dropsonde integrated water vapour (IWV)", fontsize=fs, pad=0.1)

	ax_tb.set_ylabel("TB (K)", fontsize=fs, labelpad=1.0)
	ax_tb.set_title("HAMP MWR Brightess Temperatures TB", fontsize=fs, pad=0.1)

	ax_tb.set_xlabel("%s-%s-%s"%(which_date[:4], which_date[4:6], which_date[6:]), fontsize=fs, labelpad=0.5)


	# Set axis limits:
	ax_cr.set_ylim(bottom=y_lim_height[0], top=y_lim_height[1])
	ax_q.set_ylim(bottom=y_lim_height[0], top=y_lim_height[1])
	ax_rh.set_ylim(bottom=y_lim_height[0], top=y_lim_height[1])
	ax_pw.set_ylim(bottom=0, top=np.ceil(1.05*np.nanmax(HALO_dropsondes.iwv)))

	ax_cr.set_xlim(left=x_lim[0], right=x_lim[1])
	ax_q.set_xlim(left=x_lim[0], right=x_lim[1])
	ax_rh.set_xlim(left=x_lim[0], right=x_lim[1])
	ax_pw.set_xlim(left=x_lim[0], right=x_lim[1])
	ax_tb.set_xlim(left=x_lim[0], right=x_lim[1])


	# handle some ticks and tick labels:
	ax_cr.minorticks_on()
	ax_q.minorticks_on()
	ax_rh.minorticks_on()
	ax_tb.minorticks_on()

	ax_cr.xaxis.set_ticklabels([])
	ax_q.xaxis.set_ticklabels([])
	ax_rh.xaxis.set_ticklabels([])
	ax_pw.xaxis.set_ticklabels([])
	ax_tb.xaxis.set_major_formatter(dt_fmt_major)


	# adjust subplot spacing:
	plt.subplots_adjust(hspace=0.23)			# removes space between subplots

	# limit axis width:
	ax_cr_pos = ax_cr.get_position().bounds
	ax_cr.set_position([ax_cr_pos[0], ax_cr_pos[1], ax_cr_pos[2]*0.9, ax_cr_pos[3]])

	ax_q_pos = ax_q.get_position().bounds
	ax_q.set_position([ax_q_pos[0], ax_q_pos[1], ax_q_pos[2]*0.9, ax_q_pos[3]])

	ax_rh_pos = ax_rh.get_position().bounds
	ax_rh.set_position([ax_rh_pos[0], ax_rh_pos[1], ax_rh_pos[2]*0.9, ax_rh_pos[3]])

	ax_pw_pos = ax_pw.get_position().bounds
	ax_pw.set_position([ax_pw_pos[0], ax_pw_pos[1], ax_pw_pos[2]*0.9, ax_pw_pos[3]])

	ax_tb_pos = ax_tb.get_position().bounds
	ax_tb.set_position([ax_tb_pos[0], ax_tb_pos[1], ax_tb_pos[2]*0.9, ax_tb_pos[3]])


	if save_figures:
		fig1.savefig(path_plots + f"HAMP_and_hum_curtain_EUREC4A_TBband_{mwr_band}_{which_date}.png", dpi=400)

	else:
		plt.show()

	plt.close()

	if add_DS_TBs: HALO_DS_sim.close()

print("Done....")