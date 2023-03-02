import numpy as np
import datetime as dt
import glob
import pdb
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
import os

import sys
sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')    # so that functions and modules from MOSAiC can be used
from data_tools import *


"""
	Quick plots of radar reflectivity and cloud masks from WALES, MWR and radar.
	WALES doesn't have a cloud mask variable as the others do (at least on in 
	V1, only in V1.1 (only for 20200205)).
"""


dates = [
		# '20200119',
		# '20200122', # no radar
		# '20200124',
		# '20200126',
		# '20200128',
		# '20200130',
		# '20200131',
		'20200202',
		# '20200205',
		# '20200207',
		# '20200209',
		# '20200211',
		# '20200213',
		# '20200215',
		# '20200218'
		]

path_cloudmask_mwr_radar = "/net/blanc/awalbroe/Data/EUREC4A/HAMP_cloud_mask/"
path_unified = "/data/obs/campaigns/eurec4a/HALO/unified/v0.9/"
path_plots = "/net/blanc/awalbroe/Plots/EUREC4A/HAMP_cloudmask/out_plot_cloud_masks/"
path_plots_refl_mask_TB_plot = "/net/blanc/awalbroe/Plots/EUREC4A/HAMP_cloudmask/cm_and_tb/"
basic_plot = False			# basic plot of radar refl. + lidar cloud top + cloud masks
refl_mask_TB_plot = True	# plot including radar refl. (+ lidar cloud top) + cloud masks + MWR TBs
include_freezing_lvl = True		# whether or not to include freezing level in the refl_mask_TB_plot
								# requires HALO dropsondes
TB_band = 'G'				# Specify the frequencies to be selected. Valid options:
							# 'K': 20-40 GHz, 'V': 50-60 GHz, 'W': 85-95 GHz, 'F': 110-130 GHz, 'G': 165-205 GHz
							# Combinations are also possible: e.g. 'K+V+W' = 20-95 GHz
					
save_figures = True

cloudmask_mwr_radar_version = "0.9.1"
WALES_version = "1"
unified_version = "0.9"


# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)
path_plots_dir = os.path.dirname(path_plots_refl_mask_TB_plot)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)


for date in dates:

	print(date)
	# Stati: 1 if file for current date is available; 0 if not
	mwr_cm_status = 0
	radar_cm_status = 0
	wales_status = 0
	radar_status = 0
	mwr_tb_status = 0
	dropsonde_status = 0

	# Files:
	path_WALES = f"/data/obs/campaigns/eurec4a/HALO/{date}/WALES-LIDAR/"

	mwr_cm_file = glob.glob(path_cloudmask_mwr_radar + f"EUREC4A_HALO_HAMP-MWR_cloud_mask_{date}_v{cloudmask_mwr_radar_version}.nc")
	radar_cm_file = glob.glob(path_cloudmask_mwr_radar + f"EUREC4A_HALO_HAMP-Radar_cloud_mask_{date}_v{cloudmask_mwr_radar_version}.nc")
	wales_file = glob.glob(path_WALES + f"EUREC4A_HALO_WALES_cloudtop_{date}a_V{WALES_version}.nc")
	radar_file = glob.glob(path_unified + f"radar_{date}_v{unified_version}.nc")
	mwr_tb_file = []
	dropsonde_file = []

	if len(mwr_cm_file) > 0:
		mwr_cm_file = mwr_cm_file[0]
		mwr_cm_status = 1
	if len(radar_cm_file) > 0:
		radar_cm_file = radar_cm_file[0]
		radar_cm_status = 1
	if len(wales_file) > 0:
		wales_file = wales_file[0]
		wales_status = 1
	if len(radar_file) > 0:
		radar_file = radar_file[0]
		radar_status = 1

	if refl_mask_TB_plot: # import HAMP MWR TB data
		mwr_tb_file = glob.glob(path_unified + f"radiometer_{date}_v{unified_version}.nc")

		if len(mwr_tb_file) > 0:
			mwr_tb_file = mwr_tb_file[0]
			mwr_tb_status = 1

		if include_freezing_lvl:
			dropsonde_file = glob.glob(path_unified + f"dropsondes_{date}_v{unified_version}.nc")

			if len(dropsonde_file) > 0:
				dropsonde_file = dropsonde_file[0]
				dropsonde_status = 1


	# Open datasets:
	if mwr_cm_status: MWR_CM_DS = xr.open_dataset(mwr_cm_file)
	if radar_cm_status: RADAR_CM_DS = xr.open_dataset(radar_cm_file)
	if wales_status: WALES_DS = xr.open_dataset(wales_file)
	if radar_status: RADAR_DS = xr.open_dataset(radar_file)
	if mwr_tb_status: MWR_TB_DS = xr.open_dataset(mwr_tb_file)
	if dropsonde_status: DS_DS = xr.open_dataset(dropsonde_file)

	# check if MWR and RADAR cloud mask have got the same time axis:
	# check if cloud mask and MWR TB have got the same time axis:
	# check if MWR TB and radar data have got the same time axis
	# -> (they should because it's a unified data set)
	# np.all doesn't quite work here because of small time uncertainties
	assert np.all(MWR_CM_DS.time.values == RADAR_CM_DS.time.values)
	if mwr_tb_status:
		assert np.all(np.abs(MWR_TB_DS.time.values - MWR_CM_DS.time.values) < np.timedelta64(500, 'ms'))
		assert np.all(MWR_TB_DS.time.values == RADAR_DS.time.values)


	# WALES cloud mask based on either cloud top, cloud optical thickness
	# or cloud flag:
	#####################################################################
	if wales_status:
		# Option 1: use cloud optical thickness
		# wales_cloud_mask_idx = np.where(np.isfinite(WALES_DS.cloud_ot.values))[0]		########## change to other variables
		# wales_cloud_mask = np.zeros(WALES_DS.time.shape, dtype=np.float32)
		# wales_cloud_mask[wales_cloud_mask_idx] = 1
		# WALES_DS['pseudo_cloud_mask'] = xr.DataArray(wales_cloud_mask,
													# coords={'time': WALES_DS.time},
													# dims=['time'])
		# Option 2: use cloud flag
		WALES_DS['pseudo_cloud_mask'] = WALES_DS.cloud_flag

	if date == "20200122": radar_status=0		# because it didn't work on that day but a file exists anyway
	
	if basic_plot:

		# PLOTTING:
		fs = 18
		c_mwr = (31/255, 119/255, 180/255)
		c_cr = (255/255, 127/255, 14/255)
		c_wa = (75/255, 71/255, 153/255)

		# Gridded subplots:
		fig1 = plt.figure(figsize=(19,7))
		fig1.suptitle(dt.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"), fontsize=fs)
		ax0 = plt.subplot2grid((3,1), (0,0), rowspan=2)
		ax1 = plt.subplot2grid((3,1), (2,0), rowspan=1)

		# Plot radar reflectivity:
		if radar_status and date != '20200122':	# on that date the radar didn't work
			# Mesh coordinates:
			xv, yv = np.meshgrid(RADAR_DS.height.values, RADAR_DS.time.values)
			cmap = mpl.cm.plasma
			bounds = np.arange(-40, 45, 1).tolist()
			normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)

			radar_plot = ax0.pcolormesh(yv, xv, RADAR_DS.dBZ, cmap=cmap, norm=normalize, shading='flat')

			ax0.plot(RADAR_DS.time, RADAR_DS.altitude, color=(0,0,0), linewidth=0.8, label="Flight altitude")

			# Colorbar:
			cb1 = fig1.colorbar(mappable=radar_plot, ax=ax0,
								boundaries=bounds, # to get the triangles at the colorbar boundaries
								spacing='proportional',
								extend='both', fraction=0.075, orientation='vertical', pad=0)
			cb1.set_label(label="%s"%(RADAR_DS.dBZ.name), fontsize=fs-4)	# change fontsize of colorbar label			############### delete UNITS
			cb1.ax.tick_params(labelsize=fs-4)		# change size of colorbar ticks

		if wales_status:
			ax0.plot(WALES_DS.time, WALES_DS.cloud_top, color=c_wa, linestyle='none', marker='.', markersize=1.0,
						markerfacecolor=c_wa, markeredgecolor=c_wa, alpha=0.25)
			# dummy plot for legend:
			ax0.plot([np.nan, np.nan], [np.nan, np.nan], color=c_wa, linestyle='none', marker='.', markersize=5.0,
						markerfacecolor=c_wa, markeredgecolor=c_wa, alpha=0.5, label="WALES cloud top")

		lh0, ll0 = ax0.get_legend_handles_labels()
		ax0.legend(handles=lh0, labels=ll0, loc="upper right", fontsize=fs-6)

		# ax0.set_ylim(bottom=ylims[0], top=ylims[1])																				###################### change y lims ?
		ax0.set_title("HAMP cloud radar; WALES cloud top", fontsize=fs, pad=0)
		# ax0.set_xlabel("TB$_{\mathrm{sim}}$ (K)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
		ax0.tick_params(axis='both', labelsize=fs-5)
		ax0.minorticks_on()
		ax0.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


		# Plot cloud mask(s):
		if mwr_cm_status: ax1.plot(MWR_CM_DS.time, MWR_CM_DS.cloud_mask, '.', color=c_mwr, markersize=4.0, label="MWR")
		if radar_cm_status: ax1.plot(RADAR_CM_DS.time, RADAR_CM_DS.cloud_mask+0.1, '.', color=c_cr, markersize=4.0, label="Cloud radar")
		if wales_status: ax1.plot(WALES_DS.time, WALES_DS.pseudo_cloud_mask+0.2, '.', color=c_wa, markersize=4.0, label="WALES")

		lh, ll = ax1.get_legend_handles_labels()
		ax1.legend(handles=lh, labels=ll, loc='lower right', fontsize=fs-6)

		if radar_status or wales_status:
			ax1.set_xlim(ax0.get_xlim())
		ax1.set_ylim(bottom=-0.2, top=2.4)
		ax1.set_yticks(RADAR_CM_DS.cloud_mask.flag_values)
		ax1.set_yticklabels(RADAR_CM_DS.cloud_mask.flag_meanings.split())
		ax1.tick_params(axis='both', labelsize=fs-5)
		ax1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# limit plot width to radar plot width:
		ax0_pos = ax0.get_position().bounds
		ax1_pos = ax1.get_position().bounds
		ax1.set_position([ax1_pos[0], ax1_pos[1], ax0_pos[2], ax1_pos[3]])

		if save_figures:
			fig1.savefig(path_plots + "EUREC4A_HALO_HAMP_WALES_cloud_mask_%s.png"%(date), dpi=400)
		else:
			plt.show()

		if mwr_cm_status: MWR_CM_DS.close()
		if radar_cm_status: RADAR_CM_DS.close()
		if wales_status: WALES_DS.close()
		if radar_status: RADAR_DS.close()


	if refl_mask_TB_plot:

		# select TB frequencies for the plot:
		# check length of TB_band: if > 1: combination of bands desired:
		frq_idx_sel = select_MWR_channels(MWR_TB_DS.tb.values, MWR_TB_DS.frequency.values, band=TB_band,
												return_idx=2)
		MWR_TB_DS = MWR_TB_DS.isel(frequency=frq_idx_sel)

		# Compute freezing level if desired:
		if include_freezing_lvl: # time, height, sonde_number, ta, launch_time
			n_sondes = len(DS_DS.sonde_number.values)
			sonde_height = DS_DS.height.values
			n_height_sondes = len(sonde_height)
			sonde_T = DS_DS.ta.values + 273.15		# in K
			sonde_number = DS_DS.sonde_number.values.astype(np.int32) - 1
			freezing_level = np.full((n_sondes,), np.nan)
			sonde_time = DS_DS.launch_time.values
			for kkk in sonde_number:
				if np.count_nonzero(np.isnan(sonde_T[kkk,:]))/n_height_sondes < 0.5:
					freezing_level[kkk] = sonde_height[np.nanargmin(np.abs(sonde_T[kkk,:] - 273.15))]


		# PLOTTING:
		fs = 18
		c_mwr = (31/255, 119/255, 180/255)
		c_cr = (240/255, 0/255, 0/255)
		c_wa = (75/255, 71/255, 153/255)

		# Gridded subplots:
		fig1 = plt.figure(figsize=(19,7))
		fig1.suptitle(dt.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"), fontsize=fs)
		ax0 = plt.subplot2grid((3,1), (0,0), rowspan=1)
		ax1 = plt.subplot2grid((3,1), (1,0), rowspan=2)

		# Plot radar reflectivity:
		if radar_status and date != '20200122':	# on that date the radar didn't work
			# Mesh coordinates:
			xv, yv = np.meshgrid(RADAR_DS.height.values, RADAR_DS.time.values)
			cmap = mpl.cm.plasma
			bounds = np.arange(-40, 45, 1).tolist()
			normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)

			radar_plot = ax0.pcolormesh(yv, xv, RADAR_DS.dBZ, cmap=cmap, norm=normalize, shading='flat')

			ax0.plot(RADAR_DS.time, RADAR_DS.altitude, color=(0,0,0), linewidth=0.8, label="Flight altitude")

			# Colorbar:
			cb1 = fig1.colorbar(mappable=radar_plot, ax=ax0,
								boundaries=bounds, # to get the triangles at the colorbar boundaries
								spacing='proportional',
								extend='both', fraction=0.075, orientation='vertical', pad=0)
			cb1.set_label(label="%s"%(RADAR_DS.dBZ.name), fontsize=fs-4)	# change fontsize of colorbar label			############### delete UNITS
			cb1.ax.tick_params(labelsize=fs-4)		# change size of colorbar ticks

		if include_freezing_lvl:
			ax0.plot(sonde_time, freezing_level, color=(0,0,0), linewidth=1.6, label='Freezing level')

		# include cloud masks in plot:
		# if mwr_cm_status: 
			# # Translate cloud mask values to the height axis of the radar refl. plot:
			# MWR_CM_DS.cloud_mask[MWR_CM_DS.cloud_mask < 2.0] = 0		# ignore 'probably cloudy'
			# MWR_CM_DS.cloud_mask[MWR_CM_DS.cloud_mask >= 2.0] = 4800
			# ax0.plot(MWR_CM_DS.time, MWR_CM_DS.cloud_mask, '.', color=c_mwr, markersize=8.0, label="MWR")
		ylims = [-100, 6100]
		if radar_cm_status:
			# Translate cloud mask values to the height axis of the radar refl. plot:
			RADAR_CM_DS.cloud_mask[RADAR_CM_DS.cloud_mask < 2.0] = ylims[0] + 100		# ignore 'probably cloudy'
			RADAR_CM_DS.cloud_mask[RADAR_CM_DS.cloud_mask >= 2.0] = ylims[1] - 100
			ax0.plot(RADAR_CM_DS.time, RADAR_CM_DS.cloud_mask+0.1, '|', color=c_cr, markersize=6.0, label="CM")


		lh0, ll0 = ax0.get_legend_handles_labels()
		ax0.legend(handles=lh0, labels=ll0, loc="upper right", fontsize=fs-6)

		ax0.set_ylim(bottom=ylims[0], top=ylims[1])
		ax0.set_title("HAMP cloud radar reflectivity and cloud mask (CM)", fontsize=fs, pad=0)
		ax0.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
		ax0.tick_params(axis='both', labelsize=fs-5)
		ax0.minorticks_on()
		ax0.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


		# Plot TBs:
		if mwr_tb_status:
			n_freq = len(MWR_TB_DS.frequency)
			cmap = plt.cm.get_cmap('tab20', n_freq)
			for kk, freq in enumerate(MWR_TB_DS.frequency):
				ax1.plot(MWR_TB_DS.time, MWR_TB_DS.tb[:,kk], color=cmap(kk), linewidth=1.5, 
					marker='.', markersize=1.0)

				# dummy:
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], color=cmap(kk), linewidth=1.5,
							label="%.2f"%MWR_TB_DS.frequency.values[kk])
		

		lh, ll = ax1.get_legend_handles_labels()
		ax1.legend(handles=lh, labels=ll, loc='upper right', bbox_to_anchor=(1.1,1.0), fontsize=fs-6)

		if radar_status or mwr_cm_status or radar_cm_status:
			ax1.set_xlim(ax0.get_xlim())
		# # # ax1.set_ylim(bottom=-0.2, top=2.4)
		ax1.set_ylabel("TB (K)", fontsize=fs, labelpad=1.0)
		ax1.tick_params(axis='both', labelsize=fs-5)
		ax1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# limit plot width to radar plot width:
		ax0_pos = ax0.get_position().bounds
		ax1_pos = ax1.get_position().bounds
		ax1.set_position([ax1_pos[0], ax1_pos[1], ax0_pos[2], ax1_pos[3]])

		if save_figures:
			fig1.savefig(path_plots_refl_mask_TB_plot + 
							"EUREC4A_HALO_HAMP_TBs_cloud_mask_%s_%s.png"%(TB_band.replace("+", ""),date), dpi=400)
		else:
			plt.show()

		if mwr_cm_status: MWR_CM_DS.close()
		if radar_cm_status: RADAR_CM_DS.close()
		if wales_status: WALES_DS.close()
		if radar_status: RADAR_DS.close()
		if mwr_tb_status: MWR_TB_DS.close()


print("Done")