import numpy as np
import datetime as dt
# import pandas as pd
import copy
import pdb
import matplotlib.pyplot as plt
import os
import glob
import warnings

import sys
sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")

from import_data import *
from met_tools import *
from data_tools import *
import xarray as xr

import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


################################################################


# Paths:
path_data = {'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l2/",		# path of hatpro derived products
				'radiosondes': {'raw': "/data/radiosondes/Polarstern/PS131_ATWAICE_upper_air_soundings/"}}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/mwr_level_2bc_radiosonde/"	# path of output


# Select one of the following plot_options:		###################################################
# 0: Omit flagged values:  Each data point is plotted, outliers and when flag > 0 are left out.
# 1: Like 0 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range.
plot_option = 1
considered_period = 'walsema'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2022-07-07 - 2022-08-12
									# 'walsema': entire walsema campaign (2022-06-28 - 2022-08-12)
									# 'user': user defined

plot_T_and_q_std_bias = True			# standard deviation and bias profile plots
save_figures = True
save_figures_eps = False		# save figures as vector graphics (pdf or eps)
with_titles = False				# if True, plots will have titles (False for publication plots)
which_retrievals = 'both'		# which data is to be imported: 'both' contains both profiles (Temperature and humidity)
								# oother ptions: 'ta' or 'hus' (for temperature x.or humidity profile only).
radiosonde_version = 'raw'			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'

# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2022-07-07", "2022-08-12"],
					'walsema': ["2022-06-28", "2022-08-12"],
					'user': ["2020-04-13", "2020-04-23"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"

# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)


# choose data paths
path_radiosondes = path_data['radiosondes'][radiosonde_version]

# instrument status indicating which instrument is used. Not for manual editing! The stati are set automatically!
instrument_status = {'hatpro': 0,			# if 0, HATPRO data not included, if 1, HATPRO data included
					'sonde': 0}				# same for radiosondes


# import sonde and hatpro level 2b data:
# and create a datetime variable for plotting.
hatpro_dict = import_hatpro_level2b_daterange(path_data['hatpro'], date_start, date_end, 
											which_retrieval=which_retrievals, vers='v00', campaign='walsema', around_radiosondes=True,
											path_radiosondes=path_radiosondes, s_version=radiosonde_version,
											mwr_avg=1800, verbose=1)

instrument_status['hatpro'] = 1
if which_retrievals in ['ta', 'both']:	# boundary layer scan additionally loaded if temperature profiles are asked
	hatpro_bl_dict = import_hatpro_level2c_daterange(path_data['hatpro'], date_start, date_end, 
											which_retrieval=which_retrievals, vers='v00', campaign='walsema', around_radiosondes=True,
											path_radiosondes=path_radiosondes, sample_time_tolerance=1800, verbose=1)






# # # # # # # calibration times of HATPRO: manually entered from MWR logbook
# # # # # # if skip_pre_calib:
	# # # # # # calibration_times_HATPRO = [dt.datetime(2022,7,30,6,25)]
	# # # # # # calibration_times_MiRAC = [dt.datetime(2022,7,30,7,14)]
# # # # # # else:
	# # # # # # calibration_times_HATPRO = [dt.datetime(2022,7,7,9,53), 
								# # # # # # dt.datetime(2022,7,30,6,25)]
	# # # # # # calibration_times_MiRAC = [dt.datetime(2022,7,7,10,20), dt.datetime(2022,7,30,7,14)]

# # # # # # n_calib_HATPRO = len(calibration_times_HATPRO)
# # # # # # n_calib_MiRAC = len(calibration_times_MiRAC)








# Load radiosonde data:
print("Loading radiosonde data....")
# load radiosonde data: concat each sonde to generate a (sonde_launch x height) dict.
# interpolation of radiosonde data to regular grid requried to get a 2D array
files = sorted(glob.glob(path_radiosondes + "*.txt"))
sonde_dict_temp = import_radiosondes_PS131_txt(files)
new_height = np.arange(0.0, 20000.0001, 20.0)
n_height = len(new_height)
n_sondes = len(sonde_dict_temp.keys())
sonde_dict = {'temp': np.full((n_sondes, n_height), np.nan),
				'pres': np.full((n_sondes, n_height), np.nan),
				'relhum': np.full((n_sondes, n_height), np.nan),
				'height': np.full((n_sondes, n_height), np.nan),
				'wdir': np.full((n_sondes, n_height), np.nan),
				'wspeed': np.full((n_sondes, n_height), np.nan),
				'q': np.full((n_sondes, n_height), np.nan),
				'IWV': np.full((n_sondes,), np.nan),
				'launch_time': np.zeros((n_sondes,)),			# in sec since 1970-01-01 00:00:00 UTC
				'launch_time_npdt': np.full((n_sondes,), np.datetime64("1970-01-01T00:00:00"))}

# interpolate to new grid:
for idx in sonde_dict_temp.keys():
	for key in sonde_dict_temp[idx].keys():
		if key not in ["height", "IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx),:] = np.interp(new_height, sonde_dict_temp[idx]['height'], sonde_dict_temp[idx][key])
		elif key == "height":
			sonde_dict[key][int(idx),:] = new_height
		elif key in ["IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx)] = sonde_dict_temp[idx][key]
del sonde_dict_temp
sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])

# compute absolute humidity:
sonde_dict['rho_v'] = convert_rh_to_abshum(sonde_dict['temp'], sonde_dict['relhum'])
instrument_status['sonde'] = 1


if (instrument_status['sonde'] == 0) and (instrument_status['hatpro'] == 0):
	print("Comparison with no data to compare? Are you sure?? Activating self-destruct...")
	1/0

# Create datetime out of the MWR times:
hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
if which_retrievals in ['both', 'ta']:
	hatpro_bl_dict['time_npdt'] = hatpro_bl_dict['time'].astype("datetime64[s]")


# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")


# Find indices when mwr specific time (or master time) equals a sonde launch time. Then average
# over sonde launchtime - sample_time_tolerance:launchtime + sample_time_tolerance and compute
# standard deviation as well.
sample_time_tolerance = 1800
sample_time_tolerance_bl = 1800		# tolerance for boundary layer scan

hatson_idx = np.asarray([np.argwhere((hatpro_dict['time'] >= lt) & 
				(hatpro_dict['time'] <= lt+sample_time_tolerance)).flatten() for lt in sonde_dict['launch_time']])
if which_retrievals in ['ta', 'both']:
	hatson_bl_idx = np.asarray([np.argwhere((hatpro_bl_dict['time'] >= lt-sample_time_tolerance_bl) & 
					(hatpro_bl_dict['time'] <= lt+sample_time_tolerance_bl)).flatten() for lt in sonde_dict['launch_time']])

# find where hatson, ... is not empty:
hatson_not_empty = np.asarray([kk for kk in range(len(hatson_idx)) if len(hatson_idx[kk]) > 0])


n_height_hatpro = len(hatpro_dict['height'])
if which_retrievals in ['both', 'ta']:
	n_height_hatpro_bl = len(hatpro_bl_dict['height'])
	hatpro_dict['ta_mean_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)
	hatpro_dict['ta_stddev_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)

	hatpro_bl_dict['ta_mean_sonde'] = np.full((n_sondes,n_height_hatpro_bl), np.nan)
	hatpro_bl_dict['ta_stddev_sonde'] = np.full((n_sondes,n_height_hatpro_bl), np.nan)

	k = 0
	for hat, hatbl in zip(hatson_idx, hatson_bl_idx):
		hatpro_dict['ta_mean_sonde'][k,:] = np.nanmean(hatpro_dict['ta'][hat,:], axis=0)
		hatpro_dict['ta_stddev_sonde'][k,:] = np.nanstd(hatpro_dict['ta'][hat,:], axis=0)
		hatpro_bl_dict['ta_mean_sonde'][k] = np.nanmean(hatpro_bl_dict['ta'][hatbl,:], axis=0)
		hatpro_bl_dict['ta_stddev_sonde'][k] = np.nanstd(hatpro_bl_dict['ta'][hatbl,:], axis=0)
		k += 1

if which_retrievals in ['both', 'hus']:
	hatpro_dict['hua_mean_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)
	hatpro_dict['hua_stddev_sonde'] = np.full((n_sondes,n_height_hatpro), np.nan)

	k = 0
	for hat in hatson_idx:
		hatpro_dict['hua_mean_sonde'][k,:] = np.nanmean(hatpro_dict['hua'][hat,:], axis=0)
		hatpro_dict['hua_stddev_sonde'][k,:] = np.nanstd(hatpro_dict['hua'][hat,:], axis=0)
		k += 1



########## Plotting ##########


# dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
fs = 24		# fontsize

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_RS = (1,0.435,0)			# radiosondes





if plot_T_and_q_std_bias: # standard deviation and bias profiles (as subplots):


	# Interpolate the heights to the coarser grid: This means, that the sonde data is interpolated
	# to the retrieval grid:
	height_hatpro = hatpro_dict['height']
	n_height_hatpro = len(height_hatpro)
	sonde_dict['height_hatpro'] = np.full((n_sondes, n_height_hatpro), np.nan)		# height on hatpro grid
	sonde_dict['temp_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# temp. on hatpro grid
	sonde_dict['rho_v_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# rho_v on hatpro grid
	for k in range(n_sondes):
		sonde_dict['height_hatpro'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['height'][k,:])
		sonde_dict['temp_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['temp'][k,:])
		sonde_dict['rho_v_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])


	# Compute RMSE:
	if which_retrievals in ['both', 'ta']:
		RMSE_temp_hatson = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'], sonde_dict['temp_hatgrid'], which_axis=0)
		RMSE_temp_hatson_bl = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'], sonde_dict['temp_hatgrid'], which_axis=0)

		# Compute Bias profile:
		BIAS_temp_hatson = np.nanmean(hatpro_dict['ta_mean_sonde'] - sonde_dict['temp_hatgrid'], axis=0)
		BIAS_temp_hatson_bl = np.nanmean(hatpro_bl_dict['ta_mean_sonde'] - sonde_dict['temp_hatgrid'], axis=0)

		# STD DEV profile: Update profiles of MWR; then compute RMSE profile again
		hatpro_dict['ta_mean_sonde_biascorr'] = hatpro_dict['ta_mean_sonde'] - BIAS_temp_hatson
		hatpro_bl_dict['ta_mean_sonde_biascorr'] = hatpro_bl_dict['ta_mean_sonde'] - BIAS_temp_hatson_bl
		STDDEV_temp_hatson = compute_RMSE_profile(hatpro_dict['ta_mean_sonde_biascorr'], sonde_dict['temp_hatgrid'], which_axis=0)
		STDDEV_temp_hatson_bl = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde_biascorr'], sonde_dict['temp_hatgrid'], which_axis=0)

	if which_retrievals in ['both', 'hus']:
		RMSE_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'], sonde_dict['rho_v_hatgrid'], which_axis=0)

		# Compute Bias profile:
		BIAS_rho_v_hatson = np.nanmean(hatpro_dict['hua_mean_sonde'] - sonde_dict['rho_v_hatgrid'], axis=0)

		# STD DEV profile: Update profiles of hatpro and mirac-p:
		hatpro_dict['hua_mean_sonde_biascorr'] = hatpro_dict['hua_mean_sonde'] - BIAS_rho_v_hatson
		STDDEV_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde_biascorr'], sonde_dict['rho_v_hatgrid'], which_axis=0)



	# Plotting:
	if which_retrievals in ['ta', 'both']:

		fig = plt.figure(figsize=(20,18))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile
	

		# bias profiles:
		ax_bias.plot(BIAS_temp_hatson, height_hatpro, color=(0,0,0), linewidth=3.0, label="zenith mode")
		ax_bias.plot(BIAS_temp_hatson_bl, height_hatpro, color=(0,0,0), linestyle='dashed', linewidth=2.5, label='BL mode')
		ax_bias.plot(np.full_like(height_hatpro, 0.0), height_hatpro, color=(0,0,0), linewidth=1.0)


		# std dev profiles:
		ax_std.plot(STDDEV_temp_hatson, height_hatpro, color=(0,0,0), linewidth=3.0, label="zenith mode")
		ax_std.plot(STDDEV_temp_hatson_bl, height_hatpro, color=(0,0,0), linestyle='dashed', linewidth=2.5, label='BL mode')


		# add figure identifier of subplots: a), b), ...
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:
		lh, ll = ax_bias.get_legend_handles_labels()
		ax_bias.legend(handles=lh, labels=ll, loc="upper right", fontsize=fs)

		lh, ll = ax_std.get_legend_handles_labels()
		ax_std.legend(handles=lh, labels=ll, loc="upper right", fontsize=fs)

		# axis lims:
		ax_bias.set_ylim(bottom=0, top=8000)
		ax_bias.set_xlim(left=-4, right=4)
		ax_std.set_ylim(bottom=0, top=8000)
		ax_std.set_xlim(left=0, right=3.5)

		# labels:
		ax_bias.set_xlabel("$\mathrm{T}_{\mathrm{HATPRO}} - \mathrm{T}_{\mathrm{Radiosonde}}$ (K)", fontsize=fs)
		ax_bias.set_ylabel("Height (m)", fontsize=fs)
		ax_std.set_xlabel("$\sigma_{\mathrm{T}}$ (K)", fontsize=fs)
		if with_titles:
			ax_bias.set_title("Temperature profile bias between\nHATPRO and radiosondes", fontsize=fs)
			ax_std.set_title("Temperature profile standard deviation ($\sigma_{\mathrm{T}}$)\nbetween HATPRO and radiosondes",
							fontsize=fs)

		# grid:
		ax_bias.minorticks_on()
		ax_bias.grid(which='major', axis='both')
		ax_std.minorticks_on()
		ax_std.grid(which='major', axis='both')

		# tick params:
		ax_bias.tick_params(axis='both', labelsize=fs-2)
		ax_bias.tick_params(axis='x', pad=7)
		ax_std.tick_params(axis='both', labelsize=fs-2)
		ax_std.tick_params(axis='x', pad=7)
		ax_std.yaxis.set_ticklabels([])

		# Limit axis spacing:
		plt.subplots_adjust(wspace=0.0)			# removes space between subplots

		# adjust axis positions:
		ax_pos = ax_bias.get_position().bounds
		ax_bias.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
		ax_pos = ax_std.get_position().bounds
		ax_std.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])


		plot_name = "WALSEMA_hatpro_radiosonde_temp_stddev_bias_profile"
		if save_figures: 
			fig.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
		elif save_figures_eps:
			fig.savefig(path_plots + plot_name + ".pdf", bbox_inches='tight')
		else:
			plt.show()
		plt.close()



	if which_retrievals in ['hus', 'both']:
		fig = plt.figure(figsize=(20,18))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lims = [0, 8000]			# in m
		ax2_bias_lims = [-60,60]		# relative bias (-60 to 60 %)


		# bias profiles:
		ax_bias.plot(1000*BIAS_rho_v_hatson, height_hatpro, color=(0.2,0.2,0.8,0.5), linewidth=3.0)
		ax_bias.plot(np.full_like(height_hatpro, 0.0), height_hatpro, color=(0,0,0), linewidth=1.0)

		# plot RELATIVE standard deviation: normed by radiosonde humidity:
		sonde_dict['rho_v_hatgrid_mean'] = np.nanmean(sonde_dict['rho_v_hatgrid'], axis=0)

		# plot RELATIVE bias: normed by radiosonde humidity:
		ax2_bias = ax_bias.twiny()

		# dummy plots copying the looks of ax_bias plots for legend:
		ax2_bias.plot([np.nan, np.nan], [np.nan, np.nan], color=(0.2,0.2,0.8,0.5), linewidth=3.0, label="$\Delta\\rho_{v}$")

		ax2_bias.plot(100*(BIAS_rho_v_hatson / sonde_dict['rho_v_hatgrid_mean']), height_hatpro, color=(0,0,0),
					linestyle='dashed', linewidth=2.5, label="$\Delta\\rho_{v}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")

		
		# std dev profiles:
		ax_std.plot(1000*STDDEV_rho_v_hatson, height_hatpro, color=(0.2,0.2,0.8,0.5), linewidth=3.0)

		ax2 = ax_std.twiny()

		# dummy plots copying the looks of ax_std plots for legend:
		ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0.2,0.2,0.8,0.5), linewidth=3.0, label="$\sigma_{\\rho_{v}}$")

		ax2.plot(100*(STDDEV_rho_v_hatson / sonde_dict['rho_v_hatgrid_mean']), height_hatpro, color=(0,0,0),
					linestyle='dashed', linewidth=2.5, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")

		ax2_lims = [0,100]		# relative standard deviation of humidity


		# add figure identifier of subplots (a, b,...)
		ax_bias.text(0.05, 0.98, "a)", fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_bias.transAxes)
		ax_std.text(0.05, 0.98, "b)", fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_std.transAxes)

		# legends:
		lh, ll = ax2_bias.get_legend_handles_labels()
		ax2_bias.legend(handles=lh, labels=ll, loc="center right", fontsize=fs)

		lh, ll = ax2.get_legend_handles_labels()
		ax2.legend(handles=lh, labels=ll, loc="center right", fontsize=fs)

		# axis limits:
		ax_bias.set_xlim(left=-1, right=1)
		ax_bias.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax2_bias.set_xlim(left=ax2_bias_lims[0], right=ax2_bias_lims[1])
		ax2_bias.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax_std.set_xlim(left=0, right=1.5)
		ax_std.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax2.set_xlim(left=ax2_lims[0], right=ax2_lims[1])
		ax2.set_ylim(bottom=y_lims[0], top=y_lims[1])

		# labels:
		ax_bias.set_xlabel("$\\rho_{v,\mathrm{HATPRO}} - \\rho_{v,\mathrm{Radiosonde}}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs, color=(0.2,0.2,0.8,0.75))
		ax2_bias.set_xlabel("$\\left( \\rho_{v,\mathrm{HATPRO}} - \\rho_{v,\mathrm{Radiosonde}} \\right)$ / $\overline{\\rho}_{v,\mathrm{Radiosonde}}$ ($\%$)", 
							fontsize=fs, labelpad=15, color=(0,0,0))
		ax_bias.set_ylabel("Height (m)", fontsize=fs)
		ax_std.set_xlabel("$\sigma_{\\rho_{v}}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs, color=(0.2,0.2,0.8,0.75))
		ax2.set_xlabel("$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{Radiosonde}}$ ($\%$)", fontsize=fs, labelpad=15, color=(0,0,0))
		if with_titles:
			ax_std.set_title("Humidity profile standard deviation ($\sigma_{\\rho_v}$)\nbetween HATPRO and radiosondes (RS)",
						fontsize=fs, pad=15)
			ax_bias.set_title("Humidity profile bias between\nHATPRO and radiosondes (RS)", fontsize=fs, pad=15)


		# grid:
		ax_bias.minorticks_on()
		ax_bias.grid(which='major', axis='both')
		ax_std.minorticks_on()
		ax_std.grid(which='major', axis='both')

		# tick_params:
		ax_bias.tick_params(axis='both', labelsize=fs-2)
		ax_bias.tick_params(axis='x', pad=7, labelcolor=(0.2,0.2,0.8,0.75))
		ax_std.tick_params(axis='both', labelsize=fs-2)
		ax_std.tick_params(axis='x', pad=7, labelcolor=(0.2,0.2,0.8,0.75))
		ax2.tick_params(axis='both', labelsize=fs-2)
		ax2.tick_params(axis='x', labelcolor=(0,0,0))
		ax2_bias.tick_params(axis='both', labelsize=fs-2)
		ax2_bias.tick_params(axis='x', labelcolor=(0,0,0))
		ax_std.yaxis.set_ticklabels([])

		# limit axis spacing:
		plt.subplots_adjust(wspace=0.0)			# removes space between subplots

		# adjust axis positions:
		ax_pos = ax_bias.get_position().bounds
		ax_bias.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
		ax_pos = ax_std.get_position().bounds
		ax_std.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])


		plot_name = "WALSEMA_hatpro_radiosonde_rho_v_stddev_bias_profile"
		if save_figures: 
			fig.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
		elif save_figures_eps:
			fig.savefig(path_plots + plot_name + ".pdf", bbox_inches='tight')
		else:
			plt.show()
		plt.close()
