import numpy as np
import datetime as dt
# import pandas as pd
import copy
import pdb
import matplotlib.pyplot as plt
import os
import glob
import warnings
from import_data import *
from met_tools import *
from data_tools import *
import xarray as xr
# import sys
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


################################################################


# Paths:
path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"			# hatpro derived products
path_radiosondes = {'level_2': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/",	# MOSAiC radiosondes (as nc)
					'mossonde': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/",
					'psYYMMDDwHH': "/data/testbed/datasets/MOSAiC/rs41/"}
path_plots = "/net/blanc/awalbroe/Plots/MOSAiC_radiosonde_quicklook/"		# path of output


# Select one of the following plot_options:		###################################################
# 0: Omit flagged values:  Each data point is plotted, outliers and when flag > 0 are left out.
# 1: Like 0 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range.
plot_option = 1
considered_period = 'mosaic'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# 'user': user defined
plot_T_and_q_std_bias = True			# standard deviation and bias profile plots
save_figures = True
save_figures_eps = False			# save figures as vector graphics (pdf or eps)
with_titles = False				# if True, plots will have titles (False for publication plots)
rel_q_std_bias_plot_alternative = False			# plots an alternative relative bias and std dev profile; plot_T_and_q_std_bias must be True
which_retrievals = 'both'		# which data is to be imported: 'both' contains both profiles (Temperature and humidity)
								# oother ptions: 'ta' or 'hus' (for temperature x.or humidity profile only).
radiosonde_version = 'level_2'			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'

# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-03-05", "2020-03-05"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"

# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

# plot name options:
# There are the following options for plot_name_option:
# "flag0": 					each data point is plotted, outliers are left out
plot_name_options = ["flag0", "flag0_noEEZ"]
plot_name_option = plot_name_options[plot_option]

# choose data paths
path_radiosondes = path_radiosondes[radiosonde_version]

# instrument status indicating which instrument is used. Not for manual editing! The stati are set automatically!
instrument_status = {'hatpro': 0,			# if 0, HATPRO data not included, if 1, HATPRO data included
					'sonde': 0}			# same for radiosondes


# import sonde and hatpro level 2b data:
# and create a datetime variable for plotting.
hatpro_dict = import_hatpro_level2b_daterange(path_hatpro_level2, date_start, date_end, 
											which_retrieval=which_retrievals, around_radiosondes=True,
											path_radiosondes=path_radiosondes, s_version=radiosonde_version,
											mwr_avg=900, verbose=1)

instrument_status['hatpro'] = 1
if which_retrievals in ['ta', 'both']:	# boundary layer scan additionally loaded if temperature profiles are asked
	hatpro_bl_dict = import_hatpro_level2c_daterange(path_hatpro_level2, date_start, date_end, 
											which_retrieval=which_retrievals, around_radiosondes=True,
											path_radiosondes=path_radiosondes, verbose=1)


# Load radiosonde data:
sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version=radiosonde_version, remove_failed=True, verbose=1)
instrument_status['sonde'] = 1
n_sondes = len(sonde_dict['launch_time'])

if (instrument_status['sonde'] == 0) and (instrument_status['hatpro'] == 0):
	print("Comparison with no data to compare? Are you sure?? Activating self-destruct...")
	1/0

# Create datetime out of the MWR times:
hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
if which_retrievals in ['both', 'ta']:
	hatpro_bl_dict['time_npdt'] = hatpro_bl_dict['time'].astype("datetime64[s]")
sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])

# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")


# Find indices when mwr specific time (or master time) equals a sonde launch time. Then average
# over sonde launchtime - sample_time_tolerance:launchtime + sample_time_tolerance and compute
# standard deviation as well.
sample_time_tolerance = 900
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


# Filter out Exclusive Economic Zones:
if plot_option == 1:
	# Exclusive Economic Zones (data within these regions may not be published):
	EEZ_periods_no_dt = {'range0': [datetime_to_epochtime(dt.datetime(2020,6,3,20,36)), 
									datetime_to_epochtime(dt.datetime(2020,6,8,20,0))],
					'range1': [datetime_to_epochtime(dt.datetime(2020,10,2,4,0)), 
								datetime_to_epochtime(dt.datetime(2020,10,2,20,0))],
					'range2': [datetime_to_epochtime(dt.datetime(2020,10,3,3,15)), 
								datetime_to_epochtime(dt.datetime(2020,10,4,17,0))]}

	# find when master time axis of each MWR is outside EEZ periods:
	# same for radiosondes:
	outside_eez = dict()
	if instrument_status['sonde']:
		outside_eez['sonde'] = np.full((n_sondes,), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['sonde'][(sonde_dict['launch_time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (sonde_dict['launch_time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False

	if instrument_status['hatpro']:
		outside_eez['hatpro'] = np.full((len(hatpro_dict['time']),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['hatpro'][(hatpro_dict['time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (hatpro_dict['time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False

		if which_retrievals in ['both', 'ta']:
			outside_eez['hatpro_bl'] = np.full((len(hatpro_bl_dict['time']),), True)
			for EEZ_range in EEZ_periods_no_dt.keys():
				outside_eez['hatpro_bl'][(hatpro_bl_dict['time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (hatpro_bl_dict['time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False

	# theoretically, all sonde data should be outside eezs
	assert np.all(outside_eez['sonde'])




# visualize:
set_dict = {'save_figures': True}
fs = 16
fs_small = fs - 2
fs_dwarf = fs_small - 2

# reduce to required year:
data_plot = sonde_dict['rho_v']*1000.0
data_plot_hgt = sonde_dict['height']

f1 = plt.figure(figsize=(9,11))
a1 = plt.axes()

x_lim = [0, 10]		# g kg-1
y_lim = [0, 10000]	# m
# plotting:
for k in range(n_sondes):
	a1.plot(data_plot[k,:], data_plot_hgt[k,:], linewidth=0.5, alpha=0.2)

# set axis limits:
a1.set_xlim(x_lim)
a1.set_ylim(y_lim)

# set ticks and tick labels and parameters:
a1.tick_params(axis='both', labelsize=fs_small)

# grid:
a1.minorticks_on()
a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

# set labels:
a1.set_xlabel("$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs)
a1.set_ylabel("Height (m)", fontsize=fs)
a1.set_title("Radiosonde absolute humidity $\\rho_v$", fontsize=fs)

if set_dict['save_figures']:
	plotname = "MOSAiC_radiosondes_all_abs_hum_profiles"
	f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
else:
	plt.show()




# reduce to required year:
data_plot = hatpro_dict['hua_mean_sonde']*1000.0
data_plot_hgt = hatpro_dict['height']

f1 = plt.figure(figsize=(9,11))
a1 = plt.axes()

x_lim = [0, 10]		# g kg-1
y_lim = [0, 10000]	# m
# plotting:
for k in range(n_sondes):
	a1.plot(data_plot[k,:], data_plot_hgt, linewidth=0.5, alpha=0.2)

# set axis limits:
a1.set_xlim(x_lim)
a1.set_ylim(y_lim)

# set ticks and tick labels and parameters:
a1.tick_params(axis='both', labelsize=fs_small)

# grid:
a1.minorticks_on()
a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

# set labels:
a1.set_xlabel("$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs)
a1.set_ylabel("Height (m)", fontsize=fs)
a1.set_title("HATPRO absolute humidity $\\rho_v$", fontsize=fs)

if set_dict['save_figures']:
	plotname = "MOSAiC_hatpro_all_abs_hum_profiles"
	f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
else:
	plt.show()
pdb.set_trace()

1/0




# MOSAiC Legs:
MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}


########## Plotting ##########


# dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
fs = 24		# fontsize

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_RS = (1,0.435,0)			# radiosondes


if plot_option == 1: # filter EEZ time stamp
	hatpro_dict['time'] = hatpro_dict['time'][outside_eez['hatpro']]
	hatpro_dict['flag'] = hatpro_dict['flag'][outside_eez['hatpro']]
	hatpro_dict['time_npdt'] = hatpro_dict['time'][outside_eez['hatpro']]
	if which_retrievals in ['both', 'ta']:
		hatpro_dict['ta'] = hatpro_dict['ta'][outside_eez['hatpro'],:]
	if which_retrievals in ['both', 'hus']:
		hatpro_dict['hua'] = hatpro_dict['hua'][outside_eez['hatpro'],:]



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

		# STD DEV profile: Update profiles of hatpro and mirac-p:
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


