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
path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"
path_mirac = {'RPG': "/data/obs/campaigns/mosaic/mirac-p/l1/",
				'mwr_pro': ""}		# not existing
path_radiosondes = {'level_2': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/",
					'mossonde': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/",
					'psYYMMDDwHH': "/data/testbed/datasets/MOSAiC/rs41/"}
path_plots = "/net/blanc/awalbroe/Plots/MOSAiC_radiosonde_quicklook/"


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
plot_IWV_diff_hatson_mirson = False		# IWV difference: HATPRO - sonde, MiRAC-P - sonde
plot_T_and_q_prof_sonde = False
plot_T_and_rh_prof_sonde = False		# Relative humidity and T profile
plot_T_and_q_prof_hat = False
plot_T_and_q_prof_hatmirson = False
plot_T_and_q_prof_stats = False
plot_T_and_q_prof_stats_combined = False
plot_T_and_q_std_bias = True			# standard deviation and bias profile plots
plot_T_and_q_prof_stats_legs = False		# plot_T_and_q_prof_stats mean and stddev (latter as shading) over all 5 MOSAiC legs
save_figures = True
save_figures_eps = False			# save figures as vector graphics (pdf or eps)
with_titles = False				# if True, plots will have titles (False for publication plots)
rel_q_std_bias_plot_alternative = False			# plots an alternative relative bias and std dev profile; plot_T_and_q_std_bias must be True
which_retrievals = 'hus'		# which data is to be imported: 'both' contains both profiles (Temperature and humidity)
								# Other options: 'ta' or 'hus' (for temperature x.or humidity profile only).
radiosonde_version = 'level_2'			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'
mirac_version = 'mwr_pro'				# which MiRAC-P retrieval shall be used? 'mwr_pro' or 'RPG'

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
path_mirac = path_mirac[mirac_version]

# instrument status indicating which instrument is used. Not for manual editing! The stati are set automatically!
# E.g. for T profile ('ta') retrieval only, mirac is automatically disabled.
instrument_status = {'hatpro': 0,			# if 0, HATPRO data not included, if 1, HATPRO data included
						'mirac': 0,			# same for MiRAC-P
						'sonde': 0}			# same for radiosondes


# import sonde and hatpro level 2b data and mirac hum profile data (RPG retrieval):
# and create a datetime variable for plotting. There is a filter of outliers detected 'by eye'
# already included in the daterange importer for HATPRO level2b data.
# Finally, convert absolute humidity to specific humidity
hatpro_dict = import_hatpro_level2b_daterange(path_hatpro_level2, date_start, date_end, 
											which_retrieval=which_retrievals, around_radiosondes=True,
											path_radiosondes=path_radiosondes, s_version=radiosonde_version,
											mwr_avg=900, verbose=1)
instrument_status['hatpro'] = 1
if which_retrievals in ['ta', 'both']:
	hatpro_bl_dict = import_hatpro_level2c_daterange(path_hatpro_level2, date_start, date_end, 
											which_retrieval=which_retrievals, around_radiosondes=True,
											path_radiosondes=path_radiosondes, verbose=1)
# convert to specific humidity ?

if which_retrievals in ['both', 'hus'] and path_mirac:
	# load MiRAC-P humidity profile from RPG retrieval (.HPC.NC files):
	# furthermore, furthermore, there is a filter of outliers detected 'by eye' is applied in the
	# daterange importer.
	mirac_dict = import_mirac_HUA_RPG_daterange(path_mirac_level1, date_start, date_end, around_radiosondes=True,
												path_radiosondes=path_radiosondes, verbose=1)
	instrument_status['mirac'] = 1
	n_height_mirac = len(mirac_dict['height'])

# Load radiosonde data:
sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version=radiosonde_version, remove_failed=True, verbose=1)
instrument_status['sonde'] = 1
n_sondes = len(sonde_dict['launch_time'])


# Create datetime out of the MWR times:
hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
if which_retrievals in ['both', 'ta']:
	hatpro_bl_dict['time_npdt'] = hatpro_bl_dict['time'].astype("datetime64[s]")
if instrument_status['mirac']:
	mirac_dict['time_npdt'] = mirac_dict['time'].astype("datetime64[s]")
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


if instrument_status['mirac']:
	mirson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) &
					(mirac_dict['time'] <= lt+sample_time_tolerance) & (mirac_dict['flag'] == 0)).flatten() for lt in sonde_dict['launch_time']])

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

	if instrument_status['mirac']:
		mirac_dict['hua_mean_sonde'] = np.full((n_sondes,n_height_mirac), np.nan)
		mirac_dict['hua_stddev_sonde'] = np.full((n_sondes,n_height_mirac), np.nan)

		k = 0
		for mir in mirson_idx:
			mirac_dict['hua_mean_sonde'][k,:] = np.nanmean(mirac_dict['hua'][mir,:], axis=0)
			mirac_dict['hua_stddev_sonde'][k,:] = np.nanstd(mirac_dict['hua'][mir,:], axis=0)
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
			
	if instrument_status['mirac']:
		outside_eez['mirac'] = np.full((len(mirac_dict['time']),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['mirac'][(mirac_dict['time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (mirac_dict['time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False

	# theoretically, all sonde data should be outside eezs
	assert np.all(outside_eez['sonde'])


# MOSAiC Legs:
MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}




########## Plotting ##########

fs = 34		# fontsize




# dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
fs = 24		# fontsize

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P
c_RS = (1,0.435,0)			# radiosondes


if plot_option == 1: # filter EEZ time stamp
	hatpro_dict['time'] = hatpro_dict['time'][outside_eez['hatpro']]
	hatpro_dict['flag'] = hatpro_dict['flag'][outside_eez['hatpro']]
	hatpro_dict['time_npdt'] = hatpro_dict['time'][outside_eez['hatpro']]
	if which_retrievals in ['both', 'ta']:
		hatpro_dict['ta'] = hatpro_dict['ta'][outside_eez['hatpro'],:]
	if which_retrievals in ['both', 'hus']:
		hatpro_dict['hua'] = hatpro_dict['hua'][outside_eez['hatpro'],:]


if plot_T_and_q_prof_sonde:
	for k in range(n_sondes):
		if k not in which_failed:
			print(k/n_sondes)
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(10,18)

			ax.plot(sonde_dict['temp'][k,:], sonde_dict['height'][k,:], color=(0,0,0), linewidth=1.2)

			ax.set_ylim(bottom=0, top=10000)
			ax.set_xlim(left=200, right=260)

			ax.set_xlabel("T (K)", fontsize=fs-1)
			ax.set_ylabel("Height (m)", fontsize=fs-1)
			ax.set_title("MOSAiC Radiosonde temperature (T) and \n" + r"absolute humidity ($\rho_v$) profile" + " \n" +
							sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S") +
							", IWV: %.2f"%sonde_dict['iwv'][k] + "$\mathrm{kg}\,\mathrm{m}^{-2}$", fontsize=fs)


			ax.minorticks_on()
			ax.grid(which='both', axis='both')

			ax.tick_params(axis='both', labelsize=fs-3)

			# q profile:
			ax2 = ax.twiny()
			q_color = (0,0.58,1)
			ax2.plot(1000*sonde_dict['rho_v'][k,:], sonde_dict['height'][k,:], color=q_color, linewidth=1.2)

			if sonde_dict['iwv'][k] > 12:
				ax2.set_xlim(left=0, right=10)
			elif (sonde_dict['iwv'][k] <= 12) & (sonde_dict['iwv'][k] > 5):
				ax2.set_xlim(left=0, right=5)
			else:
				ax2.set_xlim(left=0, right=2.5)
			ax2.set_xlabel("$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs-1, color=q_color)
			ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
			

			ax_pos = ax.get_position().bounds
			ax_pos = ax2.get_position().bounds
			ax.set_position([ax_pos[0]+0.15*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
			ax_pos = ax2.get_position().bounds
			ax2.set_position([ax_pos[0]+0.15*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

			filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
			if save_figures:
				fig.savefig(path_plots + "Radiosonde_T_rho_v_" + filename_suffix + ".png", dpi=400)
			else:
				plt.show()
			plt.close()


if plot_T_and_q_prof_hat:
	n_time_hat = len(hatpro_dict['time'])
	for k in range(n_time_hat):
		print(k/n_time_hat)
		fig, ax = plt.subplots(1,1)
		fig.set_size_inches(10,18)

		ax.plot(hatpro_dict['ta'][k,:], hatpro_dict['height'], color=(0,0,0), linewidth=1.2)

		ax.set_ylim(bottom=0, top=10000)
		ax.set_xlim(left=210, right=290)

		ax.set_xlabel("T (K)", fontsize=fs-1)
		ax.set_ylabel("Height (m)", fontsize=fs-1)
		ax.set_title(r"MOSAiC HATPRO temperature (T) and absolute humidity ($\rho_v$) profile" + " \n" +
						hatpro_dict['datetime'][k].strftime("%Y-%m-%d %H:%M:%S"), fontsize=fs)

		ax.grid(which='major', axis='both')

		ax.tick_params(axis='both', labelsize=fs-3)

		# q profile:
		ax2 = ax.twiny()
		q_color = (0,0.58,1)
		ax2.plot(1000*hatpro_dict['hua'][k,:], hatpro_dict['height'], color=q_color, linewidth=1.2)

		# inquire max absolute humidity to set x axis limits in g m^-3:
		max_hua_hatpro = np.amax(1000*hatpro_dict['hua'][k,:])
		if max_hua_hatpro > 5:
			ax2.set_xlim(left=0, right=10)
		elif (max_hua_hatpro <= 5) & (max_hua_hatpro > 2.5):
			ax2.set_xlim(left=0, right=5)
		else:
			ax2.set_xlim(left=0, right=2.5)
		ax2.set_xlabel("$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs-1, color=q_color)
		ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
		

		ax_pos = ax.get_position().bounds
		ax_pos = ax2.get_position().bounds
		ax.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
		ax_pos = ax2.get_position().bounds
		ax2.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])

		filename_suffix = hatpro_dict['datetime'][k].strftime("%Y%m%d_%H%M%SZ")
		if save_figures: fig.savefig(path_plots + "HATPRO_T_rho_v_" + filename_suffix + ".png", dpi=400)
		plt.close()


if plot_T_and_q_prof_hatmirson:
	# only plot those profiles where the radiosonde did not burst below 10 km altitude.
	# then apply the correct time index for each mwr.
	for k in range(n_sondes):
		if these_did_not_fail[k]:

			print(k/n_sondes)
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(11,18)

			ax.plot(sonde_dict['temp'][k,:], sonde_dict['height'][k,:], color=(0,0,0), linewidth=2)
			# only plot HATPRO data, if it 
			if hatson_idx[k] >= 0:
				ax.plot(hatpro_dict['ta'][hatson_idx[k],:], hatpro_dict['height'], color=(0,0,0), linestyle='dashed', linewidth=3)

			# plot HATPRO BL scan if available for this radiosonde launch:
			if hatson_bl_idx[k] >= 0:
				ax.plot(hatpro_bl_dict['ta'][hatson_bl_idx[k],:], hatpro_bl_dict['height'], color=(0,0,0), linestyle='dotted', linewidth=3)

			ax.set_ylim(bottom=0, top=3000)			# default: 0, 10000
			ax.set_xlim(left=230, right=250)		# default: 210, 290 is default

			ax.set_xlabel("T (K)", fontsize=fs-1)
			ax.set_ylabel("Height (m)", fontsize=fs-1)

			ax.minorticks_on()
			ax.grid(which='both', axis='both')

			ax.tick_params(axis='both', labelsize=fs-3)

			# q profile:
			ax2 = ax.twiny()
			q_color = (0,0.58,1)

			# dummy plots for legend:
			ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=2, label="$\mathrm{T}_{\mathrm{RS}}$")
			# actual humidity profile plots:
			ax2.plot(1000*sonde_dict['rho_v'][k,:], sonde_dict['height'][k,:], color=q_color, linewidth=2, label="$\\rho_{v,{\mathrm{RS}}}$")

			# only plot HATPRO data, if it 
			if hatson_idx[k] >= 0:
				ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linestyle='dashed',
						label="$\mathrm{T}_{\mathrm{HATPRO}}$", linewidth=3)
				ax2.plot(1000*hatpro_dict['hua'][hatson_idx[k],:], hatpro_dict['height'], color=q_color, linestyle='dashed',
						label="$\\rho_{v,{\mathrm{HATPRO}}}$", linewidth=3)

			# another dummy line for the BL scan:
			if hatson_bl_idx[k] >= 0:
				ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linestyle='dotted',
						label="$\mathrm{T}_{\mathrm{HATPRO, BL}}$", linewidth=3)

			if mirson_idx[k] >= 0:
				ax2.plot(1000*mirac_dict['hua'][mirson_idx[k],:], mirac_dict['height'], color=q_color, linestyle='dotted',
						label="$\\rho_{v,{\mathrm{MiRAC-P}}}$", linewidth=3)


			leg_handles, leg_labels = ax2.get_legend_handles_labels()
			ax2.legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-4,
							framealpha=0.65, bbox_to_anchor=(1.0, 1.0))

			if sonde_dict['iwv'][k] > 12:
				ax2.set_xlim(left=0, right=15)				# default: 0, 10
			elif (sonde_dict['iwv'][k] <= 12) & (sonde_dict['iwv'][k] > 5):
				ax2.set_xlim(left=0, right=5)
			else:
				ax2.set_xlim(left=0, right=1.0)				# default: 0, 2.5
			ax2.set_xlabel("$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs-1, color=q_color, labelpad=6)
			ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)

			ax.set_title("MOSAiC Radiosonde (RS), HATPRO and MiRAC-P: \n" + r"Temperature (T) and absolute humidity ($\rho_v$) profile" + " \n" +
							sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S") +
							", IWV$_{\mathrm{RS}}$: %.2f"%sonde_dict['iwv'][k] + "$\mathrm{kg}\,\mathrm{m}^{-2}$", fontsize=fs-6)
			

			ax_pos = ax.get_position().bounds
			ax_pos = ax2.get_position().bounds
			ax.set_position([ax_pos[0]+0.18*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
			ax_pos = ax2.get_position().bounds
			ax2.set_position([ax_pos[0]+0.18*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

			filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
			# plt.show()
			if save_figures:
				fig.savefig(path_plots + "RS_HAT_MIR_T_rho_v_" + filename_suffix + "_largefont.png", dpi=400)
			else:
				plt.show()
			plt.close()


if plot_T_and_q_prof_stats_combined: # RMSE profile:

	# Interpolate the heights to the coarser grid: This means, that the sonde data is interpolated
	# to the retrieval grid:
	height_hatpro = hatpro_dict['height']
	n_height_hatpro = len(height_hatpro)
	# height_mirac = mirac_dict['height']
	# n_height_mirac = len(height_mirac)
	sonde_dict['height_hatpro'] = np.full((n_sondes, n_height_hatpro), np.nan)		# height on hatpro grid
	sonde_dict['temp_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# temp. on hatpro grid
	sonde_dict['rho_v_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# rho_v on hatpro grid
	# sonde_dict['rho_v_mirgrid'] = np.full((n_sondes, n_height_mirac), np.nan)		# rho_v on mirac-p grid
	for k in range(n_sondes):
		sonde_dict['height_hatpro'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['height'][k,:])
		sonde_dict['temp_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['temp'][k,:])
		sonde_dict['rho_v_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])
		# sonde_dict['rho_v_mirgrid'][k,:] = np.interp(height_mirac, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])

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
		# RMSE_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		RMSE_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'], sonde_dict['rho_v_hatgrid'], which_axis=0)

		# Compute Bias profile:
		# BIAS_rho_v_mirson = np.nanmean(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] - sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], axis=0)
		BIAS_rho_v_hatson = np.nanmean(hatpro_dict['hua_mean_sonde'] - sonde_dict['rho_v_hatgrid'], axis=0)

		# STD DEV profile: Update profiles of hatpro and mirac-p:
		# mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] -= BIAS_rho_v_mirson
		hatpro_dict['hua_mean_sonde_biascorr'] = hatpro_dict['hua_mean_sonde'] - BIAS_rho_v_hatson


		# STDDEV_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		STDDEV_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde_biascorr'], sonde_dict['rho_v_hatgrid'], which_axis=0)



	# Also compute the STDDEV for each MOSAiC leg:
	# identify the first indices of radiosonde (launch) time that lies within a calibration period:
	mosleg_idx = list()
	for key in MOSAiC_legs.keys():
		to_append = np.argwhere((sonde_dict['launch_time_dt'] >= MOSAiC_legs[key][0]) &
			(sonde_dict['launch_time_dt'] <= MOSAiC_legs[key][1])).flatten()
		# if to_append.size > 0:
		mosleg_idx.append(to_append)
	n_mosleg = len(mosleg_idx)

	# compute rmse, bias, and std dev profiles for each MOSAiC leg:
	if which_retrievals in ['both', 'ta']:
		RMSE_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		RMSE_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))

		for k, ml in enumerate(mosleg_idx):
			RMSE_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)
			RMSE_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)

			# Compute Bias profile:
			BIAS_temp_hatson_mos[k,:] = np.nanmean(hatpro_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)
			BIAS_temp_hatson_bl_mos[k,:] = np.nanmean(hatpro_bl_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)

			# STD DEV profile: Update profiles of hatpro:
			STDDEV_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)
			STDDEV_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_bl_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)

	if which_retrievals in ['both', 'hua']:
		RMSE_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))

		for k, ml in enumerate(mosleg_idx):
			RMSE_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml], sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

			# Compute Bias profile:
			BIAS_rho_v_hatson_mos[k,:] = np.nanmean(hatpro_dict['hua_mean_sonde'][ml] - sonde_dict['rho_v_hatgrid'][ml], axis=0)

			# STD DEV profile: Update profiles of hatpro:
			STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:], 
															sonde_dict['rho_v_hatgrid'][ml], which_axis=0)




	# Plotting:
	fig = plt.figure(figsize=(10,18))
	ax = plt.axes()

	# ax.plot(RMSE_temp_hatson, height_hatpro, color=(0,0,0), linewidth=3.5)
	ax.plot(STDDEV_temp_hatson, height_hatpro, color=(0,0,0), linewidth=3.5)
	# ax.plot(RMSE_temp_hatson_bl, height_hatpro, color=(0,0,0), linewidth=2.2, linestyle='dashed')
	ax.plot(STDDEV_temp_hatson_bl, height_hatpro, color=(0,0,0), linewidth=2.2, linestyle='dashed')

	ax.set_ylim(bottom=0, top=10000)
	ax.set_xlim(left=0, right=8)

	ax.set_xlabel("$\sigma_{\mathrm{T}}$ (K)", fontsize=fs-1)
	ax.set_ylabel("Height (m)", fontsize=fs-1)

	ax.minorticks_on()
	ax.grid(which='both', axis='both')

	ax.tick_params(axis='both', labelsize=fs-3)

	# q profile:
	ax2 = ax.twiny()
	q_color = (0,0.58,1)

	# dummy plots for legend:
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=3.5, label="T: Zenith scan")
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=2.2, linestyle='dashed', label="T: Boundary layer scan")

	# humidity profile plots: (*1000 to convert to g m^-3)
	# ax2.plot(1000*RMSE_rho_v_hatson, height_hatpro, color=q_color, linewidth=3.5, label="$\\rho_{v}$: Zenith scan, N = %i"%(hatson_idx[hatson_idx >= 0]))
	ax2.plot(1000*STDDEV_rho_v_hatson, height_hatpro, color=q_color, linewidth=3.5, label="$\\rho_{v}$: Zenith scan")
	# ax2.plot(1000*RMSE_rho_v_mirson, height_mirac, color=q_color, linewidth=2.2, linestyle='dashed', label="$\\rho_{v}$: MiRAC-P - RS, N = %i"%(len(mirson_idx[mirson_idx >= 0])))


	leg_handles, leg_labels = ax2.get_legend_handles_labels()
	ax2.legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-4,
					framealpha=0.65, bbox_to_anchor=(1.0, 1.0))

	ax2.set_xlim(left=0, right=1.0)
	ax2.set_xlabel("$\sigma_{\\rho_v}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs-1, color=q_color, labelpad=12)
	ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
	# ax2.set_title("RMSE between Radiosonde (RS), HATPRO and MiRAC-P: \n" + r"Temperature (T) and absolute humidity ($\rho_v$)",
				# fontsize=fs-1)
	# ax2.set_title("RMSE between Radiosondes (RS),\n" + "HATPRO and MiRAC-P:",
				# fontsize=fs-1)
	ax2.set_title("Standard deviation $\sigma$ between\nradiosondes and HATPRO",
				fontsize=fs-1)
	

	ax_pos = ax.get_position().bounds
	ax_pos = ax2.get_position().bounds
	ax.set_position([ax_pos[0]+0.32*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
	ax_pos = ax2.get_position().bounds
	ax2.set_position([ax_pos[0]+0.32*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

	if considered_period in ['mosaic', 'mwr_range']:
		filename_suffix = "_largefont"
	else:
		filename_suffix = "_" + dt.datetime.strftime(date_range_start, "%Y%m%d") + "-" + dt.datetime.strftime(date_range_end, "%Y%m%d")
	# if save_figures: fig.savefig(path_plots + "RMSE_HAT_MIR_T_rho_v" + filename_suffix + ".png", dpi=400)
	# if save_figures: fig.savefig(path_plots + "RMSE_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
	if save_figures: fig.savefig(path_plots + "STDDEV_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
	plt.close()


if plot_T_and_q_prof_stats:	# stddev profile: (no need to filter out EEZs manually because it's around radiosondes
							# which already are outside any EEZ

	# Interpolate the heights to the coarser grid: This means, that the sonde data is interpolated
	# to the retrieval grid:
	height_hatpro = hatpro_dict['height']
	n_height_hatpro = len(height_hatpro)
	# height_mirac = mirac_dict['height']
	# n_height_mirac = len(height_mirac)
	sonde_dict['height_hatpro'] = np.full((n_sondes, n_height_hatpro), np.nan)		# height on hatpro grid
	sonde_dict['temp_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# temp. on hatpro grid
	sonde_dict['rho_v_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# rho_v on hatpro grid
	# sonde_dict['rho_v_mirgrid'] = np.full((n_sondes, n_height_mirac), np.nan)		# rho_v on mirac-p grid
	for k in range(n_sondes):
		sonde_dict['height_hatpro'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['height'][k,:])
		sonde_dict['temp_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['temp'][k,:])
		sonde_dict['rho_v_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])
		# sonde_dict['rho_v_mirgrid'][k,:] = np.interp(height_mirac, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])

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
		# RMSE_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		RMSE_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'], sonde_dict['rho_v_hatgrid'], which_axis=0)

		# Compute Bias profile:
		# BIAS_rho_v_mirson = np.nanmean(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] - sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], axis=0)
		BIAS_rho_v_hatson = np.nanmean(hatpro_dict['hua_mean_sonde'] - sonde_dict['rho_v_hatgrid'], axis=0)

		# STD DEV profile: Update profiles of hatpro and mirac-p:
		# mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] -= BIAS_rho_v_mirson
		hatpro_dict['hua_mean_sonde_biascorr'] = hatpro_dict['hua_mean_sonde'] - BIAS_rho_v_hatson


		# STDDEV_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		STDDEV_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde_biascorr'], sonde_dict['rho_v_hatgrid'], which_axis=0)



	# Also compute the STDDEV for each MOSAiC leg:
	# identify the first indices of radiosonde (launch) time that lies within a calibration period:
	mosleg_idx = list()
	for key in MOSAiC_legs.keys():
		to_append = np.argwhere((sonde_dict['launch_time_dt'] >= MOSAiC_legs[key][0]) &
			(sonde_dict['launch_time_dt'] <= MOSAiC_legs[key][1])).flatten()
		# if to_append.size > 0:
		mosleg_idx.append(to_append)
	n_mosleg = len(mosleg_idx)


	# compute rmse, bias, and std dev profiles for each MOSAiC leg:
	if which_retrievals in ['both', 'ta']:
		RMSE_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		RMSE_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))

		for k, ml in enumerate(mosleg_idx):
			RMSE_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)
			RMSE_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)

			# Compute Bias profile:
			BIAS_temp_hatson_mos[k,:] = np.nanmean(hatpro_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)
			BIAS_temp_hatson_bl_mos[k,:] = np.nanmean(hatpro_bl_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)

			# STD DEV profile: Update profiles of hatpro:
			STDDEV_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)
			STDDEV_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_bl_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)

	if which_retrievals in ['both', 'hus']:
		RMSE_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		REL_STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))

		for k, ml in enumerate(mosleg_idx):
			RMSE_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml], sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

			# Compute Bias profile:
			BIAS_rho_v_hatson_mos[k,:] = np.nanmean(hatpro_dict['hua_mean_sonde'][ml] - sonde_dict['rho_v_hatgrid'][ml], axis=0)

			# STD DEV profile: Update profiles of hatpro:
			STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:], 
															sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

			# Relative standard dev. profile:
			REL_STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(((hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:]) /
																	sonde_dict['rho_v_hatgrid'][ml]), 1, which_axis=0)




	# Plotting:
	if which_retrievals in ['ta', 'both']:
		fig = plt.figure(figsize=(10,18))
		ax = plt.axes()

		if plot_T_and_q_prof_stats_legs: # plot the mean and std. dev of std dev profile over MOSAiC legs
			STDDEV_temp_hatson_mos_mean = np.nanmean(STDDEV_temp_hatson_mos, axis=0)
			STDDEV_temp_hatson_mos_std = np.nanstd(STDDEV_temp_hatson_mos, axis=0)
			STDDEV_temp_hatson_bl_mos_mean = np.nanmean(STDDEV_temp_hatson_bl_mos, axis=0)
			STDDEV_temp_hatson_bl_mos_std = np.nanstd(STDDEV_temp_hatson_bl_mos, axis=0)


			ax.plot(STDDEV_temp_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0, label="zenith scan")
			ax.plot(STDDEV_temp_hatson_bl_mos_mean, height_hatpro, color=(0,0,0), linestyle='dashed', linewidth=2.5, label='elevation scan')
			ax.fill_betweenx(height_hatpro, STDDEV_temp_hatson_mos_mean - STDDEV_temp_hatson_mos_std, 
							STDDEV_temp_hatson_mos_mean + STDDEV_temp_hatson_mos_std, facecolor=(0.2,0.2,0.8,0.5),
							linewidth=3.5, label="zenith scan")
			ax.fill_betweenx(height_hatpro, STDDEV_temp_hatson_bl_mos_mean - STDDEV_temp_hatson_bl_mos_std, 
							STDDEV_temp_hatson_bl_mos_mean + STDDEV_temp_hatson_bl_mos_std, facecolor=(0.26,0.26,0.26,0.4),
							linewidth=2.2, label="elevation scan")
		else:
			ax.plot(STDDEV_temp_hatson, height_hatpro, color=(0,0,0), linewidth=3.5, label="zenith scan")
			ax.plot(STDDEV_temp_hatson_bl, height_hatpro, color=(0,0,0), linewidth=2.2, linestyle='dashed', label="elevation scan")

		ax.set_ylim(bottom=0, top=10000)
		ax.set_xlim(left=0, right=7)

		ax.set_xlabel("$\sigma_{\mathrm{T}}$ (K)", fontsize=fs)
		ax.set_ylabel("Height (m)", fontsize=fs)
		ax.set_title("Temperature profile standard deviation ($\sigma_{\mathrm{T}}$)\nbetween radiosondes and HATPRO",
					fontsize=fs)

		ax.minorticks_on()
		ax.grid(which='both', axis='both')

		ax.tick_params(axis='both', labelsize=fs-2)

		lh, ll = ax.get_legend_handles_labels()
		ax.legend(handles=lh, labels=ll, loc="lower right", fontsize=fs)

		ax_pos = ax.get_position().bounds
		ax.set_position([ax_pos[0]+0.32*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

		if considered_period in ['mosaic', 'mwr_range']:
			filename_suffix = "hatpro_sonde_T_mosaic"
		else:
			filename_suffix = "hatpro_sonde_T_" + dt.datetime.strftime(date_range_start, "%Y%m%d") + "-" + dt.datetime.strftime(date_range_end, "%Y%m%d")
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_MIR_T_rho_v" + filename_suffix + ".png", dpi=400)
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
		if save_figures: 
			fig.savefig(path_plots + "STDDEV_" + plot_name_option + filename_suffix + ".png", dpi=400)
		else:
			plt.show()
		plt.close()



	if which_retrievals in ['hus', 'both']:
		fig = plt.figure(figsize=(10,18))
		ax = plt.axes()

		if plot_T_and_q_prof_stats_legs: # plot the mean and std. dev of std dev profile over MOSAiC legs
			STDDEV_rho_v_hatson_mos_mean = np.nanmean(STDDEV_rho_v_hatson_mos, axis=0)
			STDDEV_rho_v_hatson_mos_std = np.nanstd(STDDEV_rho_v_hatson_mos, axis=0)

			# for k in range(n_mosleg):
				# ax.plot(1000*STDDEV_rho_v_hatson_mos[k,:], height_hatpro, linewidth=2.0, label=str(k))
			ax.plot(1000*STDDEV_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0)
			ax.fill_betweenx(height_hatpro, 1000*(STDDEV_rho_v_hatson_mos_mean - STDDEV_rho_v_hatson_mos_std), 
							1000*(STDDEV_rho_v_hatson_mos_mean + STDDEV_rho_v_hatson_mos_std), facecolor=(0.2,0.2,0.8,0.5),
							linewidth=3.5)

			# plot RELATIVE standard deviation: normed by radiosonde humidity:
			sonde_dict['rho_v_hatgrid_mean'] = np.nanmean(sonde_dict['rho_v_hatgrid'], axis=0)

			ax2 = ax.twiny()
			ax2_lims = [0,1]		# relative standard deviation of humidity

			# dummy plots copying the looks of ax plots for legend:
			ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=3.0, label="$\sigma_{\\rho_{v}}$")
			ax2.fill_betweenx([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], facecolor=(0.2,0.2,0.8,0.5),
								linewidth=3.5, label="$\sigma_{\\rho_{v}}$")

			ax2.plot(STDDEV_rho_v_hatson_mos_mean / sonde_dict['rho_v_hatgrid_mean'], height_hatpro, color=(0,0,0),
						linestyle='dashed', linewidth=2.5, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			ax2.fill_betweenx(height_hatpro, (STDDEV_rho_v_hatson_mos_mean - STDDEV_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								(STDDEV_rho_v_hatson_mos_mean + STDDEV_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			# REL_STDDEV_rho_v_hatson_mos_mean = np.nanmean(REL_STDDEV_rho_v_hatson_mos, axis=0)
			# REL_STDDEV_rho_v_hatson_mos_std = np.nanstd(REL_STDDEV_rho_v_hatson_mos, axis=0)
			# ax2.plot(REL_STDDEV_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0),
						# linestyle='dashed', linewidth=2.5, label="$\sigma_{\\rho_{v}}$ / $\\rho_{v,\mathrm{RS}}$")
			# ax2.fill_betweenx(height_hatpro, REL_STDDEV_rho_v_hatson_mos_mean - REL_STDDEV_rho_v_hatson_mos_std,
								# REL_STDDEV_rho_v_hatson_mos_mean + REL_STDDEV_rho_v_hatson_mos_std,
								# facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\sigma_{\\rho_{v}}$ / $\\rho_{v,\mathrm{RS}}$")

			ax2.set_ylim(bottom=0, top=10000)
			ax2.set_xlim(left=ax2_lims[0], right=ax2_lims[1])

			ax2.set_xlabel("$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{Radiosonde}}$", fontsize=fs, labelpad=15)

			ax2.tick_params(axis='both', labelsize=fs-2)

			lh, ll = ax2.get_legend_handles_labels()
			ax2.legend(handles=lh, labels=ll, loc="center right", fontsize=fs)

			ax2_pos = ax2.get_position().bounds
			ax2.set_position([ax2_pos[0]+0.32*ax2_pos[0], ax2_pos[1], 0.95*ax2_pos[2], ax2_pos[3]*0.95])

		else:
			ax.plot(STDDEV_rho_v_hatson, height_hatpro, color=(0,0,0), linewidth=3.5)



		ax.set_ylim(bottom=0, top=10000)
		ax.set_xlim(left=0, right=1)

		ax.set_xlabel("$\sigma_{\\rho_{v}}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs)

		ax.set_ylabel("Height (m)", fontsize=fs)
		ax.set_title("Humidity profile standard deviation ($\sigma_{\\rho_v}$)\nbetween radiosondes (RS) and HATPRO",
					fontsize=fs, pad=15)

		ax.minorticks_on()
		ax.grid(which='both', axis='both')

		ax.tick_params(axis='both', labelsize=fs-2)
		ax.tick_params(axis='x', pad=7)

		ax_pos = ax.get_position().bounds
		ax.set_position([ax_pos[0]+0.32*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

		if considered_period in ['mosaic', 'mwr_range']:
			filename_suffix = "hatpro_sonde_rho_v_mosaic"
		else:
			filename_suffix = "hatpro_sonde_rho_v_" + dt.datetime.strftime(date_range_start, "%Y%m%d") + "-" + dt.datetime.strftime(date_range_end, "%Y%m%d")
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_MIR_T_rho_v" + filename_suffix + ".png", dpi=400)
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
		if save_figures: 
			fig.savefig(path_plots + "STDDEV_" + plot_name_option + filename_suffix + ".png", dpi=400)
		else:
			plt.show()
		plt.close()


if plot_T_and_q_std_bias: # standard deviation and bias profiles (as subplots):


	# Interpolate the heights to the coarser grid: This means, that the sonde data is interpolated
	# to the retrieval grid:
	height_hatpro = hatpro_dict['height']
	n_height_hatpro = len(height_hatpro)
	# height_mirac = mirac_dict['height']
	# n_height_mirac = len(height_mirac)
	sonde_dict['height_hatpro'] = np.full((n_sondes, n_height_hatpro), np.nan)		# height on hatpro grid
	sonde_dict['temp_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# temp. on hatpro grid
	sonde_dict['rho_v_hatgrid'] = np.full((n_sondes, n_height_hatpro), np.nan)		# rho_v on hatpro grid
	# sonde_dict['rho_v_mirgrid'] = np.full((n_sondes, n_height_mirac), np.nan)		# rho_v on mirac-p grid
	for k in range(n_sondes):
		sonde_dict['height_hatpro'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['height'][k,:])
		sonde_dict['temp_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['temp'][k,:])
		sonde_dict['rho_v_hatgrid'][k,:] = np.interp(height_hatpro, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])
		# sonde_dict['rho_v_mirgrid'][k,:] = np.interp(height_mirac, sonde_dict['height'][k,:], sonde_dict['rho_v'][k,:])

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
		# RMSE_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		RMSE_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'], sonde_dict['rho_v_hatgrid'], which_axis=0)

		# Compute Bias profile:
		# BIAS_rho_v_mirson = np.nanmean(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] - sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], axis=0)
		BIAS_rho_v_hatson = np.nanmean(hatpro_dict['hua_mean_sonde'] - sonde_dict['rho_v_hatgrid'], axis=0)

		# STD DEV profile: Update profiles of hatpro and mirac-p:
		# mirac_dict['hua'][mirson_idx[mirson_idx >= 0]] -= BIAS_rho_v_mirson
		hatpro_dict['hua_mean_sonde_biascorr'] = hatpro_dict['hua_mean_sonde'] - BIAS_rho_v_hatson


		# STDDEV_rho_v_mirson = compute_RMSE_profile(mirac_dict['hua'][mirson_idx[mirson_idx >= 0]],
												# sonde_dict['rho_v_mirgrid'][these_did_not_fail_mirson], which_axis=0)
		STDDEV_rho_v_hatson = compute_RMSE_profile(hatpro_dict['hua_mean_sonde_biascorr'], sonde_dict['rho_v_hatgrid'], which_axis=0)



	# Also compute the STDDEV for each MOSAiC leg:
	# identify the first indices of radiosonde (launch) time that lies within a calibration period:
	mosleg_idx = list()
	for key in MOSAiC_legs.keys():
		to_append = np.argwhere((sonde_dict['launch_time_dt'] >= MOSAiC_legs[key][0]) &
			(sonde_dict['launch_time_dt'] <= MOSAiC_legs[key][1])).flatten()
		# if to_append.size > 0:
		mosleg_idx.append(to_append)
	n_mosleg = len(mosleg_idx)


	# compute rmse, bias, and std dev profiles for each MOSAiC leg:
	if which_retrievals in ['both', 'ta']:
		RMSE_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		RMSE_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		BIAS_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
		STDDEV_temp_hatson_bl_mos = np.zeros((n_mosleg, n_height_hatpro))

		for k, ml in enumerate(mosleg_idx):
			RMSE_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)
			RMSE_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml], sonde_dict['temp_hatgrid'][ml], which_axis=0)

			# Compute Bias profile:
			BIAS_temp_hatson_mos[k,:] = np.nanmean(hatpro_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)
			BIAS_temp_hatson_bl_mos[k,:] = np.nanmean(hatpro_bl_dict['ta_mean_sonde'][ml] - sonde_dict['temp_hatgrid'][ml], axis=0)

			# STD DEV profile: Update profiles of hatpro:
			STDDEV_temp_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)
			STDDEV_temp_hatson_bl_mos[k,:] = compute_RMSE_profile(hatpro_bl_dict['ta_mean_sonde'][ml] - BIAS_temp_hatson_bl_mos[k,:], 
															sonde_dict['temp_hatgrid'][ml], which_axis=0)

	if which_retrievals in ['both', 'hus']:

		if rel_q_std_bias_plot_alternative:
			RMSE_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			BIAS_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			REL_BIAS_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			REL_STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))

			for k, ml in enumerate(mosleg_idx):
				RMSE_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml], sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

				# Compute Bias profile:
				BIAS_rho_v_hatson_mos[k,:] = np.nanmean(hatpro_dict['hua_mean_sonde'][ml] - sonde_dict['rho_v_hatgrid'][ml], axis=0)

				# Compute relative bias profile:
				REL_BIAS_rho_v_hatson_mos[k,:] = np.nanmean((hatpro_dict['hua_mean_sonde'][ml] - sonde_dict['rho_v_hatgrid'][ml]) / 
															sonde_dict['rho_v_hatgrid'][ml], axis=0)

				# STD DEV profile: Update profiles of hatpro:
				STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:], 
																sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

				# Relative standard dev. profile:
				REL_STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(((hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:]) /
																		sonde_dict['rho_v_hatgrid'][ml]), 1, which_axis=0)

		else:
			RMSE_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			BIAS_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))
			STDDEV_rho_v_hatson_mos = np.zeros((n_mosleg, n_height_hatpro))

			for k, ml in enumerate(mosleg_idx):
				RMSE_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml], sonde_dict['rho_v_hatgrid'][ml], which_axis=0)

				# Compute Bias profile:
				BIAS_rho_v_hatson_mos[k,:] = np.nanmean(hatpro_dict['hua_mean_sonde'][ml] - sonde_dict['rho_v_hatgrid'][ml], axis=0)

				# STD DEV profile: Update profiles of hatpro:
				STDDEV_rho_v_hatson_mos[k,:] = compute_RMSE_profile(hatpro_dict['hua_mean_sonde'][ml] - BIAS_rho_v_hatson_mos[k,:], 
																sonde_dict['rho_v_hatgrid'][ml], which_axis=0)



	# Plotting:
	if which_retrievals in ['ta', 'both']:

		fig = plt.figure(figsize=(20,18))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile
	

		STDDEV_temp_hatson_mos_mean = np.nanmean(STDDEV_temp_hatson_mos, axis=0)
		STDDEV_temp_hatson_mos_std = np.nanstd(STDDEV_temp_hatson_mos, axis=0)
		STDDEV_temp_hatson_bl_mos_mean = np.nanmean(STDDEV_temp_hatson_bl_mos, axis=0)
		STDDEV_temp_hatson_bl_mos_std = np.nanstd(STDDEV_temp_hatson_bl_mos, axis=0)

		BIAS_temp_hatson_mos_mean = np.nanmean(BIAS_temp_hatson_mos, axis=0)
		BIAS_temp_hatson_mos_std = np.nanstd(BIAS_temp_hatson_mos, axis=0)
		BIAS_temp_hatson_bl_mos_mean = np.nanmean(BIAS_temp_hatson_bl_mos, axis=0)
		BIAS_temp_hatson_bl_mos_std = np.nanstd(BIAS_temp_hatson_bl_mos, axis=0)		


		# bias profiles:
		ax_bias.plot(BIAS_temp_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0, label="zenith scan")
		ax_bias.plot(BIAS_temp_hatson_bl_mos_mean, height_hatpro, color=(0,0,0), linestyle='dashed', linewidth=2.5, label='elevation scan')
		ax_bias.plot(np.full_like(height_hatpro, 0.0), height_hatpro, color=(0,0,0), linewidth=1.0)
		ax_bias.fill_betweenx(height_hatpro, BIAS_temp_hatson_mos_mean - BIAS_temp_hatson_mos_std, 
						BIAS_temp_hatson_mos_mean + BIAS_temp_hatson_mos_std, facecolor=(0.2,0.2,0.8,0.5),
						linewidth=3.5, label="zenith scan")
		ax_bias.fill_betweenx(height_hatpro, BIAS_temp_hatson_bl_mos_mean - BIAS_temp_hatson_bl_mos_std, 
						BIAS_temp_hatson_bl_mos_mean + BIAS_temp_hatson_bl_mos_std, facecolor=(0.26,0.26,0.26,0.4),
						linewidth=2.2, label="elevation scan")


		# std dev profiles:
		ax_std.plot(STDDEV_temp_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0, label="zenith scan")
		ax_std.plot(STDDEV_temp_hatson_bl_mos_mean, height_hatpro, color=(0,0,0), linestyle='dashed', linewidth=2.5, label='elevation scan')
		ax_std.fill_betweenx(height_hatpro, STDDEV_temp_hatson_mos_mean - STDDEV_temp_hatson_mos_std, 
						STDDEV_temp_hatson_mos_mean + STDDEV_temp_hatson_mos_std, facecolor=(0.2,0.2,0.8,0.5),
						linewidth=3.5, label="zenith scan")
		ax_std.fill_betweenx(height_hatpro, STDDEV_temp_hatson_bl_mos_mean - STDDEV_temp_hatson_bl_mos_std, 
						STDDEV_temp_hatson_bl_mos_mean + STDDEV_temp_hatson_bl_mos_std, facecolor=(0.26,0.26,0.26,0.4),
						linewidth=2.2, label="elevation scan")


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

		if considered_period in ['mosaic', 'mwr_range']:
			filename_suffix = "hatpro_sonde_T_mosaic"
		else:
			filename_suffix = "hatpro_sonde_T_" + dt.datetime.strftime(date_range_start, "%Y%m%d") + "-" + dt.datetime.strftime(date_range_end, "%Y%m%d")
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_MIR_T_rho_v" + filename_suffix + ".png", dpi=400)
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
		if save_figures: 
			fig.savefig(path_plots + "STDDEV_and_bias_" + plot_name_option + filename_suffix + ".png", dpi=400, bbox_inches='tight')
		elif save_figures_eps:
			fig.savefig(path_plots + "STDDEV_and_bias_" + plot_name_option + filename_suffix + ".pdf", bbox_inches='tight')
		else:
			plt.show()
		plt.close()



	if which_retrievals in ['hus', 'both']:
		fig = plt.figure(figsize=(20,18))
		ax_bias = plt.subplot2grid((1,2), (0,0))			# bias profile
		ax_std = plt.subplot2grid((1,2), (0,1))				# std dev profile

		y_lims = [0, 8000]			# in m
		ax2_bias_lims = [-60,60]		# relative bias (-60 to 60 %)

		STDDEV_rho_v_hatson_mos_mean = np.nanmean(STDDEV_rho_v_hatson_mos, axis=0)
		STDDEV_rho_v_hatson_mos_std = np.nanstd(STDDEV_rho_v_hatson_mos, axis=0)

		BIAS_rho_v_hatson_mos_mean = np.nanmean(BIAS_rho_v_hatson_mos, axis=0)
		BIAS_rho_v_hatson_mos_std = np.nanstd(BIAS_rho_v_hatson_mos, axis=0)

		# alternative relative bias and std dev:
		if rel_q_std_bias_plot_alternative:
			REL_STDDEV_rho_v_hatson_mos_mean = np.nanmean(REL_STDDEV_rho_v_hatson_mos, axis=0)
			REL_STDDEV_rho_v_hatson_mos_std = np.nanstd(REL_STDDEV_rho_v_hatson_mos, axis=0)
			REL_BIAS_rho_v_hatson_mos_mean = np.nanmean(REL_BIAS_rho_v_hatson_mos, axis=0)
			REL_BIAS_rho_v_hatson_mos_std = np.nanstd(REL_BIAS_rho_v_hatson_mos, axis=0)


		# bias profiles:
		ax_bias.plot(1000*BIAS_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0)
		ax_bias.plot(np.full_like(height_hatpro, 0.0), height_hatpro, color=(0,0,0), linewidth=1.0)
		ax_bias.fill_betweenx(height_hatpro, 1000*(BIAS_rho_v_hatson_mos_mean - BIAS_rho_v_hatson_mos_std), 
						1000*(BIAS_rho_v_hatson_mos_mean + BIAS_rho_v_hatson_mos_std), facecolor=(0.2,0.2,0.8,0.5),
						linewidth=3.5)

		# plot RELATIVE standard deviation: normed by radiosonde humidity:
		sonde_dict['rho_v_hatgrid_mean'] = np.nanmean(sonde_dict['rho_v_hatgrid'], axis=0)

		# plot RELATIVE bias: normed by radiosonde humidity:
		ax2_bias = ax_bias.twiny()

		# dummy plots copying the looks of ax_bias plots for legend:
		ax2_bias.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=3.0, label="$\Delta\\rho_{v}$")
		ax2_bias.fill_betweenx([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], facecolor=(0.2,0.2,0.8,0.5),
							linewidth=3.5, label="$\Delta\\rho_{v}$")

		if rel_q_std_bias_plot_alternative:
			ax2_bias.plot(100*REL_BIAS_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0),
						linestyle='dashed', linewidth=2.5, label="$\Delta\\rho_{v}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			ax2_bias.fill_betweenx(height_hatpro, 100*(REL_BIAS_rho_v_hatson_mos_mean - REL_BIAS_rho_v_hatson_mos_std),
								100*(REL_BIAS_rho_v_hatson_mos_mean + REL_BIAS_rho_v_hatson_mos_std),
								facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\Delta\\rho_{v}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
		else:
			ax2_bias.plot(100*(BIAS_rho_v_hatson_mos_mean / sonde_dict['rho_v_hatgrid_mean']), height_hatpro, color=(0,0,0),
						linestyle='dashed', linewidth=2.5, label="$\Delta\\rho_{v}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			ax2_bias.fill_betweenx(height_hatpro, 100*(BIAS_rho_v_hatson_mos_mean - BIAS_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								100*(BIAS_rho_v_hatson_mos_mean + BIAS_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\Delta\\rho_{v}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")

		
		# std dev profiles:
		ax_std.plot(1000*STDDEV_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0), linewidth=3.0)
		ax_std.fill_betweenx(height_hatpro, 1000*(STDDEV_rho_v_hatson_mos_mean - STDDEV_rho_v_hatson_mos_std), 
						1000*(STDDEV_rho_v_hatson_mos_mean + STDDEV_rho_v_hatson_mos_std), facecolor=(0.2,0.2,0.8,0.5),
						linewidth=3.5)


		ax2 = ax_std.twiny()

		# dummy plots copying the looks of ax_std plots for legend:
		ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=3.0, label="$\sigma_{\\rho_{v}}$")
		ax2.fill_betweenx([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], facecolor=(0.2,0.2,0.8,0.5),
							linewidth=3.5, label="$\sigma_{\\rho_{v}}$")

		if rel_q_std_bias_plot_alternative:
			ax2.plot(100*REL_STDDEV_rho_v_hatson_mos_mean, height_hatpro, color=(0,0,0),
						linestyle='dashed', linewidth=2.5, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			ax2.fill_betweenx(height_hatpro, 100*(REL_STDDEV_rho_v_hatson_mos_mean - REL_STDDEV_rho_v_hatson_mos_std),
								100*(REL_STDDEV_rho_v_hatson_mos_mean + REL_STDDEV_rho_v_hatson_mos_std),
								facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")

			ax2_lims = [0,150]		# relative standard deviation of humidity

		else:
			ax2.plot(100*(STDDEV_rho_v_hatson_mos_mean / sonde_dict['rho_v_hatgrid_mean']), height_hatpro, color=(0,0,0),
						linestyle='dashed', linewidth=2.5, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")
			ax2.fill_betweenx(height_hatpro, 100*(STDDEV_rho_v_hatson_mos_mean - STDDEV_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								100*(STDDEV_rho_v_hatson_mos_mean + STDDEV_rho_v_hatson_mos_std) / sonde_dict['rho_v_hatgrid_mean'],
								facecolor=(0.26,0.26,0.26,0.4), linewidth=2.2, label="$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{RS}}$")

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
		ax_bias.set_xlim(left=-0.6, right=0.6)
		ax_bias.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax2_bias.set_xlim(left=ax2_bias_lims[0], right=ax2_bias_lims[1])
		ax2_bias.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax_std.set_xlim(left=0, right=1)
		ax_std.set_ylim(bottom=y_lims[0], top=y_lims[1])
		ax2.set_xlim(left=ax2_lims[0], right=ax2_lims[1])
		ax2.set_ylim(bottom=y_lims[0], top=y_lims[1])

		# labels:
		ax_bias.set_xlabel("$\\rho_{v,\mathrm{HATPRO}} - \\rho_{v,\mathrm{Radiosonde}}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs)
		ax2_bias.set_xlabel("$\\left( \\rho_{v,\mathrm{HATPRO}} - \\rho_{v,\mathrm{Radiosonde}} \\right)$ / $\overline{\\rho}_{v,\mathrm{Radiosonde}}$ ($\%$)", 
							fontsize=fs, labelpad=15)
		ax_bias.set_ylabel("Height (m)", fontsize=fs)
		ax_std.set_xlabel("$\sigma_{\\rho_{v}}$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs)
		ax2.set_xlabel("$\sigma_{\\rho_{v}}$ / $\overline{\\rho}_{v,\mathrm{Radiosonde}}$ ($\%$)", fontsize=fs, labelpad=15)
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
		ax_bias.tick_params(axis='x', pad=7)
		ax_std.tick_params(axis='both', labelsize=fs-2)
		ax_std.tick_params(axis='x', pad=7)
		ax2.tick_params(axis='both', labelsize=fs-2)
		ax2_bias.tick_params(axis='both', labelsize=fs-2)
		ax_std.yaxis.set_ticklabels([])

		# limit axis spacing:
		plt.subplots_adjust(wspace=0.0)			# removes space between subplots

		# adjust axis positions:
		ax_pos = ax_bias.get_position().bounds
		ax_bias.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
		ax_pos = ax_std.get_position().bounds
		ax_std.set_position([ax_pos[0]+0.05*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])


		if considered_period in ['mosaic', 'mwr_range']:
			filename_suffix = "hatpro_sonde_rho_v_mosaic"
		else:
			filename_suffix = "hatpro_sonde_rho_v_" + dt.datetime.strftime(date_range_start, "%Y%m%d") + "-" + dt.datetime.strftime(date_range_end, "%Y%m%d")
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_MIR_T_rho_v" + filename_suffix + ".png", dpi=400)
		# if save_figures: fig.savefig(path_plots + "RMSE_HAT_T_rho_v" + filename_suffix + ".png", dpi=400)
		if save_figures: 
			fig.savefig(path_plots + "STDDEV_and_bias_" + plot_name_option + filename_suffix + ".png", dpi=400, bbox_inches='tight')
		elif save_figures_eps:
			fig.savefig(path_plots + "STDDEV_and_bias_" + plot_name_option + filename_suffix + ".pdf", bbox_inches='tight')
		else:
			plt.show()
		plt.close()


if plot_T_and_rh_prof_sonde:
	for k in range(n_sondes):
		if k not in which_failed:
			print(k/n_sondes)
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(10,18)

			ax.plot(sonde_dict['temp'][k,:], sonde_dict['height'][k,:], color=(0,0,0), linewidth=1.2)

			ax.set_ylim(bottom=0, top=10000)
			ax.set_xlim(left=215, right=290)

			ax.set_xlabel("T (K)", fontsize=fs-1)
			ax.set_ylabel("Height (m)", fontsize=fs-1)
			ax.set_title("MOSAiC Radiosonde temperature (T) and \n" + "relative humidity (RH) profile" + " \n" +
							sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S") +
							", IWV: %.2f"%sonde_dict['iwv'][k] + "$\mathrm{kg}\,\mathrm{m}^{-2}$", fontsize=fs)


			ax.minorticks_on()
			ax.grid(which='both', axis='both')

			ax.tick_params(axis='both', labelsize=fs-3)

			# q profile:
			ax2 = ax.twiny()
			q_color = (0,0.58,1)
			ax2.plot(100*sonde_dict['rh'][k,:], sonde_dict['height'][k,:], color=q_color, linewidth=1.2)

			ax2.set_xlim(left=0, right=100)
			ax2.set_xlabel("RH (%)", fontsize=fs-1, color=q_color)
			ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
			

			ax_pos = ax.get_position().bounds
			ax_pos = ax2.get_position().bounds
			ax.set_position([ax_pos[0]+0.15*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])
			ax_pos = ax2.get_position().bounds
			ax2.set_position([ax_pos[0]+0.15*ax_pos[0], ax_pos[1], 0.95*ax_pos[2], ax_pos[3]*0.95])

			filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
			if save_figures:
				fig.savefig(path_plots + "Radiosonde_T_rh_" + filename_suffix + ".png", dpi=400)
			else:
				plt.show()
			plt.close()

pdb.set_trace()