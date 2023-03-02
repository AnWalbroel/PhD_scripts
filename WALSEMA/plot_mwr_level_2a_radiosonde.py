import numpy as np
import datetime as dt
import copy
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import sys

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
import warnings
from import_data import *
from met_tools import *
from data_tools import *
from scipy import stats
import xarray as xr
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


################################################################

if len(sys.argv) == 1:
	sys.argv.append("")
elif (len(sys.argv) == 2) and (sys.argv[1] not in ["fig1", "fig2", "fig3"]):
	raise ValueError("plot_mwr_level_2a_radiosonde.py must be called with one of the following options: " +
					"'python3 plot_mwr_level_2a_radiosonde.py', 'python3 plot_mwr_level_2a_radiosonde.py " +
					'"fig1"' + "', 'python3 plot_mwr_level_2a_radiosonde.py " + '"fig2"' + "', " +
					"'python3 plot_mwr_level_2a_radiosonde.py " + '"fig3"' + "'.")

# Paths:
path_data = {'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l2/",		# path of hatpro derived products
				'mirac-p': "/data/obs/campaigns/WALSEMA/atm/mirac-p/l2/",	# path of mirac-p derived products
				'radiosondes': {'raw': "/data/radiosondes/Polarstern/PS131_ATWAICE_upper_air_soundings/"}}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/mwr_level_2a_radiosonde/"


# Select one of the following plot_options:		###################################################
# 0: Unfiltered: Each data point is plotted, outliers are not filtered!
# 1: Omit flagged values:  Each data point is plotted, outliers are left out.
# 2: Running mean and omit flagged values: Running mean over rm_window width, outliers filtered.
# 3: Master time axis: Data points are on a master time axis instead of each instrument 
#		having its own time axis -> must be used for IWV difference between different MWRs, outliers 
#		are always filtered on master time axis.
# 4: Master time axis with running mean: Data, to which a running mean is applied, are on master 
#		time axis.
# 5: Same as 4 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range.
# 6: Same as 2 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range. (recommended)!!
plot_option = 6 		# default: plot_option = 6
considered_period = 'walsema'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2022-07-07 - 2022-08-12
									# 'walsema': entire walsema campaign (2022-06-28 - 2022-08-12)
									# 'user': user defined
if sys.argv[1] == "fig1":
	plot_IWV_time_series = True				# simple IWV time series
else:
	plot_IWV_time_series = False			# simple IWV time series
if sys.argv[1] == "fig2":
	plot_IWV_scatterplots_all = True		# scatterplot of IWV: HATPRO vs. sonde, MiRAC-P vs. sonde for entire period
else:
	plot_IWV_scatterplots_all = False		# scatterplot of IWV: HATPRO vs. sonde, MiRAC-P vs. sonde for entire period
rm_window = 300					# in seconds! default: 300
save_figures = False
save_figures_eps = True		# save figures as vector graphics (pdf or eps)
skip_pre_calib = False			# if True, periods before first decent calibration is skipped; MAY ONLY BE TRUE if plot_IWV_scatterplots_calib = True !!!
scatplot_fix_axlims = True		# if True, axis limits will be hardcoded to [0, 35] or depend on MOSAiC leg; if False: axis limits
								# depend on max value of the scatterplot (recommended: True)
with_titles = False				# if True, plots will have titles (False for publication plots)

if sys.argv[1] in ['fig1', 'fig2']:
	which_retrievals = 'iwv'		# which data is to be imported: 'both' contains both integrated quantities like IWV
									# and LWP. options: IWV only: 'prw' or 'iwv', LWP only: 'clwvi' or 'lwp'.
elif sys.argv[1] == 'fig3':
	which_retrievals = 'both'		# which data is to be imported: 'both' contains both integrated quantities like IWV
									# and LWP. options: IWV only: 'prw' or 'iwv', LWP only: 'clwvi' or 'lwp'.
else:
	which_retrievals = 'iwv'		# which data is to be imported: 'both' contains both integrated quantities like IWV
									# and LWP. options: IWV only: 'prw' or 'iwv', LWP only: 'clwvi' or 'lwp'.

radiosonde_version = 'raw'			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'
mirac_version = 'v01'					# version of outout: currently available: "v01"
plot_LWP_time_series = False			# simple LWP time series
plot_LWP_time_series_daily_avg = False	# LWP time series of daily averages (HATPRO only)
if sys.argv[1] == 'fig3':
	plot_LWP_IWV_time_series_daily_avg = True	# LWP and IWV time series of HATPRO that might also include data avail plot
	include_data_availability = True	# if true in some time series plots the data availability is plotted (currently, 
										# only true when plot_LWP_IWV_time_series_daily_avg True)
else:
	plot_LWP_IWV_time_series_daily_avg = False	# LWP and IWV time series of HATPRO that might also include data avail plot
	include_data_availability = False	# if true in some time series plots the data availability is plotted (currently, 
										# only true when plot_LWP_IWV_time_series_daily_avg True)


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2022-07-07", "2022-08-12"],
					'walsema': ["2022-06-28", "2022-08-12"],
					'user': ["2020-04-13", "2020-04-23"]}
date_start = daterange_options[considered_period][0]
date_end = daterange_options[considered_period][1]


# plot name options:
# There are the following options for iwv_name_option:
# "unfiltered": 			each data point is plotted, outliers are not filtered, 
# "flag0": 					each data point is plotted, outliers are left out,
# "flag0_rmdt%i"%rm_window: running mean over rm_window width, outliers filtered,
# "master_time": 			data points master time axis instead of each instrument having its own time axis
#							-> must be used for IWV difference, outliers are always filtered on master time axis
# "master_time_rmdt%i"%rm_window: data points master time axis and running mean has been applied
# "flag0_rmdt{rm_window}_noEEZ": running mean over rm_window width, outliers filtered, EEZ takes out
iwv_name_options = ["unfiltered", "flag0", f"flag0_rmdt{rm_window}", 
					"master_time", f"master_time_rmdt{rm_window}",
					f"master_time_rmdt{rm_window}_noEEZ",
					f"flag0_rmdt{rm_window}_noEEZ"]

# choose paths:
path_radiosondes = path_data['radiosondes'][radiosonde_version] # choose radiosonde path

# if do_running_mean is True, running mean of the data will be computed or imported (if applicable):
if plot_option in [2, 4, 5, 6]:
	if plot_IWV_scatterplots_all:
		do_running_mean = False
	else:
		do_running_mean = True

elif plot_option in [0, 1, 3]:
	do_running_mean = False
else:
	raise ValueError("'plot_option' must be 0, 1, 2, 3, 4, 5 or 6.")


# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

# avoid skip_pre_calib being True if any other plot is used:
if skip_pre_calib and not plot_IWV_scatterplots_calib:
	skip_pre_calib = False

# instrument status indicating which instrument is used. Not for manual editing! The stati are set automatically!
# E.g. for LWP retrieval only, sonde and mirac are automatically disabled.
instrument_status = {'hatpro': 0,			# if 0, HATPRO data not included, if 1, HATPRO data included
						'mirac': 0,			# same for MiRAC-P
						'sonde': 0}			# same for radiosondes


# LOAD RADIOMETER DATA

# import sonde and hatpro level 2a data and mirac IWV data:
# import_hatpro_level2a_daterange is for the data structure separated into daily folders:
print("Loading HATPRO Level 2a....")
hatpro_dict = import_hatpro_level2a_daterange(path_data['hatpro'], date_start, date_end, 
											which_retrieval=which_retrievals, minute_avg=False, vers='v00', campaign='walsema', verbose=1)
instrument_status['hatpro'] = 1

# load MiRAC-P IWV values:
if which_retrievals in ['iwv', 'prw', 'both']:
	print("Loading MiRAC-P Level 2a....")
	mirac_dict = import_mirac_level2a_daterange(path_data['mirac-p'], date_start, date_end, which_retrieval=which_retrievals, 
												vers=mirac_version, minute_avg=False, verbose=1)
	instrument_status['mirac'] = 1
	mirac_dict['clwvi'] = np.full(mirac_dict['prw'].shape, np.nan)

else:
	mirac_dict = {'time': np.array([]), 'flag': np.array([]), 'clwvi': np.array([])}
	# status stays 0


if which_retrievals in ['iwv', 'prw', 'both'] and plot_IWV_time_series:

	if do_running_mean:
		# HATPRO:
		# If the running mean has not yet been performed and the resulting data saved to a netcdf file, the following
		# functions must be called: running_mean_datetime from data_tools.py:
		print("HATPRO running mean")
		hatpro_dict['prw'] = running_mean_datetime(hatpro_dict['prw'], rm_window, hatpro_dict['time'])

		# MiRAC-P:
		# If the running mean has not yet been performed and the resulting data saved to a netcdf file, the following
		# functions must be called: running_mean_datetime and save_IWV_running_mean from data_tools.py:
		print("MiRAC-P running mean")
		mirac_dict['prw'] = running_mean_datetime(mirac_dict['prw'], rm_window, mirac_dict['time'])


# Create datetime out of the MWR times:
if instrument_status['hatpro']:
	hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
if instrument_status['mirac']:
	mirac_dict['time_npdt'] = mirac_dict['time'].astype("datetime64[s]")


# Import Radiosonde data if IWV or both variables are asked:
if which_retrievals in ['iwv', 'prw', 'both']:
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


# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")

# calibration times of HATPRO: manually entered from MWR logbook
if skip_pre_calib:
	calibration_times_HATPRO = [dt.datetime(2022,7,30,6,25)]
	calibration_times_MiRAC = [dt.datetime(2022,7,30,7,14)]
else:
	calibration_times_HATPRO = [dt.datetime(2022,7,7,9,53), 
								dt.datetime(2022,7,30,6,25)]
	calibration_times_MiRAC = [dt.datetime(2022,7,7,10,20), dt.datetime(2022,7,30,7,14)]

n_calib_HATPRO = len(calibration_times_HATPRO)
n_calib_MiRAC = len(calibration_times_MiRAC)


# If it is desired to plot the IWV difference between one of the microwave radiometers (mwr) and the 
# radiosonde:
# First, find indices when mwr specific time (or master time) equals a sonde launch time. Then average
# over sonde launchtime:launchtime + 15 minutes and compute standard deviation as well.
if plot_IWV_scatterplots_all:
	# no absolute value search because we want the closest mwr time after radiosonde launch!
	launch_window = 1800		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged

	if plot_option in [0, 1, 2, 6]:	# find instrument specific time indices:

		hatson_idx = np.asarray([np.argwhere((hatpro_dict['time'] >= lt) & 
						(hatpro_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])
		mirson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) &
						(mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

		hatpro_dict['prw_mean_sonde'] = np.full((n_sondes,), np.nan)
		hatpro_dict['prw_stddev_sonde'] = np.full((n_sondes,), np.nan)
		mirac_dict['prw_mean_sonde'] = np.full((n_sondes,), np.nan)
		mirac_dict['prw_stddev_sonde'] = np.full((n_sondes,), np.nan)
		k = 0
		for hat, mir in zip(hatson_idx, mirson_idx):
			hatpro_dict['prw_mean_sonde'][k] = np.nanmean(hatpro_dict['prw'][hat])
			hatpro_dict['prw_stddev_sonde'][k] = np.nanstd(hatpro_dict['prw'][hat])
			mirac_dict['prw_mean_sonde'][k] = np.nanmean(mirac_dict['prw'][mir])
			mirac_dict['prw_stddev_sonde'][k] = np.nanstd(mirac_dict['prw'][mir])
			k = k + 1


import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
dt_fmt = mdates.DateFormatter("%d %b") # (e.g. "Feb 23")
datetick_auto = False
fs = 24		# fontsize

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P
c_RS = (1,0.435,0)			# radiosondes

# create x_ticks depending on the date range: roughly 20 x_ticks are planned
# round off to days if number of days > 15:
date_range_delta = (date_range_end - date_range_start)
if date_range_delta < dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=6)
	dt_fmt = mdates.DateFormatter("%d %b %H:%M")
elif date_range_delta < dt.timedelta(days=11) and date_range_delta >= dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=12)
	dt_fmt = mdates.DateFormatter("%d %b %H:%M")
elif date_range_delta >= dt.timedelta(days=11) and date_range_delta < dt.timedelta(21):
	x_tick_delta = dt.timedelta(days=1)
else:
	x_tick_delta = dt.timedelta(days=3)

x_ticks_dt = mdates.drange(date_range_start, date_range_end, x_tick_delta)	# alternative if the xticklabel is centered


########## Plotting ##########

if plot_IWV_time_series:

	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig1, ax1 = plt.subplots(1,2)
	fig1.set_size_inches(22,10)

	axlim = [0, 40]			# axis limits for IWV plot in kg m^-2


	SONDE_IWV_plot = ax1[0].plot(sonde_dict['launch_time_dt'], sonde_dict['IWV'], linestyle='none', marker='.', linewidth=0.5,
									markersize=15.0, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='Radiosonde')

	HATPRO_IWV_plot = ax1[0].plot(hatpro_dict['time_npdt'], hatpro_dict['prw'],
									color=c_H, linewidth=1.2, label='HATPRO')
	MIRAC_IWV_plot = ax1[0].plot(mirac_dict['time_npdt'], mirac_dict['prw'],
									color=c_M, linewidth=1.2, label='MiRAC-P')


	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)

	# legend + dummy lines for thicker lines in the legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper right', fontsize=fs,
					framealpha=1.0, markerscale=2.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=dt.datetime(2022,6,28,0), right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles:
		the_title = ax1[0].set_title("Integrated Water Vapour (IWV) during WALSEMA (" + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
							" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")", fontsize=fs, pad=50)
		the_title.set_position((0.5, 1.2))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig1.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%Y-%m-%d, %H:%M UTC"))

	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linestyle='dashed', linewidth=2,
				label="$\\bf{MiRAC-P}$")
	for ct_mirac in calibration_times_MiRAC: 
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig1.get_facecolor(),
				label=dt.datetime.strftime(ct_mirac, "%Y-%m-%d, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs)
	lo.get_title().set_fontweight('bold')

	iwv_name_base = "WALSEMA_hatpro_mirac-p_radiosonde_IWV_time_series"
	iwv_name_suffix_def = ""
	if considered_period == 'user':
		iwv_name_suffix_def = "_" + date_start.replace("-","") + "-" + date_end.replace("-","")

	if save_figures:
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')
	elif save_figures_eps:
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_IWV_scatterplots_all:
	# IWV scatterplot for entire MOSAiC period: first for HATPRO:
	fig6 = plt.figure(figsize=(10,20))
	ax1 = plt.subplot2grid((2,1), (0,0))			# HATPRO
	ax2 = plt.subplot2grid((2,1), (1,0))			# MiRAC-P
	
	# handle axis limits:
	if scatplot_fix_axlims:
		axlim = np.asarray([0, 35])

	else:
		axlim = [0, np.ceil(np.nanmax(np.concatenate((sonde_dict['IWV'],
							hatpro_dict['prw_mean_sonde'], mirac_dict['prw_mean_sonde']), axis=0))+2)]

	# Filter for times after first calibration:
	first_calib_hat = datetime_to_epochtime(calibration_times_HATPRO[0])
	first_calib_mir = datetime_to_epochtime(calibration_times_MiRAC[0])
	idx_post_calib = np.where(sonde_dict['launch_time'] > first_calib_hat)[0]
	sonde_dict['IWV'] = sonde_dict['IWV'][idx_post_calib]
	hatpro_dict['prw_mean_sonde'] = hatpro_dict['prw_mean_sonde'][idx_post_calib]
	hatpro_dict['prw_stddev_sonde'] = hatpro_dict['prw_stddev_sonde'][idx_post_calib]
	mirac_dict['prw_mean_sonde'] = mirac_dict['prw_mean_sonde'][idx_post_calib]
	mirac_dict['prw_stddev_sonde'] = mirac_dict['prw_stddev_sonde'][idx_post_calib]

	# compute retrieval statistics:
	ret_stat_dict_hat = compute_retrieval_statistics(sonde_dict['IWV'],
													hatpro_dict['prw_mean_sonde'],
													compute_stddev=True)
	ret_stat_dict_mir = compute_retrieval_statistics(sonde_dict['IWV'],
													mirac_dict['prw_mean_sonde'],
													compute_stddev=True)


	# plotting:
	ax1.errorbar(sonde_dict['IWV'], hatpro_dict['prw_mean_sonde'], 
						yerr=hatpro_dict['prw_stddev_sonde'],
						ecolor=c_H, elinewidth=1.6, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='HATPRO')
	ax2.errorbar(sonde_dict['IWV'], mirac_dict['prw_mean_sonde'], 
						yerr=mirac_dict['prw_stddev_sonde'],
						ecolor=c_M, elinewidth=1.6, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='MiRAC-P')


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	nonnan_hatson = np.argwhere(~np.isnan(hatpro_dict['prw_mean_sonde']) &
						~np.isnan(sonde_dict['IWV'])).flatten()
	y_fit = hatpro_dict['prw_mean_sonde'][nonnan_hatson]
	x_fit = sonde_dict['IWV'][nonnan_hatson]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
		m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
		a = m_fit[0]
		b = m_fit[1]

		ds_fit = ax1.plot(axlim, a*axlim + b, color=c_H, linewidth=1.2, label="Best fit")

	# plot a line for orientation which would represent a perfect fit:
	ax1.plot(axlim, axlim, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")

	nonnan_mirson = np.argwhere(~np.isnan(mirac_dict['prw_mean_sonde']) &
						~np.isnan(sonde_dict['IWV'])).flatten()
	y_fit = mirac_dict['prw_mean_sonde'][nonnan_mirson]
	x_fit = sonde_dict['IWV'][nonnan_mirson]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
		m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
		a = m_fit[0]
		b = m_fit[1]

		ds_fit = ax2.plot(axlim, a*axlim + b, color=c_M, linewidth=1.2, label="Best fit")

	# plot a line for orientation which would represent a perfect fit:
	ax2.plot(axlim, axlim, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")


	# add figure identifier of subplots: a), b), ...
	ax1.text(0.02, 0.98, "a)", color=(0,0,0), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax1.transAxes)
	ax2.text(0.02, 0.98, "b)", color=(0,0,0), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax2.transAxes)


	# add statistics:
	ax1.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(ret_stat_dict_hat['N'], 
			np.nanmean(np.concatenate((sonde_dict['IWV'], hatpro_dict['prw_mean_sonde']), axis=0)),
			ret_stat_dict_hat['bias'], ret_stat_dict_hat['rmse'], ret_stat_dict_hat['stddev'], ret_stat_dict_hat['R']),
			horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fs-2)
	ax2.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(ret_stat_dict_mir['N'], 
			np.nanmean(np.concatenate((sonde_dict['IWV'], mirac_dict['prw_mean_sonde']), axis=0)),
			ret_stat_dict_mir['bias'], ret_stat_dict_mir['rmse'], ret_stat_dict_mir['stddev'], ret_stat_dict_mir['R']),
			horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, fontsize=fs-2)


	# Legends:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs,
				framealpha=0.5)
	leg_handles, leg_labels = ax2.get_legend_handles_labels()
	ax2.legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs,
				framealpha=0.5)


	# set axis limits:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax2.set_ylim(bottom=axlim[0], top=axlim[1])

	ax1.set_xlim(left=axlim[0], right=axlim[1])
	ax2.set_xlim(left=axlim[0], right=axlim[1])


	# set axis ticks, ticklabels and tick parameters:
	ax1.minorticks_on()
	ax2.minorticks_on()

	ax1.tick_params(axis='both', labelsize=fs-2)
	ax2.tick_params(axis='both', labelsize=fs-2)


	# aspect ratio:
	ax1.set_aspect('equal')
	ax2.set_aspect('equal')


	# grid:
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	ax2.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


	# labels:
	ax1.set_ylabel("IWV$_{\mathrm{HATPRO}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles: ax1.set_title("IWV comparison of HATPRO and\nradiosonde during WALSEMA", fontsize=fs)

	ax2.set_ylabel("IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax2.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles: ax2.set_title("IWV comparison of MiRAC-P and\nradiosonde during WALSEMA", fontsize=fs)


	# Limit axis spacing:
	plt.subplots_adjust(hspace=0.15)			# removes space between subplots


	iwv_name_base = "WALSEMA_hatpro_mirac-p_radiosonde_IWV_scatterplot"
	if scatplot_fix_axlims:
		iwv_name_suffix_def = "_fixax"
	else:
		iwv_name_suffix_def = ""
	if save_figures:
		fig6.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')

	elif save_figures_eps:
		fig6.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_LWP_IWV_time_series_daily_avg and plot_option == 6:
	# only consider steps outside EEZs
	hatpro_clwvi = xr.DataArray(hatpro_dict['clwvi'], 
								coords=[hatpro_dict['time_npdt']], 
								dims=["time"])
	hatpro_clwvi_daily = hatpro_clwvi#.resample(time="1D").mean()

	hatpro_prw = xr.DataArray(hatpro_dict['prw'],
								coords=[hatpro_dict['time_npdt']],
								dims=['time'])
	hatpro_prw = hatpro_prw.resample(time="1D").mean()


	# LWP time series of daily averages of HATPRO
	fig1 = plt.figure(figsize=(16,8))
	ax1 = plt.axes()
	
	axlim = [-25, 750]		# axis limits for LWP plot in g m^-2
	ax2lim = [0, 40]		# axis limits for IWV plot in kg m-2


	HATPRO_LWP_plot = ax1.plot(hatpro_clwvi_daily.time.values, 1000*hatpro_clwvi_daily.values,
									color=(0,0,0), linewidth=1.5)

	ax2 = ax1.twinx()

	# dummy plot of LWP for legend
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=2.0, label="LWP")
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, linestyle='dashed', label="IWV")

	HATPRO_IWV_plot = ax2.plot(hatpro_prw.time.values, hatpro_prw.values, color=c_H, linewidth=1.5,
								linestyle='dashed', zorder=-1)

	lh, ll = ax1.get_legend_handles_labels()
	ax2.legend(handles=lh, labels=ll, loc="upper left", fontsize=fs)


	# set axis limits:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax1.set_xlim(left=dt.datetime(2022,6,28,0,0), right=date_range_end)
	ax1.xaxis.set_major_formatter(dt_fmt)
	ax2.set_ylim(bottom=ax2lim[0], top=ax2lim[1])

	# labels:
	ax1.set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax2.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, color=c_H)
	if with_titles:
		the_title = ax1.set_title(("Daily average of Liquid Water Path (LWP) and Integrated Water Vapour (IWV)\nduring WALSEMA (" + 
							dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
							" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")"), fontsize=fs, pad=40)
		the_title.set_position((0.5, 1.2))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1.set_xticks(x_ticks_dt)
		ax1.tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	ax2.yaxis.set_ticks(ax2.get_yticks()[1:])
	ytick_labels = [yt for yt in ax2.get_yticks()]
	ax2.yaxis.set_ticklabels(ytick_labels)

	ax1.tick_params(axis='y', labelsize=fs-2)
	ax2.tick_params(axis='y', labelcolor=c_H, labelsize=fs-2)

	ax1.grid(which='major', axis='both')

	ax1_pos = ax1.get_position().bounds
	ax1.set_position([ax1_pos[0], ax1_pos[1]+0.05*ax1_pos[3], ax1_pos[2], ax1_pos[3]*0.9])

	ax1_pos = ax2.get_position().bounds
	ax2.set_position([ax1_pos[0], ax1_pos[1]+0.05*ax1_pos[3], ax1_pos[2], ax1_pos[3]*0.9])


	iwv_name_base = "WALSEMA_hatpro_LWP_IWV_daily_mean_time_series"
	iwv_name_suffix_def = ""
	if considered_period == 'user':
		iwv_name_suffix_def = "_" + date_start.replace("-","") + "-" + date_end.replace("-","")

	if save_figures:
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')

	elif save_figures_eps:
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()

