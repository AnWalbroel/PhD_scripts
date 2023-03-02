import numpy as np
import datetime as dt
import copy
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
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


# Paths:
path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"
path_mirac = {'RPG': "/data/obs/campaigns/mosaic/mirac-p/l1/",
				'mwr_pro': "/data/obs/campaigns/mosaic/mirac-p/l2/"}
path_arm = "/data/obs/campaigns/mosaic/arm/mwrret/"
path_radiosondes = {'level_2': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/",
					'mossonde': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/",
					'psYYMMDDwHH': "/data/testbed/datasets/MOSAiC/rs41/"}
path_plots = {'default': "/net/blanc/awalbroe/Plots/MOSAiC_mwr_sondes_IWV/",
				'publication': "/net/blanc/awalbroe/Plots/figures_data_paper/"}
MiRAC_outlier_file = "/net/blanc/awalbroe/Codes/MOSAiC/MiRAC-P_outliers.txt"
path_save_concat_IWV = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/"
path_IWV_JR = "/net/blanc/awalbroe/Data/satellite_JR/"


# Select one of the following plot_options:		###################################################
# 0: Unfiltered: Each data point is plotted, outliers are not filtered!
# 1: Omit flagged values:  Each data point is plotted, outliers are left out.
# 2: Running mean and omit flagged values: Running mean over rm_window width, outliers filtered.
# 3: Master time axis: Data points are on a master time axis instead of each instrument 
#		having its own time axis -> must be used for IWV difference, outliers are always 
#		filtered on master time axis.
# 4: Master time axis with running mean: Data, to which a running mean is applied, are on master 
#		time axis.
# 5: Same as 4 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range.
# 6: Same as 2 but the times, when Polarstern was within Exclusive Economic Zones are filtered out
#		because we are not permitted to publish data in that range.
plot_option = 6 		# for plot_option in range(0,5):			# default: plot_option = 0
considered_period = 'mosaic'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-10-22 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# 'user': user defined
plot_IWV_time_series = False			# simple IWV time series
plot_IWV_diff_hat_mir = False		# IWV difference: HATPRO - MiRAC-P
plot_IWV_diff_hat_mir_rel = False	# IWV difference: HATPRO - MiRAC-P normed by the mean of the two IWV
plot_IWV_diff_hat_arm = False		# IWV difference: HATPRO - ARM
plot_IWV_diff_hatson_mirson = False		# IWV difference: HATPRO - sonde, MiRAC-P - sonde
plot_IWV_diff_hatson_mirson_rel = False		# IWV difference: HATPRO - sonde, MiRAC-P - sonde -> normed by the mean of the two IWV
plot_IWV_diff_mwrson_sonde = False			# y axis: IWV difference = IWV_mwr - IWV_sonde, x axis: IWV_sonde
plot_IWV_scatterplots_calib = False			# scatterplots of IWV: HATPRO vs. sonde; MiRAC-P vs. sonde on calibration periods
plot_IWV_scatterplots_legs = False			# scatterplots of IWV: HATPRO vs. sonde; MiRAC-P vs. sonde on MOSAiC legs
plot_IWV_scatterplots_all = True			# scatterplot of IWV: HATPRO vs. sonde, MiRAC-P vs. sonde for entire period
plot_IWV_overview_boxplot = False			# Boxplot of IWV statistics for entire MOSAiC period \EEZs (plot_option = 5 only!)
plot_IWV_mwrson_diff_histogram = False		# histogram of IWV difference (MWR - sonde)
plot_IWV_histogram = False					# histogram showing the frequency occurrence of IWV (plot_option = 5 only!)
include_JR_IWV = False					# option to include Janna Rueckert's Optimal Estimation (OE) IWV from satellite data
rm_window = 300					# in seconds! default: 300
save_figures = False
save_figures_eps = False		# save figures as vector graphics (pdf or eps)
skip_pre_calib = False			# if True, periods before first decent calibration is skipped; MAY ONLY BE TRUE if plot_IWV_scatterplots_calib = True !!!
scatplot_fix_axlims = True		# if True, axis limits will be hardcoded to [0, 35] or depend on MOSAiC leg; if False: axis limits
								# depend on max value of the scatterplot
with_titles = False				# if True, plots will have titles (False for publication plots)
include_data_availability = False	# if true in some time series plots the data availability is plotted (currently, 
									# only true when plot_LWP_IWV_time_series_daily_avg True)
which_retrievals = 'iwv'		# which data is to be imported: 'both' contains both integrated quantities like IWV
								# and LWP. Other options: 'prw' or 'clwvi' ('iwv' or 'lwp').
plot_output = "default"		# defines which directory to choose for plot output: options: 'default', 'publication' (see path_plots)
radiosonde_version = 'level_2'			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'
mirac_version = 'mwr_pro'				# which MiRAC-P retrieval shall be used? 'mwr_pro' or 'RPG'
mirac_version_version = 'v01'			# version of mwr_pro outout: currently available: "v01"; only used if mirac_version == 'mwr_pro'
plot_LWP_time_series = False			# simple LWP time series
plot_LWP_time_series_daily_avg = False	# LWP time series of daily averages (HATPRO only)
plot_LWP_IWV_time_series_daily_avg = False	# LWP and IWV time series of HATPRO that might also include data avail plot
plot_LWP_overview_boxplot = False		# Boxplot of LWP statistics for entire MOSAiC period \EEZs (plot_option = 5 only!)


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2019-10-22", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-04-13", "2020-04-23"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"


# Choose whether you want to have all data points or minute averages (saves memory):
# Instead, the option to apply a running mean over a specified number of seconds
# is offered. Recommended: False and use running mean instead because it takes 
# a lot of computation time (test run: 40 minutes all together) if set True and
# no benefit (smoother) compared to a running mean.
minute_average = False

# plot name options:
# There are the following options for iwv_name_option:
# "unfiltered": 			each data point is plotted, outliers are not filtered, 
# "flag0": 					each data point is plotted, outliers are left out,
# "flag0_rmdt%i"%rm_window: running mean over rm_window width, outliers filtered,
# "master_time": 			data points master time axis instead of each instrument having its own time axis
#							-> must be used for IWV difference, outliers are always filtered on master time axis
# "master_time_rmdt%i"%rm_window: data points master time axis and running mean has been applied
iwv_name_options = ["unfiltered", "flag0", f"flag0_rmdt{rm_window}", 
					"master_time", f"master_time_rmdt{rm_window}",
					f"master_time_rmdt{rm_window}_noEEZ",
					f"flag0_rmdt{rm_window}_noEEZ"]
iwv_name_option = iwv_name_options[plot_option]

# choose paths:
path_plots = path_plots[plot_output]					# choose plot path
path_radiosondes = path_radiosondes[radiosonde_version] # choose radiosonde path
path_mirac = path_mirac[mirac_version]					# choose mirac path

# if do_running_mean is True, running mean of the data will be computed or imported (if applicable):
if plot_option in [2, 4, 5, 6]:
	if plot_IWV_scatterplots_calib or plot_IWV_scatterplots_legs or plot_IWV_scatterplots_all:
		do_running_mean = False
	else:
		do_running_mean = True
	
	# these filenames are with considered period = mosaic and will be used to load IWV and LWP for the specified
	# daterange
	IWV_concat_hatpro_filename = "IWV_MOSAiC_HATPRO_master_running_mean_mosaic_v00.nc"
	if mirac_version == 'RPG':
		IWV_concat_mirac_filename = "IWV_MOSAiC_MiRAC-P_master_running_mean_mosaic_v00.nc"
	elif mirac_version == 'mwr_pro':
		IWV_concat_mirac_filename = "IWV_MOSAiC_MiRAC-P_%s_master_running_mean_mosaic_v00.nc"%mirac_version_version
	IWV_concat_arm_filename = "IWV_MOSAiC_ARM_master_running_mean_mosaic_v00.nc"

	IWV_rm_hatpro_filename = "IWV_MOSAiC_HATPRO_running_mean_mosaic_v00.nc"
	if mirac_version == 'RPG':
		IWV_rm_mirac_filename = "IWV_MOSAiC_MiRAC-P_running_mean_mosaic_v00.nc"
	elif mirac_version == 'mwr_pro':
		IWV_rm_mirac_filename = "IWV_MOSAiC_MiRAC-P_%s_running_mean_mosaic_v00.nc"%mirac_version_version
	IWV_rm_arm_filename = "IWV_MOSAiC_ARM_running_mean_mosaic_v00.nc"

	# LWP_concat_hatpro_filename = "LWP_MOSAiC_HATPRO_master_running_mean_mosaic_v00.nc"
	# LWP_concat_mirac_filename = "LWP_MOSAiC_MiRAC-P_master_running_mean_mosaic_v00.nc"		# for LWP, we remain with the RPG version
	# LWP_concat_arm_filename = "LWP_MOSAiC_ARM_master_running_mean_mosaic_v00.nc"
	# LWP_rm_hatpro_filename = "LWP_MOSAiC_HATPRO_running_mean_mosaic_v00.nc"		# has been deactivated on 2021-12-10
	# LWP_rm_mirac_filename = "LWP_MOSAiC_MiRAC-P_running_mean_mosaic_v00.nc"		# for LWP, we remain with the RPG version
	# LWP_rm_arm_filename = "LWP_MOSAiC_ARM_running_mean_mosaic_v00.nc"

elif plot_option in [0, 1, 3]:
	do_running_mean = False

	# these filenames are with considered period = mosaic and will be used to load IWV and LWP for the specified
	# daterange
	IWV_concat_hatpro_filename = "IWV_MOSAiC_HATPRO_master_mosaic_v00.nc"
	if mirac_version == 'RPG':
		IWV_concat_mirac_filename = "IWV_MOSAiC_MiRAC-P_master_mosaic_v00.nc"
	elif mirac_version == 'mwr_pro':
		IWV_concat_mirac_filename = "IWV_MOSAiC_MiRAC-P_%s_master_mosaic_v00.nc"%mirac_version_version
	IWV_concat_arm_filename = "IWV_MOSAiC_ARM_master_mosaic_v00.nc"

	LWP_concat_hatpro_filename = "LWP_MOSAiC_HATPRO_master_mosaic_v00.nc"
	LWP_concat_mirac_filename = "LWP_MOSAiC_MiRAC-P_master_mosaic_v00.nc"		# for LWP, we remain with the RPG version
	LWP_concat_arm_filename = "LWP_MOSAiC_ARM_master_mosaic_v00.nc"
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
# E.g. for LWP retrieval only, radiosonde and JR_sat are automatically disabled.
instrument_status = {'hatpro': 0,			# if 0, HATPRO data not included, if 1, HATPRO data included
						'mirac': 0,			# same for MiRAC-P
						'arm': 0,			# same for ARM MWR
						'sonde': 0,			# same for radiosondes
						'JR_sat': 0}		# same for Janna Rueckert's OE IWV from satellite data



# Only need to import the MWR and radiosonde files via daterange importer if no precomputed
# and concatenated data is available. For plot_options >= 3, the daterange importer should
# no longer be relevant.
# Check if all precomputed files exist:
if plot_option in [3,4,5]:
	all_precomputed_files_exist = np.all(np.asarray([os.path.exists(path_save_concat_IWV + IWV_concat_hatpro_filename),
												os.path.exists(path_save_concat_IWV + IWV_rm_hatpro_filename),
												os.path.exists(path_save_concat_IWV + IWV_concat_mirac_filename),
												os.path.exists(path_save_concat_IWV + IWV_rm_mirac_filename),
												os.path.exists(path_save_concat_IWV + IWV_concat_arm_filename),
												os.path.exists(path_save_concat_IWV + IWV_rm_arm_filename),
												os.path.exists(path_save_concat_IWV + LWP_concat_hatpro_filename),
												os.path.exists(path_save_concat_IWV + LWP_rm_hatpro_filename),
												os.path.exists(path_save_concat_IWV + LWP_concat_mirac_filename),
												os.path.exists(path_save_concat_IWV + LWP_rm_mirac_filename),
												os.path.exists(path_save_concat_IWV + LWP_concat_arm_filename),
												os.path.exists(path_save_concat_IWV + LWP_rm_arm_filename)]))
elif plot_option in [2,6]:
	check_files = list()
	if which_retrievals in ['iwv', 'prw', 'both']:
		check_files += [os.path.exists(path_save_concat_IWV + IWV_rm_hatpro_filename),
						os.path.exists(path_save_concat_IWV + IWV_rm_mirac_filename)]
	# elif which_retrievals in ['lwp', 'clwvi', 'both']:								# deactivated on 2021-12-10
		# check_files += [os.path.exists(path_save_concat_IWV + LWP_rm_hatpro_filename),
						# os.path.exists(path_save_concat_IWV + LWP_rm_mirac_filename)]

	all_precomputed_files_exist = np.all(np.asarray(check_files))
# If precomputed files don't exist yet, it is required to load in the data via daterange importers.
# This is also necessary if plot_option 0, 1 or 2 is selected.
if (plot_option < 3) or (plot_option == 6) or (not all_precomputed_files_exist):

	# import sonde and hatpro level 2a data and mirac IWV data:
	# and create a datetime variable for plotting and apply running mean, if specified:
	hatpro_dict = import_hatpro_level2a_daterange(path_hatpro_level2, date_start, date_end, 
												which_retrieval=which_retrievals, minute_avg=minute_average, verbose=1)
	instrument_status['hatpro'] = 1

	# load MiRAC-P IWV values from RPG retrieval (.IWV.NC files):
	# furthermore, flag RPG IWV, if abs((IWV(i+1) - IWV(i))/1 second) > 0.3 kg m^-2 s^-1
	# and apply running mean, if specified
	# furthermore, there is a filter of outliers detected 'by eye' (2021-05-14):
	if mirac_version == 'RPG':
		mirac_dict = import_mirac_IWV_LWP_RPG_daterange(path_mirac, date_start, date_end, which_retrieval=which_retrievals,
						minute_avg=minute_average, verbose=1)
		if which_retrievals in ['both', 'iwv', 'prw']:
			mirac_iwv_outlier_idx = np.argwhere(np.abs((mirac_dict['IWV'][1:] - mirac_dict['IWV'][:-1]) / 
										(mirac_dict['time'][1:] - mirac_dict['time'][:-1])) > 0.3).flatten()
			mirac_dict['RF'][mirac_iwv_outlier_idx] = 2.0
		mirac_dict['RF'] = outliers_per_eye(mirac_dict['RF'], mirac_dict['time'], instrument='mirac', filename=MiRAC_outlier_file)

		# Renaming RPG convention to MWR_PRO convention:
		mirac_dict['flag'] = mirac_dict['RF']
		mirac_dict['prw'] = mirac_dict['IWV']
		del mirac_dict['RF'], mirac_dict['IWV']
		instrument_status['mirac'] = 1

	elif mirac_version == 'mwr_pro':
		if which_retrievals in ['iwv', 'prw', 'both']:
			mirac_dict = import_mirac_level2a_daterange(path_mirac, date_start, date_end, which_retrieval=which_retrievals, 
														vers=mirac_version_version, minute_avg=minute_average, verbose=1)
			instrument_status['mirac'] = 1
		else:
			mirac_dict = {'time': np.array([]), 'flag': np.array([]), 'clwvi': np.array([])}


	# Import ARM data:
	# # # # # arm_dict = import_arm_def_daterange(path_arm, date_start, date_end, which_retrieval=which_retrievals, verbose=1)
	# # # # # instrument_status['arm'] = 1

	if (considered_period not in ['mosaic', 'mwr_range']) and plot_option in [0,1,2,6]:
		# Create datetime out of the MWR times directly after importing:
		# Unfortunately, it is neccessary to do this twice because the time axis saved in the previously imported
		# files doesn't always start at 00:00:00 of a day (although there are measurements for this) but (e.g. for
		# mirac-p) frequently starts at 00:00:02, while the missing two seconds are saved to the wrong file
		# (namely that of the previous day, which is, of course, not considered in the daterange importer).
		hatpro_dict['time0'] = hatpro_dict['time']
		mirac_dict['time0'] = mirac_dict['time']
		# # # # # arm_dict['time0'] = arm_dict['time']


	if which_retrievals in ['iwv', 'prw', 'both']:


		if do_running_mean:
			# HATPRO:
			# If the running mean has not yet been performed and the resulting data saved to a netcdf file, the following
			# functions must be called: running_mean_datetime and save_IWV_running_mean from data_tools.py:
			print("HATPRO running mean")
			if (not os.path.exists(path_save_concat_IWV + IWV_rm_hatpro_filename)) and (considered_period in ['mosaic']):
				hatpro_dict['prw'][hatpro_dict['flag']==0] = running_mean_datetime(hatpro_dict['prw'][hatpro_dict['flag']==0], 
																rm_window, hatpro_dict['time'][hatpro_dict['flag']==0])
				save_IWV_running_mean(path_save_concat_IWV + IWV_rm_hatpro_filename, hatpro_dict, rm_window, 'hatpro')
			else:
				# If running means has already been performed and the netcdf files containing the averaged IWV data exist,
				# simply import them:
				rm_window, hatpro_dict['prw'], hatpro_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + IWV_rm_hatpro_filename,
													date_start, date_end, instrument='hatpro')

			# MiRAC-P:
			# If the running mean has not yet been performed and the resulting data saved to a netcdf file, the following
			# functions must be called: running_mean_datetime and save_IWV_running_mean from data_tools.py:
			print("MiRAC-P running mean")
			if (not os.path.exists(path_save_concat_IWV + IWV_rm_mirac_filename)) and (considered_period in ['mosaic']):
				mirac_dict['prw'][mirac_dict['flag']==0] = running_mean_datetime(mirac_dict['prw'][mirac_dict['flag']==0], 
															rm_window, mirac_dict['time'][mirac_dict['flag']==0])
				save_IWV_running_mean(path_save_concat_IWV + IWV_rm_mirac_filename, mirac_dict, rm_window, 'mirac')
			else:
				# If running means has already been performed and the netcdf files containing the averaged IWV data exist,
				# simply import them:
				rm_window, mirac_dict['prw'], mirac_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + IWV_rm_mirac_filename,
													date_start, date_end, instrument='mirac')

			# ARM:
			# If the running mean has not yet been performed and the resulting data saved to a netcdf file, the following
			# functions must be called: running_mean_datetime and save_IWV_running_mean from data_tools.py:
			print("ARM running mean")
			if (not os.path.exists(path_save_concat_IWV + IWV_rm_arm_filename)) and (considered_period in ['mosaic']) and instrument_status['arm']:
				arm_dict['prw'][arm_dict['prw_flag']==0] = running_mean_datetime(arm_dict['prw'][arm_dict['prw_flag']==0], 
															rm_window, arm_dict['time'][arm_dict['prw_flag']==0])
				save_IWV_running_mean(path_save_concat_IWV + IWV_rm_arm_filename, arm_dict, rm_window, 'arm')
			elif instrument_status['arm']:
				# If running means has already been performed and the netcdf files containing the averaged IWV data exist,
				# simply import them:
				rm_window, arm_dict['prw'], arm_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + IWV_rm_arm_filename,
													date_start, date_end, instrument='arm')

	
	"""															# deactivated on 2021-12-10: NO RUNNING MEAN FOR LWP
	elif which_retrievals in ['lwp', 'clwvi', 'both']:


		if do_running_mean:
			# HATPRO:
			print("HATPRO running mean")
			if (not os.path.exists(path_save_concat_IWV + LWP_rm_hatpro_filename)) and (considered_period in ['mosaic']):
				hatpro_dict['clwvi'][hatpro_dict['flag']==0] = running_mean_datetime(hatpro_dict['clwvi'][hatpro_dict['flag']==0],
																rm_window, hatpro_dict['time'][hatpro_dict['flag']==0])
				save_LWP_running_mean(path_save_concat_IWV + LWP_rm_hatpro_filename, hatpro_dict, rm_window, 'hatpro')
			else:
				# If running means has already been performed and the netcdf files containing the averaged LWP data exist,
				# simply import them:
				rm_window, hatpro_dict['clwvi'], hatpro_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + LWP_rm_hatpro_filename,
													date_start, date_end, instrument='hatpro')

			# MiRAC-P:
			print("MiRAC-P running mean")
			if (not os.path.exists(path_save_concat_IWV + LWP_rm_mirac_filename)) and (considered_period in ['mosaic']) and (mirac_version == 'RPG') and (instrument_status['mirac']):
				mirac_dict['clwvi'][mirac_dict['flag']==0] = running_mean_datetime(mirac_dict['clwvi'][mirac_dict['flag']==0],
																rm_window, mirac_dict['time'][mirac_dict['flag']==0])
				save_LWP_running_mean(path_save_concat_IWV + LWP_rm_mirac_filename, mirac_dict, rm_window, 'mirac')
			elif instrument_status['mirac']:
				# If running means has already been performed and the netcdf files containing the averaged LWP data exist,
				# simply import them:
				rm_window, mirac_dict['clwvi'], mirac_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + LWP_rm_mirac_filename,
													date_start, date_end, instrument='mirac')

			# ARM:
			print("ARM running mean")
			if (not os.path.exists(path_save_concat_IWV + LWP_rm_arm_filename)) and (considered_period in ['mosaic']) and instrument_status['arm']:
				arm_dict['lwp'][arm_dict['lwp_flag']==0] = running_mean_datetime(arm_dict['lwp'][arm_dict['lwp_flag']==0], 
															rm_window, arm_dict['time'][arm_dict['lwp_flag']==0])
				save_LWP_running_mean(path_save_concat_IWV + LWP_rm_arm_filename, arm_dict, rm_window, 'arm')
			elif instrument_status['arm']:
				# If running means has already been performed and the netcdf files containing the averaged IWV data exist,
				# simply import them:
				rm_window, arm_dict['lwp'], arm_dict['time'] = import_concat_IWV_LWP_mwr_running_mean(path_save_concat_IWV + LWP_rm_arm_filename,
													date_start, date_end, instrument='arm')
	"""


	# Create datetime out of the MWR times:
	if instrument_status['hatpro']:
		hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
	if instrument_status['mirac']:
		mirac_dict['time_npdt'] = mirac_dict['time'].astype("datetime64[s]")
	if instrument_status['arm']:
		arm_dict['time_npdt'] = arm_dict['time'].astype("datetime64[s]")

# Import Radiosonde data if IWV or both variables are asked:
if which_retrievals in ['iwv', 'prw', 'both']:
	# Load radiosonde data and compute IWV:
	sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version=radiosonde_version, remove_failed=True, verbose=1)
	n_sondes = len(sonde_dict['launch_time'])
	sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])
	if radiosonde_version in ['mossonde', 'psYYMMDDwHH']:	# then IWV has to be computed:
		sonde_dict['iwv'] = np.asarray([compute_IWV_q(sonde_dict['q'][k,:], sonde_dict['pres'][k,:]) for k in range(n_sondes)])
	instrument_status['sonde'] = 1
	
	# import (preliminary) OE result data from Janna Rueckert if desired:
	if plot_option == 5 and include_JR_IWV:
		IWV_JR_dict = import_IWV_OE_JR(path_IWV_JR + "oem_mosaic_mean.csv", date_start, date_end)
		IWV_JR_DS = xr.open_dataset(path_IWV_JR + "oem_mean_mar_apr.nc")
		time_dt_jr_temporary = np.array([dt.datetime.strptime(tttt, "%Y-%m-%d %H:%M:%S") for tttt in IWV_JR_DS.time.values])
		IWV_JR_DS['time'] = xr.DataArray(datetime_to_epochtime(time_dt_jr_temporary), dims=['time'])
		instrument_status['JR_sat'] = 1


# Master time axis:
if plot_option in [3, 4, 5]:

	# initialise dictionaries for the MWRs in case they do not exist yet
	try: hatpro_dict
	except NameError: hatpro_dict = {}
	try: mirac_dict
	except NameError: mirac_dict = {}
	try: arm_dict
	except NameError: arm_dict = {}


	if which_retrievals in ['both', 'iwv', 'prw']:
		# If netcdf files of IWV on a master time axis (entire MOSAiC period with 1 second spacing) have not been
		# created, the following function must be called to create them (long computation time if the chosen
		# period is longer than 2 months!!!):

		if not os.path.exists(path_save_concat_IWV + IWV_concat_hatpro_filename):
			print("Master time axis: HATPRO")
			create_IWV_concat_master_time(date_start, date_end, hatpro_dict, 'hatpro',
											path_save_concat_IWV, IWV_concat_hatpro_filename)

		if not os.path.exists(path_save_concat_IWV + IWV_concat_mirac_filename):
			print("Master time axis: MiRAC-P")
			create_IWV_concat_master_time(date_start, date_end, mirac_dict, 'mirac',
											path_save_concat_IWV, IWV_concat_mirac_filename)

		if not os.path.exists(path_save_concat_IWV + IWV_concat_arm_filename):
			print("Master time axis: ARM")
			create_IWV_concat_master_time(date_start, date_end, arm_dict, 'arm',
											path_save_concat_IWV, IWV_concat_arm_filename)
		
		# If the above mentioned function has already been performed and the netcdf file of IWV on master
		# time axis exists, it suffices to import the IWV (and time axis):
		master_time, hatpro_dict['prw_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + IWV_concat_hatpro_filename,
													date_start, date_end)
		master_time, mirac_dict['prw_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + IWV_concat_mirac_filename,
													date_start, date_end)
		master_time, arm_dict['prw_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + IWV_concat_arm_filename,
													date_start, date_end)

	# Same for LWP:
	if which_retrievals in ['both', 'lwp', 'clwvi']:
		# If netcdf files of LWP on a master time axis (entire MOSAiC period with 1 second spacing) have not been
		# created, the following function must be called to create them (long computation time if the chosen
		# period is longer than 2 months!!!):
		if not os.path.exists(path_save_concat_IWV + LWP_concat_hatpro_filename):
			print("Master time axis: HATPRO")
			create_LWP_concat_master_time(date_start, date_end, hatpro_dict, 'hatpro',
											path_save_concat_IWV, LWP_concat_hatpro_filename)

		if not os.path.exists(path_save_concat_IWV + LWP_concat_mirac_filename):
			print("Master time axis: MiRAC-P")
			create_LWP_concat_master_time(date_start, date_end, mirac_dict, 'mirac',
											path_save_concat_IWV, LWP_concat_mirac_filename)

		if not os.path.exists(path_save_concat_IWV + LWP_concat_arm_filename):
			print("Master time axis: ARM")
			create_LWP_concat_master_time(date_start, date_end, arm_dict, 'arm',
											path_save_concat_IWV, LWP_concat_arm_filename)
		
		# If the above mentioned function has already been performed and the netcdf file of LWP on master
		# time axis exists, it suffices to import the LWP (and time axis):
		master_time, hatpro_dict['clwvi_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + LWP_concat_hatpro_filename,
													date_start, date_end)
		master_time, mirac_dict['clwvi_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + LWP_concat_mirac_filename,
													date_start, date_end)
		master_time, arm_dict['lwp_master'] = import_concat_IWV_LWP_mwr_master_time(path_save_concat_IWV + LWP_concat_arm_filename,
													date_start, date_end)

	master_time_dt = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in master_time])


	# IWV difference: HATPRO - MiRAC-P: Use the IWV data on the master time axis to avoid
	# inconsistencies of the instrument specific time axes on each other:
	if plot_IWV_diff_hat_mir: IWV_diff_hat_mir = hatpro_dict['prw_master'] - mirac_dict['IWV_master']
	if plot_IWV_diff_hat_arm: IWV_diff_hat_arm = hatpro_dict['prw_master'] - arm_dict['prw_master']
	if plot_IWV_diff_hat_mir_rel:
		IWV_diff_hat_mir_rel = ((hatpro_dict['prw_master'] - mirac_dict['IWV_master']) /
								(0.5*(hatpro_dict['prw_master'] + mirac_dict['IWV_master'])))


if (considered_period not in ['mosaic', 'mwr_range']) and (plot_option in [2,6]):
	# interpolate flag to new time:
	time_okay_hatpro = np.all(hatpro_dict['time'] == hatpro_dict['time0'])
	time_okay_mirac = np.all(mirac_dict['time'] == mirac_dict['time0'])
	assert time_okay_hatpro
	assert time_okay_mirac
	hatpro_dict['flag'] = np.interp(hatpro_dict['time'], hatpro_dict['time0'], hatpro_dict['flag'], left=-99, right=-99)
	mirac_dict['flag'] = np.interp(mirac_dict['time'], mirac_dict['time0'], mirac_dict['flag'], left=100, right=100)
	if (which_retrievals in ['both', 'lwp', 'clwvi']) and instrument_status['arm']:
		arm_dict['lwp_flag'] = np.interp(arm_dict['time'], arm_dict['time0'], arm_dict['lwp_flag'], left=100, right=100)
	elif (which_retrievals in ['both', 'iwv', 'prw']) and instrument_status['arm']:
		arm_dict['prw_flag'] = np.interp(arm_dict['time'], arm_dict['time0'], arm_dict['prw_flag'], left=100, right=100)


# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")

# calibration times of HATPRO: manually entered from MWR logbook
if skip_pre_calib:
	calibration_times_HATPRO = [dt.datetime(2019,12,14,18,30), 
								dt.datetime(2020,3,1,11,0), dt.datetime(2020,5,2,12,0),
								dt.datetime(2020,7,6,9,33), dt.datetime(2020,8,12,9,17)]
	calibration_times_MiRAC = [dt.datetime(2020,7,6,12,19), dt.datetime(2020,8,12,9,37)]
else:
	calibration_times_HATPRO = [dt.datetime(2019,10,19,6,0), dt.datetime(2019,12,14,18,30), 
								dt.datetime(2020,3,1,11,0), dt.datetime(2020,5,2,12,0),
								dt.datetime(2020,7,6,9,33), dt.datetime(2020,8,12,9,17)]
	calibration_times_MiRAC = [dt.datetime(2019,10,19,6,30), dt.datetime(2019,10,22,5,40),
							dt.datetime(2020,7,6,12,19), dt.datetime(2020,8,12,9,37)]

n_calib_HATPRO = len(calibration_times_HATPRO)
n_calib_MiRAC = len(calibration_times_MiRAC)

# MOSAiC Legs:
MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}


if plot_option == 5:
	# Exclusive Economic Zones (data within these regions may not be published):
	reftime = dt.datetime(1970,1,1)
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
		outside_eez['hatpro'] = np.full((len(master_time),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['hatpro'][(master_time >= EEZ_periods_no_dt[EEZ_range][0]) & (master_time <= EEZ_periods_no_dt[EEZ_range][1])] = False
		if instrument_status['mirac']: outside_eez['mirac'] = outside_eez['hatpro']		# equal because of master time axis

	elif instrument_status['mirac']:
		outside_eez['mirac'] = np.full((len(master_time),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['mirac'][(master_time >= EEZ_periods_no_dt[EEZ_range][0]) & (master_time <= EEZ_periods_no_dt[EEZ_range][1])] = False

	# if Janna Rueckert's results are to be included, EEZ also must be filtered out.
	# interpolation to master time makes as little sense as interpolating sonde launch times to master time axis
	if include_JR_IWV:
		# outside_eez_JR = np.full((len(IWV_JR_dict['time']),), True)
		# for EEZ_range in EEZ_periods_no_dt.keys():
			# outside_eez_JR[(IWV_JR_dict['time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (IWV_JR_dict['time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False
		outside_eez_JR = np.full((len(IWV_JR_DS.time.values),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez_JR[(IWV_JR_DS.time.values >= EEZ_periods_no_dt[EEZ_range][0]) & (IWV_JR_DS.time.values <= EEZ_periods_no_dt[EEZ_range][1])] = False

elif plot_option == 6:
	# Exclusive Economic Zones (data within these regions may not be published):
	reftime = dt.datetime(1970,1,1)
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

	if instrument_status['mirac']:
		outside_eez['mirac'] = np.full((len(mirac_dict['time']),), True)
		for EEZ_range in EEZ_periods_no_dt.keys():
			outside_eez['mirac'][(mirac_dict['time'] >= EEZ_periods_no_dt[EEZ_range][0]) & (mirac_dict['time'] <= EEZ_periods_no_dt[EEZ_range][1])] = False



# If it is desired to plot the IWV difference between one of the microwave radiometers (mwr) and the 
# radiosonde:
# First, find indices when mwr specific time (or master time) equals a sonde launch time. Then average
# over sonde launchtime:launchtime + 15 minutes and compute standard deviation as well.
if (plot_IWV_diff_hatson_mirson or plot_IWV_scatterplots_calib or plot_IWV_scatterplots_legs
	or plot_IWV_scatterplots_all or plot_IWV_diff_mwrson_sonde or plot_IWV_mwrson_diff_histogram):
	# no absolute value search because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged

	if plot_option in [0, 1, 2, 6]:	# find instrument specific time indices:

		if plot_option == 0:
			hatson_idx = np.asarray([np.argwhere((hatpro_dict['time'] >= lt) & 
							(hatpro_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])
			mirson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) &
							(mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])
		
		else:
			hatson_idx = np.asarray([np.argwhere((hatpro_dict['time'] >= lt) &
							(hatpro_dict['time'] <= lt+launch_window) & (hatpro_dict['flag'] == 0)).flatten() for lt in sonde_dict['launch_time']])
			mirson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) &
							(mirac_dict['time'] <= lt+launch_window) & (mirac_dict['flag'] == 0)).flatten() for lt in sonde_dict['launch_time']])

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

	elif plot_option in [3, 4, 5]:		# find indices on master time axis

		hatson_idx = np.asarray([np.argwhere((master_time >= lt) & 
						(master_time <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])
		# mirson_idx = hatson_idx		# in this case because they share the time axis

		hatpro_dict['prw_mean_sonde'] = np.full((n_sondes,), np.nan)
		hatpro_dict['prw_stddev_sonde'] = np.full((n_sondes,), np.nan)
		mirac_dict['prw_mean_sonde'] = np.full((n_sondes,), np.nan)
		mirac_dict['prw_stddev_sonde'] = np.full((n_sondes,), np.nan)
		k = 0
		for hat in hatson_idx:
			hatpro_dict['prw_mean_sonde'][k] = np.nanmean(hatpro_dict['prw_master'][hat])
			hatpro_dict['prw_stddev_sonde'][k] = np.nanstd(hatpro_dict['prw_master'][hat])
			mirac_dict['prw_mean_sonde'][k] = np.nanmean(mirac_dict['prw_master'][hat])
			mirac_dict['prw_stddev_sonde'][k] = np.nanstd(mirac_dict['prw_master'][hat])
			k = k + 1
	
	# now that we have an idea of the IWV from mwrs during sonde launch times, we can compute the difference:
	hatson_iwv = hatpro_dict['prw_mean_sonde'] - sonde_dict['iwv']
	mirson_iwv = mirac_dict['prw_mean_sonde'] - sonde_dict['iwv']

	if plot_IWV_diff_hatson_mirson_rel:
		IWV_diff_hatson_rel = (hatpro_dict['prw_mean_sonde'] - sonde_dict['iwv']) / sonde_dict['iwv']
		IWV_diff_mirson_rel = (mirac_dict['prw_mean_sonde'] - sonde_dict['iwv']) / sonde_dict['iwv']


if plot_IWV_scatterplots_calib:	# compute statistics: number of sondes/entries, correlation coefficient,
								# mean, RMSE, bias per calibration period
	# identify the first indices of radiosonde (launch) time that lies within a calibration period:
	if skip_pre_calib:
		calib_idx_hat = [np.argwhere((sonde_dict['launch_time_dt'] >= dt.datetime(2019,10,19,6,0)) & 
						(sonde_dict['launch_time_dt'] <= calibration_times_HATPRO[0])).flatten()]
		calib_idx_mir = [np.argwhere((sonde_dict['launch_time_dt'] >= dt.datetime(2019,10,22,5,40)) &
						(sonde_dict['launch_time_dt'] <= calibration_times_MiRAC[0])).flatten()]
	else:
		calib_idx_hat = [np.argwhere((sonde_dict['launch_time_dt'] >= date_range_start) & 
						(sonde_dict['launch_time_dt'] <= calibration_times_HATPRO[0])).flatten()]
		calib_idx_mir = [np.argwhere((sonde_dict['launch_time_dt'] >= date_range_start) &
						(sonde_dict['launch_time_dt'] <= calibration_times_MiRAC[0])).flatten()]
	for k, ct_hatpro in enumerate(calibration_times_HATPRO):
		if k < n_calib_HATPRO-1:
			# if (ct_hatpro >= date_range_start) & (ct_hatpro <= date_range_end):
			calib_idx_hat.append(np.argwhere((sonde_dict['launch_time_dt'] >= ct_hatpro) & 
				(sonde_dict['launch_time_dt'] <= calibration_times_HATPRO[k+1])).flatten())
		else:
			# if (ct_hatpro >= date_range_start) & (ct_hatpro <= date_range_end):
			calib_idx_hat.append(np.argwhere(sonde_dict['launch_time_dt'] >= ct_hatpro).flatten())

	for k, ct_mirac in enumerate(calibration_times_MiRAC):
		if k < n_calib_MiRAC-1:
			# if (ct_mirac >= date_range_start) & (ct_mirac <= date_range_end):
			calib_idx_mir.append(np.argwhere((sonde_dict['launch_time_dt'] >= ct_mirac) &
				(sonde_dict['launch_time_dt'] <= calibration_times_MiRAC[k+1])).flatten())
		else:
			# if (ct_mirac >= date_range_start) & (ct_mirac <= date_range_end):
			calib_idx_mir.append(np.argwhere(sonde_dict['launch_time_dt'] >= ct_mirac).flatten())
	

	# compute statistics:
	n_calib_hat = len(calib_idx_hat)
	n_calib_mir = len(calib_idx_mir)
	N_sondes_calib_hat = np.zeros((n_calib_hat,))	# will contain the number of samples of that calib period
	N_sondes_calib_mir = np.zeros((n_calib_mir,))	# will contain the number of samples of that calib period
	bias_calib_hat = np.zeros((n_calib_hat,))		# will contain the bias for each calib period
	bias_calib_mir = np.zeros((n_calib_mir,))		# will contain the bias for each calib period
	rmse_calib_hat = np.zeros((n_calib_hat,))		# will contain the rmse for each calib period
	rmse_calib_mir = np.zeros((n_calib_mir,))		# will contain the rmse for each calib period
	std_calib_hat = np.zeros((n_calib_hat,))		# will contain the std deviation for each calib period
	std_calib_mir = np.zeros((n_calib_mir,))		# will contain the std deviation for each calib period
	R_calib_hat = np.zeros((n_calib_hat,))			# will contain the correl coeff. for each calib period
	R_calib_mir = np.zeros((n_calib_mir,))			# will contain the correl coeff. for each calib period

	for k, cc in enumerate(calib_idx_hat):
		# for each calibration period: compute statistics:
		stat_dict = compute_retrieval_statistics(sonde_dict['iwv'][cc], hatpro_dict['prw_mean_sonde'][cc], compute_stddev=True)
		N_sondes_calib_hat[k] = stat_dict['N']
		bias_calib_hat[k] = stat_dict['bias']
		rmse_calib_hat[k] = stat_dict['rmse']
		std_calib_hat[k] = stat_dict['stddev']
		R_calib_hat[k] = stat_dict['R']

	for k, cc in enumerate(calib_idx_mir):
		# for each calibration period: compute statistics:
		stat_dict = compute_retrieval_statistics(sonde_dict['iwv'][cc], mirac_dict['prw_mean_sonde'][cc], compute_stddev=True)
		N_sondes_calib_mir[k] = stat_dict['N']
		bias_calib_mir[k] = stat_dict['bias']
		rmse_calib_mir[k] = stat_dict['rmse']
		std_calib_mir[k] = stat_dict['stddev']
		R_calib_mir[k] = stat_dict['R']

	del stat_dict


if plot_IWV_scatterplots_legs:	# compute statistics: number of sondes/entries, correlation coefficient,
								# mean, RMSE, bias per MOSAiC leg:
	# identify the first indices of radiosonde (launch) time that lies within a calibration period:
	mosleg_idx = list()
	for key in MOSAiC_legs.keys():
		to_append = np.argwhere((sonde_dict['launch_time_dt'] >= MOSAiC_legs[key][0]) &
			(sonde_dict['launch_time_dt'] <= MOSAiC_legs[key][1])).flatten()
		# if to_append.size > 0:
		mosleg_idx.append(to_append)


	# compute statistics:
	n_mosleg = len(mosleg_idx)
	N_sondes_mosleg_hat = np.zeros((n_mosleg,))	# will contain the number of samples of each calib period
	N_sondes_mosleg_mir = np.zeros((n_mosleg,))	# will contain the number of samples of each calib period
	bias_mosleg_hat = np.zeros((n_mosleg,))		# will contain the bias for each calib period
	bias_mosleg_mir = np.zeros((n_mosleg,))		# will contain the bias for each calib period
	rmse_mosleg_hat = np.zeros((n_mosleg,))		# will contain the rmse for each calib period
	rmse_mosleg_mir = np.zeros((n_mosleg,))		# will contain the rmse for each calib period
	std_mosleg_hat = np.zeros((n_mosleg,))		# will contain the std dev for each calib period
	std_mosleg_mir = np.zeros((n_mosleg,))		# will contain the std dev for each calib period
	R_mosleg_hat = np.zeros((n_mosleg,))			# will contain the correl coeff. for each calib period
	R_mosleg_mir = np.zeros((n_mosleg,))			# will contain the correl coeff. for each calib period

	for k, cc in enumerate(mosleg_idx):
		# for each mosaig leg compute statistics:
		stat_dict_hat = compute_retrieval_statistics(sonde_dict['iwv'][cc], hatpro_dict['prw_mean_sonde'][cc], compute_stddev=True)
		stat_dict_mir = compute_retrieval_statistics(sonde_dict['iwv'][cc], mirac_dict['prw_mean_sonde'][cc], compute_stddev=True)
		N_sondes_mosleg_hat[k] = stat_dict_hat['N']
		N_sondes_mosleg_mir[k] = stat_dict_mir['N']
		bias_mosleg_hat[k] = stat_dict_hat['bias']
		bias_mosleg_mir[k] = stat_dict_mir['bias']
		rmse_mosleg_hat[k] = stat_dict_hat['rmse']
		rmse_mosleg_mir[k] = stat_dict_mir['rmse']
		std_mosleg_hat[k] = stat_dict_hat['stddev']
		std_mosleg_mir[k] = stat_dict_mir['stddev']
		R_mosleg_hat[k] = stat_dict_hat['R']
		R_mosleg_mir[k] = stat_dict_mir['R']

	del stat_dict_hat, stat_dict_mir


if plot_IWV_overview_boxplot and plot_option == 5:
	# Xarray automatically converts master time to numpy datetime64!
	# Create xarray dataset:
	IWV_MWR_DS = xr.Dataset({
					'IWV_hatpro':		(['time'], hatpro_dict['prw_master'],
										{'units': "kg m-2"}),
					'IWV_mirac':		(['time'], mirac_dict['prw_master'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], master_time_dt)})
	# Group by months and make a list out of the groups:
	# to access a group, you have to address the correct group by IWV_MWR_grouped_DS[groupnumber][1]
	# additionally, we need to remove nan values because python's matplotlib cannot ignore them itself
	IWV_MWR_DS_grouped = list(IWV_MWR_DS.resample(time='1M'))
	IWV_MWR_grouped_hatpro = [dss[1].IWV_hatpro.values[~np.isnan(dss[1].IWV_hatpro.values)] for dss in IWV_MWR_DS_grouped]
	IWV_MWR_grouped_mirac = [dss[1].IWV_mirac.values[~np.isnan(dss[1].IWV_mirac.values)] for dss in IWV_MWR_DS_grouped]
	MasterTime_MWR_grouped = [dss[1].time.values for dss in IWV_MWR_DS_grouped]
	
	# Same for radiosonde:
	IWV_RS_DS = xr.Dataset({
					'IWV_radiosonde':	(['time'], sonde_dict['iwv'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], sonde_dict['launch_time_dt'])})
	IWV_RS_DS_grouped = list(IWV_RS_DS.resample(time='1M'))
	IWV_RS_grouped = [dss[1].IWV_radiosonde.values[~np.isnan(dss[1].IWV_radiosonde.values)] for dss in IWV_RS_DS_grouped]
	SondeTime_grouped = [dss[1].time.values for dss in IWV_RS_DS_grouped]

	# Same for Janna Rueckert's IWV if desired:
	if include_JR_IWV:
		IWV_JR_DS = xr.Dataset({
						'IWV_satellite':	(['time'], IWV_JR_dict['IWV'],
											{'units': "kg m-2"})},
						coords = 			{'time': (['time'], IWV_JR_dict['datetime'])})
		IWV_JR_DS_grouped = list(IWV_JR_DS.resample(time='1M'))
		IWV_JR_grouped = [dss[1].IWV_satellite.values[~np.isnan(dss[1].IWV_satellite.values)] for dss in IWV_JR_DS_grouped]
		SatelliteTime_grouped = [dss[1].time.values for dss in IWV_JR_DS_grouped]

	del IWV_MWR_DS, IWV_RS_DS, IWV_MWR_DS_grouped

elif plot_IWV_overview_boxplot and plot_option == 6:  # group IWV by months with xarray:
	# Xarray automatically converts time to numpy datetime64!
	# Create xarray dataset:
	IWV_HAT_DS = xr.Dataset({
					'IWV_hatpro':		(['time'], hatpro_dict['prw'],
										{'units': "kg m-2"})},
					coords=				{'time': (['time'], hatpro_dict['time_npdt'])})
	IWV_MIR_DS = xr.Dataset({
					'IWV_mirac':		(['time'], mirac_dict['prw'],
										{'units': "kg m-2"})},
					coords=				{'time': (['time'], mirac_dict['time_npdt'])})
	# Group by months and make a list out of the groups:
	# to access a group, you have to address the correct group by IWV_MWR_grouped_DS[groupnumber][1]
	# additionally, we need to remove nan values because python's matplotlib cannot ignore them itself
	IWV_HAT_DS_grouped = list(IWV_HAT_DS.resample(time='1M'))
	IWV_MIR_DS_grouped = list(IWV_MIR_DS.resample(time='1M'))
	IWV_MWR_grouped_hatpro = [dss[1].IWV_hatpro.values[~np.isnan(dss[1].IWV_hatpro.values)] for dss in IWV_HAT_DS_grouped]
	IWV_MWR_grouped_mirac = [dss[1].IWV_mirac.values[~np.isnan(dss[1].IWV_mirac.values)] for dss in IWV_MIR_DS_grouped]
	HAT_time_grouped = [dss[1].time.values for dss in IWV_HAT_DS_grouped]
	MIR_time_grouped = [dss[1].time.values for dss in IWV_MIR_DS_grouped]
	
	# Same for radiosonde:
	IWV_RS_DS = xr.Dataset({
					'IWV_radiosonde':	(['time'], sonde_dict['iwv'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], sonde_dict['launch_time'].astype("datetime64[s]"))})
	IWV_RS_DS_grouped = list(IWV_RS_DS.resample(time='1M'))
	IWV_RS_grouped = [dss[1].IWV_radiosonde.values[~np.isnan(dss[1].IWV_radiosonde.values)] for dss in IWV_RS_DS_grouped]
	SondeTime_grouped = [dss[1].time.values for dss in IWV_RS_DS_grouped]

	del IWV_HAT_DS, IWV_MIR_DS, IWV_RS_DS, IWV_HAT_DS_grouped, IWV_MIR_DS_grouped


if plot_LWP_overview_boxplot and plot_option == 5: # group LWP by months with xarray:
	# Xarray automatically converts master time to numpy datetime64!
	# Create xarray dataset:
	LWP_MWR_DS = xr.Dataset({
					'LWP_hatpro':		(['time'], hatpro_dict['clwvi_master'],
										{'units': "kg m-2"}),
					'LWP_mirac':		(['time'], mirac_dict['clwvi_master'],
										{'units': "kg m-2"}),
					'LWP_arm':			(['time'], arm_dict['lwp_master'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], master_time_dt)})
	# Group by months and make a list out of the groups:
	# to access a group, you have to address the correct group by IWV_MWR_grouped_DS[groupnumber][1]
	# additionally, we need to remove nan values because python's matplotlib cannot ignore them itself
	LWP_MWR_DS_grouped = list(LWP_MWR_DS.resample(time='1M'))
	LWP_MWR_grouped_hatpro = [dss[1].LWP_hatpro.values[~np.isnan(dss[1].LWP_hatpro.values)] for dss in LWP_MWR_DS_grouped]
	LWP_MWR_grouped_mirac = [dss[1].LWP_mirac.values[~np.isnan(dss[1].LWP_mirac.values)] for dss in LWP_MWR_DS_grouped]
	LWP_MWR_grouped_arm = [dss[1].LWP_arm.values[~np.isnan(dss[1].LWP_arm.values)] for dss in LWP_MWR_DS_grouped]
	MasterTime_MWR_grouped = [dss[1].time.values for dss in LWP_MWR_DS_grouped]

	del LWP_MWR_DS

elif plot_LWP_overview_boxplot and plot_option == 6: # group LWP by months with xarray:
	# Xarray automatically converts master time to numpy datetime64!
	# Create xarray dataset:
	LWP_HAT_DS = xr.Dataset({
					'LWP_hatpro':		(['time'], hatpro_dict['clwvi'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], hatpro_dict['time_npdt'])})
	LWP_MIR_DS = xr.Dataset({
					'LWP_mirac':		(['time'], mirac_dict['clwvi'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], mirac_dict['time_npdt'])})
	LWP_ARM_DS = xr.Dataset({
					'LWP_arm':		(['time'], arm_dict['lwp'],
										{'units': "kg m-2"})},
					coords = 			{'time': (['time'], arm_dict['time_npdt'])})
	# Group by months and make a list out of the groups:
	# to access a group, you have to address the correct group by IWV_MWR_grouped_DS[groupnumber][1]
	# additionally, we need to remove nan values because python's matplotlib cannot ignore them itself
	LWP_HAT_DS_grouped = list(LWP_HAT_DS.resample(time='1M'))
	LWP_MIR_DS_grouped = list(LWP_MIR_DS.resample(time='1M'))
	LWP_ARM_DS_grouped = list(LWP_ARM_DS.resample(time='1M'))
	LWP_MWR_grouped_hatpro = [dss[1].LWP_hatpro.values[~np.isnan(dss[1].LWP_hatpro.values)] for dss in LWP_HAT_DS_grouped]
	LWP_MWR_grouped_mirac = [dss[1].LWP_mirac.values[~np.isnan(dss[1].LWP_mirac.values)] for dss in LWP_MIR_DS_grouped]
	LWP_MWR_grouped_arm = [dss[1].LWP_arm.values[~np.isnan(dss[1].LWP_arm.values)] for dss in LWP_ARM_DS_grouped]
	HAT_time_grouped = [dss[1].time.values for dss in LWP_HAT_DS_grouped]
	MIR_time_grouped = [dss[1].time.values for dss in LWP_MIR_DS_grouped]
	ARM_time_grouped = [dss[1].time.values for dss in LWP_ARM_DS_grouped]

	del LWP_HAT_DS, LWP_MIR_DS, LWP_ARM_DS


# dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
dt_fmt = mdates.DateFormatter("%b %d") # (e.g. "Feb 23")
datetick_auto = False
fs = 24		# fontsize

# colors:
c_JR = (1,0.663,0)			# Janna Rueckert's IWV data
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P
c_RS = (1,0.435,0)			# radiosondes
c_ARM = (0.80,0.1,0.1)		# ARM MWR

# create x_ticks depending on the date range: roughly 20 x_ticks are planned
# round off to days if number of days > 15:
date_range_delta = (date_range_end - date_range_start)
if date_range_delta < dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=6)
	# dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta < dt.timedelta(days=11) and date_range_delta >= dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=12)
	# dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta >= dt.timedelta(days=11) and date_range_delta < dt.timedelta(21):
	x_tick_delta = dt.timedelta(days=1)
else:
	x_tick_delta = dt.timedelta(days=(date_range_delta/20).days)
	# x_tick_delta = dt.timedelta(days=2)

x_ticks_dt = mdates.drange(date_range_start, date_range_end, x_tick_delta)	# alternative if the xticklabel is centered
# x_ticks_dt = mdates.drange(date_range_start + x_tick_delta, date_range_end + x_tick_delta, x_tick_delta)

# MOSAiC Legs text locations in plot
MOSAiC_Legs_Text = ["Leg 1", "Leg 2", "Leg 3", "Leg 4", "Leg 5"]
MLT_x_pos = []
for key in MOSAiC_legs.keys():
	if MOSAiC_legs[key][0] < date_range_start and MOSAiC_legs[key][1] <= date_range_end:
		MLT_x_pos.append(date_range_start + 0.5*(MOSAiC_legs[key][1] - date_range_start))
	elif MOSAiC_legs[key][0] >= date_range_start and MOSAiC_legs[key][1] > date_range_end:
		MLT_x_pos.append(MOSAiC_legs[key][0] + 0.5*(date_range_end - MOSAiC_legs[key][0]))
	else:
		MLT_x_pos.append(MOSAiC_legs[key][0] + 0.5*(MOSAiC_legs[key][1] - MOSAiC_legs[key][0]))



########## Plotting ##########
if (plot_option == 6) and which_retrievals in ['prw', 'both', 'iwv']:
	# removing flagged values from plotting
	mirac_dict['prw'][mirac_dict['flag'] > 0] = np.nan
	hatpro_dict['prw'][hatpro_dict['flag'] > 0] = np.nan
	mirac_dict['prw'][~outside_eez['mirac']] = np.nan
	hatpro_dict['prw'][~outside_eez['hatpro']] = np.nan

	# also remove values before first (second) calibration for HATPRO (MiRAC-P):
	first_calib_HATPRO = datetime_to_epochtime(calibration_times_HATPRO[0])
	first_calib_MiRAC = datetime_to_epochtime(calibration_times_MiRAC[1])
	no_calib_hatpro = np.where(hatpro_dict['time'] < first_calib_HATPRO)[0]
	no_calib_mirac = np.where(mirac_dict['time'] < first_calib_MiRAC)[0]
	hatpro_dict['prw'][no_calib_hatpro] = np.nan
	mirac_dict['prw'][no_calib_mirac] = np.nan

	if (plot_IWV_diff_hatson_mirson or (plot_IWV_scatterplots_calib and not skip_pre_calib) or plot_IWV_scatterplots_legs
		or plot_IWV_scatterplots_all or plot_IWV_diff_mwrson_sonde or plot_IWV_mwrson_diff_histogram):
		no_calib_hatpro = np.where(sonde_dict['launch_time'] < first_calib_HATPRO)[0]
		no_calib_mirac = np.where(sonde_dict['launch_time'] < first_calib_MiRAC)[0]
		hatpro_dict['prw_mean_sonde'][no_calib_hatpro] = np.nan
		hatpro_dict['prw_stddev_sonde'][no_calib_hatpro] = np.nan
		mirac_dict['prw_mean_sonde'][no_calib_mirac] = np.nan
		mirac_dict['prw_stddev_sonde'][no_calib_mirac] = np.nan

if (plot_option == 6) and which_retrievals in ['lwp', 'clwvi', 'both']:
	# removing flagged values from plotting
	# also remove values before first (second) calibration for HATPRO (MiRAC-P):
	if instrument_status['mirac']: 
		mirac_dict['clwvi'][mirac_dict['flag'] > 0] = np.nan
		first_calib_MiRAC = datetime_to_epochtime(calibration_times_MiRAC[1])
		no_calib_mirac = np.where(mirac_dict['time'] < first_calib_MiRAC)[0]
		mirac_dict['prw'][no_calib_mirac] = np.nan

	hatpro_dict['clwvi'][hatpro_dict['flag'] > 0] = np.nan
	first_calib_HATPRO = datetime_to_epochtime(calibration_times_HATPRO[0])
	no_calib_hatpro = np.where(hatpro_dict['time'] < first_calib_HATPRO)[0]
	hatpro_dict['clwvi'][no_calib_hatpro] = np.nan
	hatpro_dict['clwvi'][~outside_eez['hatpro']] = np.nan


if plot_IWV_time_series:

	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig1, ax1 = plt.subplots(1,2)
	fig1.set_size_inches(22,10)

	if include_data_availability:
		axlim = [-5, 35]		# axis limits for IWV plot in kg m^-2

		# compute data availability: cycle through all days
		date_start_npdt = np.datetime64(date_start + "T00:00:00")
		date_end_npdt = np.datetime64(date_end + "T00:00:00")
		now_date_npdt = date_start_npdt
		n_days = (date_end_npdt - date_start_npdt).astype("timedelta64[D]").astype("int32")+1
		kk = 0
		date_array = np.full((n_days,), now_date_npdt)
		mirac_data_flagged = np.zeros((n_days,), dtype="float32")		# fraction of available data
		while now_date_npdt <= date_end_npdt:
			# find indices of MiRAC-P in the range [now_date_npdt:now_date_npdt + 1 day)
			miracix = np.where((mirac_dict['time_npdt'] >= now_date_npdt) & 
								(mirac_dict['time_npdt'] < now_date_npdt + np.timedelta64(1, "D")))[0]

			n_tot_mir = len(miracix)
			n_flagged_mir = np.count_nonzero(np.isnan(mirac_dict['prw'][miracix]))

			date_array[kk] = now_date_npdt
			if n_tot_mir > 0:
				mirac_data_flagged[kk] = n_flagged_mir / n_tot_mir		# fraction of flagged data
			else:
				mirac_data_flagged[kk] = 1.0			# then 100 % is flagged
			
			kk += 1
			now_date_npdt += np.timedelta64(1, "D")
		
	else:
		axlim = [0, 35]			# axis limits for IWV plot in kg m^-2


	if plot_option not in [5,6]:
		SONDE_IWV_plot = ax1[0].plot(sonde_dict['launch_time_dt'], sonde_dict['iwv'], linestyle='none', marker='.', linewidth=0.5,
										markersize=15.0, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
										markeredgewidth=0.5, label='Radiosonde')
	else:
		SONDE_IWV_plot = ax1[0].plot(sonde_dict['launch_time_dt'][outside_eez['sonde']], sonde_dict['iwv'][outside_eez['sonde']], linestyle='none', 
										marker='.', linewidth=0.5,
										markersize=15.0, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
										markeredgewidth=0.5, label='Radiosonde')

	if plot_option == 0:
		MIRAC_IWV_plot = ax1[0].plot(mirac_dict['datetime'], mirac_dict['prw'],
									color=c_M, linewidth=1.0, label="MiRAC-P")
		HATPRO_IWV_plot = ax1[0].plot(hatpro_dict['datetime'], hatpro_dict['prw'], 
									color=c_H, linewidth=1.0, label='HATPRO')
		ARM_IWV_plot = ax1[0].plot(arm_dict['datetime'], arm_dict['prw'],
									color=c_ARM, linewidth=1.0, alpha=0.6, label='ARM')
	elif plot_option in [1, 2]:
		MIRAC_IWV_plot = ax1[0].plot(mirac_dict['datetime'][mirac_dict['flag'] == 0.0], mirac_dict['prw'][mirac_dict['flag'] == 0.0],
										color=c_M, linewidth=1.0)
		HATPRO_IWV_plot = ax1[0].plot(hatpro_dict['datetime'][hatpro_dict['flag'] == 0.0], hatpro_dict['prw'][hatpro_dict['flag'] == 0.0], 
										color=c_H, linewidth=1.0)
		ARM_IWV_plot = ax1[0].plot(arm_dict['datetime'][arm_dict['prw_flag'] == 0], arm_dict['prw'][arm_dict['prw_flag'] == 0],
									color=c_ARM, linewidth=1.0, alpha=0.6)
	elif plot_option in [3, 4]:
		MIRAC_IWV_plot = ax1[0].plot(master_time_dt, mirac_dict['prw_master'],
										color=c_M, linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=c_M, markeredgecolor=c_M, 
									markeredgewidth=0.5)
		HATPRO_IWV_plot = ax1[0].plot(master_time_dt, hatpro_dict['prw_master'], 
										color=c_H, linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=c_H, markeredgecolor=c_H, 
									markeredgewidth=0.5)
		ARM_IWV_plot = ax1[0].plot(master_time_dt, arm_dict['prw_master'], linestyle='none', marker='.', linewidth=0.5,
									markersize=3.5, markerfacecolor=c_ARM, markeredgecolor=c_ARM, 
									markeredgewidth=0.5, color=c_ARM, alpha=0.6)
		ARM_IWV_plot_dummy = ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='.', linewidth=0.5,
									markersize=15, markerfacecolor=c_ARM, markeredgecolor=c_ARM, 
									markeredgewidth=0.5, color=c_ARM, alpha=0.8)
	elif plot_option in [5]:
		arm_nonnan = np.where(~np.isnan(arm_dict['prw_master'][outside_eez]))[0]
		# ARM_IWV_plot = ax1[0].plot(master_time_dt[arm_nonnan], arm_dict['prw_master'][arm_nonnan], # linestyle='none', marker='.', linewidth=0.5,
									# # markersize=3.5, markerfacecolor=c_ARM, markeredgecolor=c_ARM, 
									# # markeredgewidth=0.5, 
									# linewidth=1.0,
									# color=c_ARM, alpha=0.7)
		# ARM_IWV_plot_dummy = ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], # linestyle='none', marker='.', linewidth=0.5,
									# # markersize=15, markerfacecolor=c_ARM, markeredgecolor=c_ARM, 
									# # markeredgewidth=0.5, 
									# linewidth=1.2,
									# color=c_ARM, alpha=0.7)
		MIRAC_IWV_plot = ax1[0].plot(master_time_dt[outside_eez], mirac_dict['prw_master'][outside_eez],
										color=c_M, linewidth=1.2, # marker='.', markersize=1.5,
									# markerfacecolor=c_M, markeredgecolor=c_M, 
									# markeredgewidth=0.5, 
									)
		HATPRO_IWV_plot = ax1[0].plot(master_time_dt[outside_eez], hatpro_dict['prw_master'][outside_eez], 
										color=c_H, linewidth=1.2, # marker='.', markersize=1.5,
									# markerfacecolor=c_H, markeredgecolor=c_H, 
									# markeredgewidth=0.5, 
									)
	elif plot_option in [6]:
		HATPRO_IWV_plot = ax1[0].plot(hatpro_dict['time_npdt'][outside_eez['hatpro']], hatpro_dict['prw'][outside_eez['hatpro']],
										color=c_H, linewidth=1.2)
		MIRAC_IWV_plot = ax1[0].plot(mirac_dict['time_npdt'][outside_eez['mirac']], mirac_dict['prw'][outside_eez['mirac']],
										color=c_M, linewidth=1.2)

	if include_data_availability:
		# scale the data availability to 0 - -5: 0 = 0% flagged; -5 = 100 % flagged:
		mirac_data_flagged *= -5
		ax1[0].bar(date_array, mirac_data_flagged, width=1.0, color=(0.9,0,0))



	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend + dummy lines for thicker lines in the legend:
	ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, label='HATPRO')
	ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linewidth=2.0, label='MiRAC-P')
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	if include_JR_IWV:
		ax1[0].legend(handles=[SONDE_IWV_plot[0], ARM_IWV_plot_dummy[0], MIRAC_IWV_plot[0], HATPRO_IWV_plot[0], (JR_IWV_plot[0], JR_IWV_std_dummy[0])], 
						labels=iwv_leg_labels, loc='upper left', fontsize=fs,
						framealpha=1.0, markerscale=1.5)
	else:
		ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0, markerscale=1.5)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=dt.datetime(2019,10,22,6,0), right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles:
		the_title = ax1[0].set_title("Integrated Water Vapour (IWV) during MOSAiC (" + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
							" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")", fontsize=fs, pad=50)
		the_title.set_position((0.5, 1.2))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
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

	if include_data_availability:
		# dummy plot for legend:
		ax1[1].bar([np.nan, np.nan], [np.nan, np.nan], width=1.0, color=(0.9,0,0), 
					label="MiRAC-P data flagged")

		ytick_labels = [yt for yt in ax1[0].get_yticks()]
		ax1[0].yaxis.set_ticklabels(["100$\,\%$"] + ytick_labels[1:])

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_mirac_sonde_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_mirac_sonde_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')

	elif save_figures_eps:
		iwv_name_base = "IWV_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_mirac_sonde_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_mirac_sonde_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_IWV_diff_hat_arm and (plot_option in [3, 4]):

	# IWV difference: HATPRO - ARM
	fig20, ax1 = plt.subplots(1,2)
	fig20.set_size_inches(22,10)

	axlim = [-4, 4]		# axis limits for IWV plot in kg m^-2

	MIRAC_IWV_plot = ax1[0].plot(master_time_dt, IWV_diff_hat_arm,
									color=(0,0,0), linewidth=1.0, marker='.', markersize=2.0,
									markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label="HATPRO - ARM")

	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("IWV$_{\mathrm{HATPRO}} - $IWV$_{\mathrm{ARM}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig20.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-3, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig20.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs+2)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_diff_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro-arm_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro-arm_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig20.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig20.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_diff_hat_mir and (plot_option in [3, 4]):

	# IWV difference: HATPRO - MiRAC-P
	fig2, ax1 = plt.subplots(1,2)
	fig2.set_size_inches(22,10)

	axlim = [-4, 4]		# axis limits for IWV plot in kg m^-2

	MIRAC_IWV_plot = ax1[0].plot(master_time_dt, IWV_diff_hat_mir,
									color=(0,0,0), linewidth=1.0, marker='.', markersize=2.0,
									markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label="HATPRO - MiRAC-P")

	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("IWV$_{\mathrm{HATPRO}} - $IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig2.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig2.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linestyle='dashed', linewidth=2,
				label="$\\bf{MiRAC-P}$")
	for ct_mirac in calibration_times_MiRAC: 
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig2.get_facecolor(),
				label=dt.datetime.strftime(ct_mirac, "%d.%m.%Y, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs+2)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_diff_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro-mirac_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro-mirac_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig2.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig2.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_diff_hat_mir_rel and (plot_option in [3, 4]):

	# IWV difference: HATPRO - MiRAC-P but normed by the average of the HATPRO and MiRAC-P observations
	fig21, ax1 = plt.subplots(1,2)
	fig21.set_size_inches(22,10)

	axlim = [-1, 1]		# axis limits for IWV plot in kg m^-2

	MIRAC_IWV_plot = ax1[0].plot(master_time_dt, IWV_diff_hat_mir_rel,
									color=(0,0,0), linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label="HATPRO - MiRAC-P")

	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$(\mathrm{IWV}_{\mathrm{H.}} - \mathrm{IWV}_{\mathrm{M.}}) / (0.5\cdot(\mathrm{IWV}_{\mathrm{H.}} + \mathrm{IWV}_{\mathrm{M.}}))$ ()", fontsize=fs-2)

	if datetick_auto:
		fig21.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig21.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linestyle='dashed', linewidth=2,
				label="$\\bf{MiRAC-P}$")
	for ct_mirac in calibration_times_MiRAC: 
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig21.get_facecolor(),
				label=dt.datetime.strftime(ct_mirac, "%d.%m.%Y, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs+2)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_rel_diff_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro-mirac_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro-mirac_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig21.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig21.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_diff_hatson_mirson:
	# IWV difference: HATPRO - sonde and MiRAC-P - sonde:
	fig3, ax1 = plt.subplots(1,2)
	fig3.set_size_inches(22,10)

	axlim = [-5, 5]		# axis limits for IWV difference plot in kg m^-2

	MIRSON_plot = ax1[0].errorbar(sonde_dict['launch_time_dt'], mirson_iwv, yerr=mirac_dict['IWV_stddev_sonde'],
								ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
								linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P - Radiosonde')
	HATSON_plot = ax1[0].errorbar(sonde_dict['launch_time_dt'], hatson_iwv, yerr=hatpro_dict['prw_stddev_sonde'],
								ecolor=c_H, elinewidth=1.2, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
								linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='HATPRO - Radiosonde')


	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta$IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig3.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig3.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linestyle='dashed', linewidth=2,
				label="$\\bf{MiRAC-P}$")
	for ct_mirac in calibration_times_MiRAC: 
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig3.get_facecolor(),
				label=dt.datetime.strftime(ct_mirac, "%d.%m.%Y, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs+2)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_diff_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_and_mirac_vs_sonde_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_and_mirac_vs_sonde_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig3.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig3.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_diff_hatson_mirson_rel:
	# IWV difference: HATPRO - sonde and MiRAC-P - sonde: but normed by the average of MWR + sonde
	fig30, ax1 = plt.subplots(1,2)
	fig30.set_size_inches(22,10)

	axlim = [-1, 1]		# axis limits for IWV plot in kg m^-2

	MIRSON_plot = ax1[0].errorbar(sonde_dict['launch_time_dt'], IWV_diff_mirson_rel,
								ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
								linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P - Radiosonde')
	HATSON_plot = ax1[0].errorbar(sonde_dict['launch_time_dt'], IWV_diff_hatson_rel,
								ecolor=c_H, elinewidth=1.2, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
								linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='HATPRO - Radiosonde')


	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)
	for ct_mirac in calibration_times_MiRAC:
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_M, linestyle='dashed', linewidth=2)
	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV} \, / \, \overline{\mathrm{IWV}}$ ()", fontsize=fs)

	if datetick_auto:
		fig30.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# add dummy lines for the legend:
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in calibration_times_HATPRO: 
		if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig30.get_facecolor(),
				label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))
	ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linestyle='dashed', linewidth=2,
				label="$\\bf{MiRAC-P}$")
	for ct_mirac in calibration_times_MiRAC: 
		if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# same color as background to be invisible
			ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig30.get_facecolor(),
				label=dt.datetime.strftime(ct_mirac, "%d.%m.%Y, %H:%M UTC"))

	cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	lo.get_title().set_fontsize(fs+2)
	lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "IWV_rel_diff_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_and_mirac_vs_sonde_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_and_mirac_vs_sonde_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig30.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig30.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_diff_mwrson_sonde and (plot_option in [2,6]):
	# x axis: IWV_sonde; y axis: IWV_MWR - IWV_sonde

	# HATPRO:
	fig1 = plt.figure(figsize=(16,10))
	ax1 = plt.axes()

	ylim = [-4, 4]		# axis limits: y
	xlim = [0, 35]		# axis limits: x
	
	mwrsonson_plot = ax1.errorbar(sonde_dict['iwv'], hatpro_dict['prw_mean_sonde'] - sonde_dict['iwv'],
									yerr=hatpro_dict['prw_stddev_sonde'], ecolor=c_H, elinewidth=1.6, 
									capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0), linestyle='none',
									marker='.', markersize=10.0, linewidth=1.2, capthick=1.6)

	ax1.plot([xlim[0], xlim[1]], [0.0, 0.0], color=(0,0,0), linewidth=1.0)		# dummy to highlight the 0 line

	ax1.set_ylim(bottom=ylim[0], top=ylim[1])
	ax1.set_xlim(left=xlim[0], right=xlim[1])

	ax1.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_ylabel("IWV$_{\mathrm{HATPRO}} - \mathrm{IWV}_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	the_title = ax1.set_title("Integrated Water Vapour (IWV) difference between\nHATPRO and radiosonde", fontsize=fs)
	the_title.set_position((0.5, 1.2))

	ax1.tick_params(axis='both', labelsize=fs-2)
	ax1.grid(which='major', axis='both')

	
	if save_figures:
		iwv_name_base = "IWV_diff_hatpro_sonde_vs_sonde_"
		if considered_period != 'user':
			iwv_name_suffix_def = considered_period
		else:
			iwv_name_suffix_def = date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


	# MiRAC-P:
	fig1 = plt.figure(figsize=(16,10))
	ax1 = plt.axes()

	ylim = [-4, 4]		# axis limits: y
	xlim = [0, 35]		# axis limits: x
	
	mwrsonson_plot = ax1.errorbar(sonde_dict['iwv'], mirac_dict['prw_mean_sonde'] - sonde_dict['iwv'],
									yerr=mirac_dict['prw_stddev_sonde'], ecolor=c_M, elinewidth=1.6, 
									capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0), linestyle='none',
									marker='.', markersize=10.0, linewidth=1.2, capthick=1.6)

	ax1.plot([xlim[0], xlim[1]], [0.0, 0.0], color=(0,0,0), linewidth=1.0)		# dummy to highlight the 0 line

	ax1.set_ylim(bottom=ylim[0], top=ylim[1])
	ax1.set_xlim(left=xlim[0], right=xlim[1])

	ax1.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_ylabel("IWV$_{\mathrm{MiRAC-P}} - \mathrm{IWV}_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	the_title = ax1.set_title("Integrated Water Vapour (IWV) difference between\nMiRAC-P and radiosonde", fontsize=fs)
	the_title.set_position((0.5, 1.2))

	ax1.tick_params(axis='both', labelsize=fs-2)
	ax1.grid(which='major', axis='both')

	
	if save_figures:
		iwv_name_base = "IWV_diff_mirac_sonde_vs_sonde_"
		if considered_period != 'user':
			iwv_name_suffix_def = considered_period
		else:
			iwv_name_suffix_def = date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_scatterplots_calib:
	# IWV scatterplot for calibration times:
	fs = 14
	# HATPRO:
	fig4, ax1 = plt.subplots(3,3)
	fig4.set_size_inches(12,12)
	ax1 = ax1.flatten()

	if scatplot_fix_axlims: axlim = np.asarray([0, 35])

	for k in range(n_calib_hat):	# each panel represents one calibration period

		if not scatplot_fix_axlims:
			if calib_idx_hat[k].size > 0:
				axlim = np.asarray([0, np.nanmax(np.concatenate((sonde_dict['iwv'][calib_idx_hat[k]],
								hatpro_dict['prw_mean_sonde'][calib_idx_hat[k]]), axis=0))+2])
			else:
				axlim = np.asarray([0,1])

		
		ax1[k].errorbar(sonde_dict['iwv'][calib_idx_hat[k]], hatpro_dict['prw_mean_sonde'][calib_idx_hat[k]], 
							yerr=hatpro_dict['prw_stddev_sonde'][calib_idx_hat[k]],
							ecolor=c_H, elinewidth=1.2, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
							linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='HATPRO')

		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_hatson = np.argwhere(~np.isnan(hatpro_dict['prw_mean_sonde'][calib_idx_hat[k]]) &
							~np.isnan(sonde_dict['iwv'][calib_idx_hat[k]])).flatten()
		y_fit = hatpro_dict['prw_mean_sonde'][calib_idx_hat[k][nonnan_hatson]]
		x_fit = sonde_dict['iwv'][calib_idx_hat[k][nonnan_hatson]]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = ax1[k].plot(axlim, a*axlim + b, color=c_H, linewidth=0.75, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		ax1[k].plot(axlim, axlim, color=(0,0,0), linewidth=0.75, alpha=0.5, label="Theoretical perfect fit")


		# add statistics:
		ax1[k].text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(N_sondes_calib_hat[k], 
				np.nanmean(np.concatenate((sonde_dict['iwv'][calib_idx_hat[k]], hatpro_dict['prw_mean_sonde'][calib_idx_hat[k]]), axis=0)),
				bias_calib_hat[k], rmse_calib_hat[k], std_calib_hat[k], R_calib_hat[k]),
				horizontalalignment='right', verticalalignment='bottom', transform=ax1[k].transAxes, fontsize=fs-4)


		# set axis limits and labels:
		ax1[k].set_ylim(bottom=axlim[0], top=axlim[1])
		ax1[k].set_xlim(left=axlim[0], right=axlim[1])

		ax1[k].tick_params(axis='both', labelsize=fs-4)

		ax1[k].set_aspect('equal')


		ax1[k].minorticks_on()
		ax1[k].set_yticks(ax1[k].get_xticks())
		ax1[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1[k].set_ylim(bottom=axlim[0], top=axlim[1])
		ax1[k].set_xlim(left=axlim[0], right=axlim[1])

		if k%3 == 0: ax1[k].set_ylabel("IWV$_{\mathrm{HATPRO}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		if (k >= 4) and not skip_pre_calib:
			ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		elif (k >= 3) and skip_pre_calib:
			ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		

		the_title = fig4.suptitle("IWV comparison of HATPRO and radiosonde\nfor each calibration period", fontsize=fs)
		the_title.set_position((0.5, 0.94))

		if (k > 0) and (k < n_calib_hat-1):
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(calibration_times_HATPRO[k-1], "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_HATPRO[k], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		elif (k == 0) and not skip_pre_calib:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(date_range_start, "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_HATPRO[0], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		elif (k == 0) and skip_pre_calib:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(dt.datetime(2019,10,19,6,0), "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_HATPRO[0], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		else:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(calibration_times_HATPRO[k-1], "%Y-%m-%d") + " - " +
						dt.datetime.strftime(date_range_end, "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)

	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()

	# use the lower left panel for additional information (legend; MOSAiC leg time)
	ax1[6].axis('off')
	ax1[7].axis('off')
	ax1[8].axis('off')

	ax1[7].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), fontsize=fs)

	if save_figures:
		iwv_name_base = "IWV_scatterplot_CAL_total_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_hatpro_vs_sonde"
		else:
			iwv_name_suffix_def = "_hatpro_vs_sonde"
		fig4.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig4.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


	# IWV scatterplot for calibration periods: now MiRAC-P:
	fs = 17
	fig5, ax1 = plt.subplots(2,3)
	fig5.set_size_inches(16,10)
	ax1 = ax1.flatten()

	if scatplot_fix_axlims: axlim = np.asarray([0, 35])

	for k in range(n_calib_mir):	# each panel represents one MOSAiC leg

		if not scatplot_fix_axlims:
			if calib_idx_mir[k].size > 0:
				axlim = np.asarray([0, np.nanmax(np.concatenate((sonde_dict['iwv'][calib_idx_mir[k]],
								mirac_dict['IWV_mean_sonde'][calib_idx_mir[k]]), axis=0))+2])
			else:
				axlim = np.asarray([0,1])
		
		ax1[k].errorbar(sonde_dict['iwv'][calib_idx_mir[k]], mirac_dict['prw_mean_sonde'][calib_idx_mir[k]], 
							yerr=mirac_dict['prw_stddev_sonde'][calib_idx_mir[k]],
							ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
							linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P')


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_mirson = np.argwhere(~np.isnan(mirac_dict['prw_mean_sonde'][calib_idx_mir[k]]) &
							~np.isnan(sonde_dict['iwv'][calib_idx_mir[k]])).flatten()
		y_fit = mirac_dict['prw_mean_sonde'][calib_idx_mir[k][nonnan_mirson]]
		x_fit = sonde_dict['iwv'][calib_idx_mir[k][nonnan_mirson]]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = ax1[k].plot(axlim, a*axlim + b, color=c_M, linewidth=0.75, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		ax1[k].plot(axlim, axlim, color=(0,0,0), linewidth=0.75, alpha=0.5, label="Theoretical perfect fit")


		# add statistics:
		ax1[k].text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(N_sondes_calib_mir[k], 
				np.nanmean(np.concatenate((sonde_dict['iwv'][calib_idx_mir[k]], mirac_dict['prw_mean_sonde'][calib_idx_mir[k]]), axis=0)),
				bias_calib_mir[k], rmse_calib_mir[k], std_calib_mir[k], R_calib_mir[k]),
				horizontalalignment='right', verticalalignment='bottom', transform=ax1[k].transAxes, fontsize=fs-4)


		# set axis limits and labels:
		ax1[k].set_ylim(bottom=axlim[0], top=axlim[1])
		ax1[k].set_xlim(left=axlim[0], right=axlim[1])

		ax1[k].tick_params(axis='both', labelsize=fs-4)

		ax1[k].set_aspect('equal')


		ax1[k].minorticks_on()
		ax1[k].set_yticks(ax1[k].get_xticks())
		ax1[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1[k].set_ylim(bottom=axlim[0], top=axlim[1])
		ax1[k].set_xlim(left=axlim[0], right=axlim[1])

		if k%3 == 0: ax1[k].set_ylabel("IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		if (k >= 3) and not skip_pre_calib:
			ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		elif skip_pre_calib:
			ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

		the_title = fig5.suptitle("IWV comparison of MiRAC-P and radiosonde\nfor each calibration period", fontsize=fs)


		if (k > 0) and (k < n_calib_mir-1):
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(calibration_times_MiRAC[k-1], "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_MiRAC[k], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		elif (k == 0) and not skip_pre_calib:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(date_range_start, "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_MiRAC[0], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		elif (k == 0) and skip_pre_calib:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(dt.datetime(2019,10,22,5,40), "%Y-%m-%d") + " - " +
						dt.datetime.strftime(calibration_times_MiRAC[0], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)
		else:
			ax1[k].text(0.01, 0.99, dt.datetime.strftime(calibration_times_MiRAC[k-1], "%Y-%m-%d") + " - " +
						dt.datetime.strftime(date_range_end, "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
						transform=ax1[k].transAxes, fontsize=fs-4)

	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()

	# use the lower left panel for additional information (legend; MOSAiC leg time)
	ax1[3].axis('off')
	ax1[4].axis('off')
	ax1[5].axis('off')

	ax1[4].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=fs)

	if save_figures:
		iwv_name_base = "IWV_scatterplot_CAL_total_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_mirac_vs_sonde"
		else:
			iwv_name_suffix_def = "_mirac_vs_sonde"
			
		fig5.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# fig5.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", orientation='landscape')
	else:
		plt.show()


if plot_IWV_scatterplots_legs:
	fs = 17
	# IWV scatterplot for mosaic legs: first for HATPRO:
	fig6, ax1 = plt.subplots(2,3)
	fig6.set_size_inches(16,10)
	ax1 = ax1.flatten()
	
	# handle axis limits:
	if scatplot_fix_axlims:
		axlim = np.asarray([[0, 10], [0, 8], [0, 20], [0, 35], [0, 25]])

	else:
		axlim = np.zeros((5,2))
		for k in range(n_mosleg):
			if mosleg_idx[k].size > 0:
				axlim[k] = [0, np.ceil(np.nanmax(np.concatenate((sonde_dict['iwv'][mosleg_idx[k]],
								hatpro_dict['prw_mean_sonde'][mosleg_idx[k]]), axis=0))+2)]
			else:
				axlim[k] = [0, 1]

	for k in range(n_mosleg):	# each panel represents one MOSAiC leg
		
		ax1[k].errorbar(sonde_dict['iwv'][mosleg_idx[k]], hatpro_dict['prw_mean_sonde'][mosleg_idx[k]], 
							yerr=hatpro_dict['prw_stddev_sonde'][mosleg_idx[k]],
							ecolor=c_H, elinewidth=1.2, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
							linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='HATPRO')

		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_hatson = np.argwhere(~np.isnan(hatpro_dict['prw_mean_sonde'][mosleg_idx[k]]) &
							~np.isnan(sonde_dict['iwv'][mosleg_idx[k]])).flatten()
		y_fit = hatpro_dict['prw_mean_sonde'][mosleg_idx[k][nonnan_hatson]]
		x_fit = sonde_dict['iwv'][mosleg_idx[k][nonnan_hatson]]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = ax1[k].plot(axlim[k], a*axlim[k] + b, color=c_H, linewidth=0.75, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		ax1[k].plot(axlim[k], axlim[k], color=(0,0,0), linewidth=0.75, alpha=0.5, label="Theoretical perfect fit")

		# add statistics:
		ax1[k].text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(N_sondes_mosleg_hat[k], 
				np.nanmean(np.concatenate((sonde_dict['iwv'][mosleg_idx[k]], hatpro_dict['prw_mean_sonde'][mosleg_idx[k]]), axis=0)),
				bias_mosleg_hat[k], rmse_mosleg_hat[k], std_mosleg_hat[k], R_mosleg_hat[k]),
				horizontalalignment='right', verticalalignment='bottom', transform=ax1[k].transAxes, fontsize=fs-4)


		# set axis limits and labels:
		ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		ax1[k].set_xlim(left=axlim[k][0], right=axlim[k][1])

		ax1[k].tick_params(axis='both', labelsize=fs-4)

		ax1[k].set_aspect('equal')


		ax1[k].minorticks_on()
		ax1[k].set_yticks(ax1[k].get_xticks())
		ax1[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		ax1[k].set_xlim(left=axlim[k][0], right=axlim[k][1])

		if k%3 == 0: ax1[k].set_ylabel("IWV$_{\mathrm{HATPRO}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		if k >= 3: ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

		# ax1[k].text(0.01, 0.99, MOSAiC_Legs_Text[k], verticalalignment='top', horizontalalignment='left',
					# transform=ax1[k].transAxes, fontweight='bold', fontsize=fs-4)
		MOS_leg_keys = list(MOSAiC_legs.keys())
		ax1[k].text(0.01, 0.99, dt.datetime.strftime(MOSAiC_legs[MOS_leg_keys[k]][0], "%Y-%m-%d") + " - " +
					dt.datetime.strftime(MOSAiC_legs[MOS_leg_keys[k]][1], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
					transform=ax1[k].transAxes, fontsize=fs-4)
		ax1[k].set_title(MOSAiC_Legs_Text[k], fontsize=fs, fontweight='bold')

	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()

	# use the lower left panel for additional information (legend; MOSAiC leg time)
	ax1[5].axis('off')

	ax1[5].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs)

	if save_figures:
		iwv_name_base = "IWV_scatterplot_MOS_total_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_hatpro_vs_sonde"
		else:
			iwv_name_suffix_def = "_hatpro_vs_sonde"
		fig6.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# plt.show()
	else:
		plt.show()



	# IWV scatterplot for mosaic legs: now MiRAC-P:
	fig7, ax1 = plt.subplots(2,3)
	fig7.set_size_inches(16,10)
	ax1 = ax1.flatten()

	# handle axis limits:
	if scatplot_fix_axlims:
		axlim = np.asarray([[0, 10], [0, 8], [0, 20], [0, 35], [0, 25]])

	else:
		axlim = np.zeros((5,2))
		for k in range(n_mosleg):
			if mosleg_idx[k].size > 0:
				axlim[k] = [0, np.ceil(np.nanmax(np.concatenate((sonde_dict['iwv'][mosleg_idx[k]],
								mirac_dict['prw_mean_sonde'][mosleg_idx[k]]), axis=0))+2)]
			else:
				axlim[k] = [0, 1]

	for k in range(n_mosleg):	# each panel represents one MOSAiC leg
		
		ax1[k].errorbar(sonde_dict['iwv'][mosleg_idx[k]], mirac_dict['prw_mean_sonde'][mosleg_idx[k]], 
							yerr=mirac_dict['prw_stddev_sonde'][mosleg_idx[k]],
							ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
							linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P')

		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		nonnan_mirson = np.argwhere(~np.isnan(mirac_dict['prw_mean_sonde'][mosleg_idx[k]]) &
							~np.isnan(sonde_dict['iwv'][mosleg_idx[k]])).flatten()
		y_fit = mirac_dict['prw_mean_sonde'][mosleg_idx[k][nonnan_mirson]]
		x_fit = sonde_dict['iwv'][mosleg_idx[k][nonnan_mirson]]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = ax1[k].plot(axlim[k], a*axlim[k] + b, color=c_M, linewidth=0.75, label="Best fit")

		# plot a line for orientation which would represent a perfect fit:
		ax1[k].plot(axlim[k], axlim[k], color=(0,0,0), linewidth=0.75, alpha=0.5, label="Theoretical perfect fit")

		# add statistics:
		ax1[k].text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(N_sondes_mosleg_mir[k], 
				np.nanmean(np.concatenate((sonde_dict['iwv'][mosleg_idx[k]], mirac_dict['prw_mean_sonde'][mosleg_idx[k]]), axis=0)),
				bias_mosleg_mir[k], rmse_mosleg_mir[k], std_mosleg_mir[k], R_mosleg_mir[k]),
				horizontalalignment='right', verticalalignment='bottom', transform=ax1[k].transAxes, fontsize=fs-4)


		# set axis limits and labels:
		ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		ax1[k].set_xlim(left=axlim[k][0], right=axlim[k][1])

		ax1[k].tick_params(axis='both', labelsize=fs-4)

		ax1[k].set_aspect('equal')


		ax1[k].minorticks_on()
		ax1[k].set_yticks(ax1[k].get_xticks())
		ax1[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		ax1[k].set_xlim(left=axlim[k][0], right=axlim[k][1])

		if k%3 == 0: ax1[k].set_ylabel("IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		if k >= 3: ax1[k].set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

		# ax1[k].text(0.01, 0.99, MOSAiC_Legs_Text[k], verticalalignment='top', horizontalalignment='left',
					# transform=ax1[k].transAxes, fontweight='bold', fontsize=fs-4)
		MOS_leg_keys = list(MOSAiC_legs.keys())
		ax1[k].text(0.01, 0.99, dt.datetime.strftime(MOSAiC_legs[MOS_leg_keys[k]][0], "%Y-%m-%d") + " - " +
					dt.datetime.strftime(MOSAiC_legs[MOS_leg_keys[k]][1], "%Y-%m-%d"), verticalalignment='top', horizontalalignment='left',
					transform=ax1[k].transAxes, fontsize=fs-4)
		ax1[k].set_title(MOSAiC_Legs_Text[k], fontsize=fs, fontweight='bold')

	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()

	# use the lower left panel for additional information (legend; MOSAiC leg time)
	ax1[5].axis('off')

	ax1[5].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs)

	if save_figures:
		iwv_name_base = "IWV_scatterplot_MOS_total_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_mirac_vs_sonde"
		else:
			iwv_name_suffix_def = "_mirac_vs_sonde"
			
		fig7.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
		# plt.show()
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
		axlim = [0, np.ceil(np.nanmax(np.concatenate((sonde_dict['iwv'],
							hatpro_dict['prw_mean_sonde'], mirac_dict['prw_mean_sonde']), axis=0))+2)]


	# compute retrieval statistics:
	ret_stat_dict_hat = compute_retrieval_statistics(sonde_dict['iwv'],
													hatpro_dict['prw_mean_sonde'],
													compute_stddev=True)
	ret_stat_dict_mir = compute_retrieval_statistics(sonde_dict['iwv'],
													mirac_dict['prw_mean_sonde'],
													compute_stddev=True)



	# plotting:
	ax1.errorbar(sonde_dict['iwv'], hatpro_dict['prw_mean_sonde'], 
						yerr=hatpro_dict['prw_stddev_sonde'],
						ecolor=c_H, elinewidth=1.6, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='HATPRO')
	ax2.errorbar(sonde_dict['iwv'], mirac_dict['prw_mean_sonde'], 
						yerr=mirac_dict['prw_stddev_sonde'],
						ecolor=c_M, elinewidth=1.6, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='MiRAC-P')


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	nonnan_hatson = np.argwhere(~np.isnan(hatpro_dict['prw_mean_sonde']) &
						~np.isnan(sonde_dict['iwv'])).flatten()
	y_fit = hatpro_dict['prw_mean_sonde'][nonnan_hatson]
	x_fit = sonde_dict['iwv'][nonnan_hatson]

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
						~np.isnan(sonde_dict['iwv'])).flatten()
	y_fit = mirac_dict['prw_mean_sonde'][nonnan_mirson]
	x_fit = sonde_dict['iwv'][nonnan_mirson]

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
			np.nanmean(np.concatenate((sonde_dict['iwv'], hatpro_dict['prw_mean_sonde']), axis=0)),
			ret_stat_dict_hat['bias'], ret_stat_dict_hat['rmse'], ret_stat_dict_hat['stddev'], ret_stat_dict_hat['R']),
			horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fs-2)
	ax2.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(ret_stat_dict_mir['N'], 
			np.nanmean(np.concatenate((sonde_dict['iwv'], mirac_dict['prw_mean_sonde']), axis=0)),
			ret_stat_dict_mir['bias'], ret_stat_dict_mir['rmse'], ret_stat_dict_mir['stddev'], ret_stat_dict_mir['R']),
			horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, fontsize=fs-2)


	# Legends:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs)
	leg_handles, leg_labels = ax2.get_legend_handles_labels()
	ax2.legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs)


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
	if with_titles: ax1.set_title("IWV comparison of HATPRO and\nradiosonde during MOSAiC", fontsize=fs)

	ax2.set_ylabel("IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax2.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles: ax2.set_title("IWV comparison of MiRAC-P and\nradiosonde during MOSAiC", fontsize=fs)


	# Limit axis spacing:
	plt.subplots_adjust(hspace=0.15)			# removes space between subplots


	if save_figures:
		iwv_name_base = "IWV_scatterplot_mosaic_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_hatpro_mirac_vs_sonde"
		else:
			iwv_name_suffix_def = "_hatpro_mirac_vs_sonde"
		fig6.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')
		# plt.show()

	elif save_figures_eps:
		iwv_name_base = "IWV_scatterplot_mosaic_"
		if scatplot_fix_axlims:
			iwv_name_suffix_def = "_fixax_hatpro_mirac_vs_sonde"
		else:
			iwv_name_suffix_def = "_hatpro_mirac_vs_sonde"
		fig6.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_IWV_overview_boxplot and (plot_option == 5):
	# IWV box plot MiRAC-P, HATPRO and radiosonde (and later: + ARM)
	fig11, ax1 = plt.subplots(1,2)
	fig11.set_size_inches(22,10)

	axlim = [0, 30]		# axis limits for IWV plot in kg m^-2

	labels_bp = ['Oct 19', 'Nov 19', 'Dec 19', 'Jan 20', 'Feb 20', 'Mar 20',
				'Apr 20', 'May 20', 'Jun 20', 'Jul 20', 'Aug 20', 'Sep 20']

	whis_lims = [1,99]

	def make_boxplot_great_again(bp, col):	# change color and set linewidth to 1.5
		plt.setp(bp['boxes'], color=col, linewidth=1.5)
		plt.setp(bp['whiskers'], color=col, linewidth=1.5)
		plt.setp(bp['caps'], color=col, linewidth=1.5)
		plt.setp(bp['medians'], color=col, linewidth=1.5)

	if include_JR_IWV:
		n_ins_plus = 5					# number of instruments in the box plot + 1
		n_groups = len(IWV_RS_grouped)		# e.g. number of months in data set
		n_boxes = n_ins_plus*n_groups

		# positions of boxes for each instrument:
		pos_h = [n_ins_plus*k + 2 for k in range(n_groups)]
		pos_m = [n_ins_plus*k + 3 for k in range(n_groups)]
		pos_s = [n_ins_plus*k + 1 for k in range(n_groups)]
		pos_jr = [n_ins_plus*k + 4 for k in range(n_groups)]
		pos_label = [n_ins_plus*k + n_ins_plus/2 for k in range(n_groups)]

		# BOXPLOT: whis = [1, 99] # to have 1st and 99th percentile as whiskers
		# Radiosonde
		bp_plot_s = ax1[0].boxplot(IWV_RS_grouped, sym='', positions=pos_s, whis=whis_lims, widths=0.5)
		# HATPRO
		bp_plot_h = ax1[0].boxplot(IWV_MWR_grouped_hatpro, sym='', positions=pos_h, whis=whis_lims, widths=0.5)
		# MiRAC-P
		bp_plot_m = ax1[0].boxplot(IWV_MWR_grouped_mirac, sym='', positions=pos_m, whis=whis_lims, widths=0.5)
		# Satellite data from Janna Rueckert:
		bp_plot_jr = ax1[0].boxplot(IWV_JR_grouped, sym='', positions=pos_jr, whis=whis_lims, widths=0.5)

		make_boxplot_great_again(bp_plot_s, col=c_RS)
		make_boxplot_great_again(bp_plot_h, col=c_H)
		make_boxplot_great_again(bp_plot_m, col=c_M)
		make_boxplot_great_again(bp_plot_jr, col=c_JR)

		# create dummy plots for legend:
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_RS, label='Radiosonde')
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_H, label='HATPRO')
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_M, label='MiRAC-P')
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_JR, label='AMSR2 - Janna Rückert')

	else:
		n_ins_plus = 4					# number of instruments in the box plot + 1
		n_groups = len(IWV_RS_grouped)		# e.g. number of months in data set
		n_boxes = n_ins_plus*n_groups

		# positions of boxes for each instrument:
		pos_h = [n_ins_plus*k + 2 for k in range(n_groups)]
		pos_m = [n_ins_plus*k + 3 for k in range(n_groups)]
		pos_s = [n_ins_plus*k + 1 for k in range(n_groups)]

		# BOXPLOT: whis = [5, 95] # to have 5th and 95th percentile as whiskers
		# Radiosonde
		bp_plot_s = ax1[0].boxplot(IWV_RS_grouped, sym='', positions=pos_s, whis=whis_lims, widths=0.5)
		# HATPRO
		bp_plot_h = ax1[0].boxplot(IWV_MWR_grouped_hatpro, sym='', positions=pos_h, whis=whis_lims, widths=0.5)
		# MiRAC-P
		bp_plot_m = ax1[0].boxplot(IWV_MWR_grouped_mirac, sym='', positions=pos_m, whis=whis_lims, widths=0.5)

		make_boxplot_great_again(bp_plot_s, col=c_RS)
		make_boxplot_great_again(bp_plot_h, col=c_H)
		make_boxplot_great_again(bp_plot_m, col=c_M)

		# create dummy plots for legend:
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_RS, label='Radiosonde')
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_H, label='HATPRO')
		ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=c_M, label='MiRAC-P')

	# # legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1[0].set_title("Integrated Water Vapour (IWV) during MOSAiC: October 2019 - October 2020", fontsize=fs)

	if include_JR_IWV: ax1[0].set_xticks(pos_label)
	else: ax1[0].set_xticks(pos_h)
	ax1[0].set_xticklabels(labels_bp)

	ax1[0].tick_params(axis='both', labelsize=fs-2)

	ax1[0].grid(which='major', axis='y', color=(0.5,0.5,0.5), alpha=0.5)

	# hide grid lines when box plot is above:
	ax1[0].set_axisbelow(True)


	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	if save_figures:
		iwv_name_base = "IWV_overview_boxplot_"
		instruments_string = "_hatpro_mirac_sonde_"
		if include_JR_IWV: instruments_string = instruments_string + "amsr2_"
		if considered_period != 'user':
			iwv_name_suffix_def = instruments_string + considered_period
		else:
			iwv_name_suffix_def = instruments_string + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig11.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_mwrson_diff_histogram and (plot_option in [2,6]):
	
	# HATPRO:
	# eventually separate into three IWV categories: IWV in [0,5), [5,10), [10,100)
	mask_bot = ((sonde_dict['iwv'] >= 0) & (sonde_dict['iwv'] < 5))
	mask_mid = ((sonde_dict['iwv'] >= 5) & (sonde_dict['iwv'] < 10))
	mask_top = ((sonde_dict['iwv'] >= 10) & (sonde_dict['iwv'] < 100))
	bias_bot = hatpro_dict['prw_mean_sonde'][mask_bot] - sonde_dict['iwv'][mask_bot]
	bias_mid = hatpro_dict['prw_mean_sonde'][mask_mid] - sonde_dict['iwv'][mask_mid]
	bias_top = hatpro_dict['prw_mean_sonde'][mask_top] - sonde_dict['iwv'][mask_top]
	bias_categorized = [bias_bot, bias_mid, bias_top]
	bias = hatpro_dict['prw_mean_sonde'] - sonde_dict['iwv']

	n_bias = float(len(bias))
	# weights_bias = np.ones_like(bias) / n_bias
	weights_bias_categorized = [np.ones_like(bias_bot) / n_bias, 
								np.ones_like(bias_mid) / n_bias, 
								np.ones_like(bias_top) / n_bias]


	fig1 = plt.figure(figsize=(16,10))
	ax1 = plt.axes()

	x_lim = [-6.25, 6.25]			# bias in mm or kg m^-2
	y_lim = [0, 0.60]

	# ax1.hist(bias, bins=np.arange(-6.25, 6.26, 0.5), weights=weights_bias, color=c_M, ec=(0,0,0))
	ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
				color=['blue', 'lightgrey', 'red'], ec=(0,0,0),
				label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])

	ax1.text(0.98, 0.96, "Min = %.2f\n Max = %.2f\n Mean = %.2f\n Median = %.2f"%(np.nanmin(bias),
				np.nanmax(bias), np.nanmean(bias), np.nanmedian(bias)), ha='right', va='top', 
				transform=ax1.transAxes, fontsize=fs-2)

	# legend:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	ax1.set_ylim(bottom=y_lim[0], top=y_lim[1])
	ax1.set_xlim(left=x_lim[0], right=x_lim[1])

	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1.set_xlabel("IWV$_{\mathrm{HATPRO}} - \mathrm{IWV}_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_ylabel("Frequency occurrence", fontsize = fs)
	the_title = ax1.set_title("IWV difference histogram (HATPRO - radiosonde)", fontsize=fs)
	the_title.set_position((0.5, 1.2))

	ax1.tick_params(axis='both', labelsize=fs-2)

	if save_figures:
		name_base = "IWV_diff_hatpro_sonde_histogram_"
		if considered_period != 'user':
			name_suffix_def = considered_period
		else:
			name_suffix_def = date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + name_base + iwv_name_option + name_suffix_def + ".png", dpi=400)
	else:
		plt.show()
		plt.close()


	# MiRAC-P:
	bias_bot = mirac_dict['prw_mean_sonde'][mask_bot] - sonde_dict['iwv'][mask_bot]
	bias_mid = mirac_dict['prw_mean_sonde'][mask_mid] - sonde_dict['iwv'][mask_mid]
	bias_top = mirac_dict['prw_mean_sonde'][mask_top] - sonde_dict['iwv'][mask_top]
	bias_categorized = [bias_bot, bias_mid, bias_top]
	bias = mirac_dict['prw_mean_sonde'] - sonde_dict['iwv']

	n_bias = float(len(bias))
	# weights_bias = np.ones_like(bias) / n_bias
	weights_bias_categorized = [np.ones_like(bias_bot) / n_bias, 
								np.ones_like(bias_mid) / n_bias, 
								np.ones_like(bias_top) / n_bias]


	fig1 = plt.figure(figsize=(16,10))
	ax1 = plt.axes()

	# ax1.hist(bias, bins=np.arange(-6.25, 6.26, 0.5), weights=weights_bias, color=c_M, ec=(0,0,0))
	ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
				color=['blue', 'lightgrey', 'red'], ec=(0,0,0),
				label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])

	ax1.text(0.98, 0.96, "Min = %.2f\n Max = %.2f\n Mean = %.2f\n Median = %.2f"%(np.nanmin(bias),
				np.nanmax(bias), np.nanmean(bias), np.nanmedian(bias)), ha='right', va='top', 
				transform=ax1.transAxes, fontsize=fs-2)

	# legend:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	ax1.set_ylim(bottom=y_lim[0], top=y_lim[1])
	ax1.set_xlim(left=x_lim[0], right=x_lim[1])

	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1.set_xlabel("IWV$_{\mathrm{MiRAC-P}} - \mathrm{IWV}_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_ylabel("Frequency occurrence", fontsize = fs)
	the_title = ax1.set_title("IWV difference histogram (MiRAC-P - radiosonde)", fontsize=fs)
	the_title.set_position((0.5, 1.2))

	ax1.tick_params(axis='both', labelsize=fs-2)

	if save_figures:
		name_base = "IWV_diff_mirac_sonde_histogram_"
		if considered_period != 'user':
			name_suffix_def = considered_period
		else:
			name_suffix_def = date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + name_base + iwv_name_option + name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_histogram and plot_option == 5:

	# IWV histogram (frequency occurrence) MiRAC-P, HATPRO and radiosonde
	# top plot: HATPRO and Radiosonde; bottom plot: MiRAC-P and Radiosonde
	fig11, ax1 = plt.subplots(2,2)
	fig11.set_size_inches(22,10)

	# Max. and min.IWV:
	min_IWV_mir = np.nanmin(mirac_dict['IWV_master'][outside_eez['mirac']])
	max_IWV_mir = np.nanmax(mirac_dict['IWV_master'][outside_eez['mirac']])
	min_IWV_hat = np.nanmin(hatpro_dict['prw_master'][outside_eez['hatpro']])
	max_IWV_hat = np.nanmax(hatpro_dict['prw_master'][outside_eez['hatpro']])
	min_IWV_sonde = np.nanmin(sonde_dict['iwv'][outside_eez['sonde']])
	max_IWV_sonde = np.nanmax(sonde_dict['iwv'][outside_eez['sonde']])

	axlim = [0, 30]		# axis limits for IWV axis (x) in kg m^-2
	plot_bins = np.arange(axlim[0], axlim[1])


	# Radiosonde:
	ax1[0,0].hist(sonde_dict['iwv'][outside_eez['sonde']], bins=plot_bins, density=True, color=c_RS,
				alpha=0.5, label='Radiosonde')
	# HATPRO:
	ax1[0,0].hist(hatpro_dict['prw_master'][outside_eez['hatpro']], bins=plot_bins, density=True, color=c_H,
				alpha=0.5, label='HATPRO')


	# legend:
	leg_handles, leg_labels = ax1[0,0].get_legend_handles_labels()
	ax1[0,0].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	# ax1[0].set_ylim(bottom=0, top=0.25)
	ax1[0,0].set_xlim(left=axlim[0], right=axlim[1])
	# # # # ax1[0,0].set_ylabel("Frequency occurrence (%)", fontsize=fs)
	ax1[0,0].set_title("Integrated Water Vapour (IWV) distribution during MOSAiC", fontsize=fs)

	# y axis shall display percentage:
	ax1[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))

	# x axis with minor ticks:
	ax1[0,0].xaxis.set_minor_locator(AutoMinorLocator())
	ax1[0,0].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1[0,0].tick_params(axis='both', labelsize=fs-2)

	ax1_pos = ax1[0,0].get_position().bounds
	ax1[0,0].set_position([ax1_pos[0], ax1_pos[1], 1.6*ax1_pos[2], ax1_pos[3]])


	# Lower plot:
	# Radiosonde:
	ax1[1,0].hist(sonde_dict['iwv'][outside_eez['sonde']], bins=plot_bins, density=True, color=c_RS,
				alpha=0.5, label='Radiosonde')
	# MiRAC-P:
	ax1[1,0].hist(mirac_dict['IWV_master'][outside_eez['mirac']], bins=plot_bins, density=True, color=c_M,
				alpha=0.33, label='MiRAC-P')

	# legend:
	leg_handles, leg_labels = ax1[1,0].get_legend_handles_labels()
	ax1[1,0].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[1,0].set_xlim(left=axlim[0], right=axlim[1])
	# # # ax1[1,0].set_ylabel("Frequency occurrence (%)", fontsize=fs)
	ax1[1,0].set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	# y axis shall display percentage:
	ax1[1,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))

	# x axis with minor ticks:
	ax1[1,0].xaxis.set_minor_locator(AutoMinorLocator())
	ax1[1,0].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1[1,0].tick_params(axis='both', labelsize=fs-2)

	ax1_pos = ax1[1,0].get_position().bounds
	ax1[1,0].set_position([ax1_pos[0], ax1_pos[1], 1.6*ax1_pos[2], ax1_pos[3]])



	# create the text box with add. info:
	ax1[0,1].axis('off')
	ax1[1,1].axis('off')

	ax1[0,1].text(0.05, 1.0, "Min; Max IWV:\n HATPRO: %.2f; %.2f\n MiRAC-P: %.2f; %.2f\n Radiosonde: %.2f; %.2f"%(
			min_IWV_hat, max_IWV_hat, min_IWV_mir, max_IWV_mir, min_IWV_sonde, max_IWV_sonde),
			ha='left', va='top', transform=ax1[0,1].transAxes, fontsize=fs-4,
			bbox=dict(boxstyle='round', ec=(0,0,0), fc=(1,1,1), alpha=0.5))

	ax2_pos = ax1[0,1].get_position().bounds
	ax1[0,1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1], 0.4*ax2_pos[2], ax2_pos[3]])

	# y axis label in the middle:
	fig11.text(0.06, 0.5, "Frequency occurrence (%)", va='center', rotation='vertical', fontsize=fs)

	if save_figures:
		iwv_name_base = "IWV_overview_histogram_"
		instruments_string = "_hatpro_mirac_sonde_"
		if considered_period != 'user':
			iwv_name_suffix_def = instruments_string + considered_period
		else:
			iwv_name_suffix_def = instruments_string + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig11.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_LWP_time_series:

	# # # # For Michael Lonardi: Compute Average LWP and "mode":
	# # # hatpro_lwp_avg = 1000*np.nanmean(hatpro_dict['clwvi'][hatpro_dict['flag'] == 0.0])
	# # # mirac_lwp_avg = 1000*np.nanmean(mirac_dict['LWP'][mirac_dict['RF'] == 0.0])
	# # # arm_lwp_avg = 1000*np.nanmean(arm_dict['lwp'][arm_dict['lwp_flag'] == 0])

	# # # hatpro_lwp_mode = stats.mode(1000*hatpro_dict['clwvi'][hatpro_dict['flag'] == 0.0])[0][0]
	# # # mirac_lwp_mode = stats.mode(1000*mirac_dict['LWP'][mirac_dict['RF'] == 0.0])[0][0]
	# # # arm_lwp_mode = stats.mode(1000*arm_dict['lwp'][arm_dict['lwp_flag'] == 0])[0][0]

	# LWP time series (MiRAC-P), HATPRO
	fig8, ax1 = plt.subplots(1,2)
	fig8.set_size_inches(22,10)

	axlim = [-250, 1000]		# axis limits for LWP plot in g m^-2

	# colours for the different instruments
	c_hatpro = c_H
	c_mirac = c_M
	c_arm = c_ARM

	if plot_option == 0:

		MIRAC_LWP_plot = ax1[0].plot(mirac_dict['datetime'], 1000*mirac_dict['LWP'],
									color=c_mirac, linewidth=1.0, label="MiRAC-P")

		HATPRO_LWP_plot = ax1[0].plot(hatpro_dict['datetime'], 1000*hatpro_dict['clwvi'], 
									color=c_hatpro, linewidth=1.0, label='HATPRO')

		ARM_LWP_plot = ax1[0].plot(arm_dict['datetime'], 1000*arm_dict['lwp'],
									color=c_arm, linewidth=1.0, label='ARM', alpha=0.5)
	elif plot_option in [1, 2]:

		hatpro_flag_idx = np.where(hatpro_dict['flag'] == 0.0)[0]
		# MIRAC_LWP_plot = ax1[0].plot(mirac_dict['datetime'][mirac_dict['RF'] == 0.0], 1000*mirac_dict['LWP'][mirac_dict['RF'] == 0.0],
										# color=c_mirac, linewidth=1.0, label="MiRAC-P")
		HATPRO_LWP_plot = ax1[0].plot(hatpro_dict['datetime'][hatpro_flag_idx], 1000*hatpro_dict['clwvi'][hatpro_flag_idx], 
										color=c_hatpro, linewidth=1.0, label='HATPRO')

		# ARM_LWP_plot = ax1[0].plot(arm_dict['datetime'][arm_dict['lwp_flag']==0], 1000*arm_dict['lwp'][arm_dict['lwp_flag']==0],
									# color=c_arm, linewidth=1.0, label='ARM', alpha=0.5)

		# # # # Include text box with avg and mode values for Michael Lonardi:
		# # # ax1[0].text(0.985, 0.97, "HATPRO: Mean = %.2f $\mathrm{g}\,\mathrm{m}^{-2}$ \nMode = %.2f $\mathrm{g}\,\mathrm{m}^{-2}$ \n"%(hatpro_lwp_avg, hatpro_lwp_mode) + 
						# # # "ARM: Mean = %.2f $\mathrm{g}\,\mathrm{m}^{-2}$ \nMode = %.2f $\mathrm{g}\,\mathrm{m}^{-2}$"%(arm_lwp_avg, arm_lwp_mode),
						# # # horizontalalignment='right', verticalalignment='top', transform=ax1[0].transAxes, fontsize=fs-2,
						# # # bbox=dict(boxstyle='round', ec=(0,0,0), fc=(1,1,1), alpha=0.5))

	elif plot_option in [3, 4]:
		# MIRAC_LWP_plot = ax1[0].plot(master_time_dt, 1000*mirac_dict['LWP_master'],
										# color=c_mirac, linewidth=1.0, marker='.', markersize=1.5,
									# markerfacecolor=c_mirac, markeredgecolor=c_mirac, 
									# markeredgewidth=0.5, label="MiRAC-P")
		HATPRO_LWP_plot = ax1[0].plot(master_time_dt, 1000*hatpro_dict['clwvi_master'], 
										color=c_hatpro, linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=c_hatpro, markeredgecolor=c_hatpro, 
									markeredgewidth=0.5, label='HATPRO')
		ARM_IWV_plot = ax1[0].plot(master_time_dt, 1000*arm_dict['lwp_master'], linestyle='none', marker='.', linewidth=0.5,
									markersize=7.5, markerfacecolor=c_arm, markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, color=c_arm, alpha=0.4, label='ARM')

	elif plot_option in [5]:
		ARM_IWV_plot = ax1[0].plot(master_time_dt[outside_eez], 1000*arm_dict['lwp_master'][outside_eez], 
									linestyle='none', marker='.', linewidth=0.5,
									markersize=4.5, markerfacecolor=c_arm, markeredgecolor=c_arm, 
									markeredgewidth=0.5, color=c_arm, alpha=0.4)
		ARM_IWV_plot_dummy = ax1[0].plot([np.nan, np.nan], [np.nan, np.nan],
									linestyle='none', marker='.', linewidth=0.5,
									markersize=8.5, markerfacecolor=c_arm, markeredgecolor=c_arm, 
									markeredgewidth=0.5, color=c_arm, alpha=0.8, label='ARM')
		HATPRO_LWP_plot = ax1[0].plot(master_time_dt[outside_eez], 1000*hatpro_dict['clwvi_master'][outside_eez], 
										color=c_H, linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=c_H, markeredgecolor=c_H, 
									markeredgewidth=0.5, label='HATPRO')
		MIRAC_LWP_plot = ax1[0].plot(master_time_dt[outside_eez], 1000*mirac_dict['LWP_master'][outside_eez],
										color=c_M, linewidth=1.0, marker='.', markersize=1.5,
									markerfacecolor=c_M, markeredgecolor=c_M, 
									markeredgewidth=0.5, label="MiRAC-P")


	# # # for ct_hatpro in calibration_times_HATPRO: 
		# # # if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# # # ax1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_hatpro, linestyle='dashed', linewidth=2)
	# # # for ct_mirac in calibration_times_MiRAC:
		# # # if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# # # ax1[0].plot([ct_mirac, ct_mirac], axlim, color=c_mirac, linestyle='dashed', linewidth=2)
	# # # for leg in [*MOSAiC_legs.keys()]:
		# # # if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			# # # ax1[0].plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		# # # if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			# # # ax1[0].plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)

	# legend:
	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=0.5)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
	de_title = ax1[0].set_title("Liquid Water Path (LWP), " + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_range_end, "%Y-%m-%d"), fontsize=fs)
	de_title.set_position((0.5, 1.05))

	if datetick_auto:
		fig8.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1[0].tick_params(axis='x', labelsize=fs-3, labelrotation=90)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	# # # # include texts indicating the legs of MOSAiC:
	# # # for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		# # # if x_pos >= date_range_start and x_pos <= date_range_end:
			# # # ax1[0].text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs+2, ha='center', va='bottom')

	ax1_pos = ax1[0].get_position().bounds
	# ax1[0].set_position([ax1_pos[0], ax1_pos[1], 0.80*ax1_pos[2], ax1_pos[3]])
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	# # # # add dummy lines for the legend:
	# # # ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_hatpro, linestyle='dashed', linewidth=2,
				# # # label="$\\bf{HATPRO}$")
	# # # for ct_hatpro in calibration_times_HATPRO: 
		# # # if ct_hatpro <= date_range_end and ct_hatpro >= date_range_start:
			# # # # same color as background to be invisible
			# # # ax1[1].plot([ct_hatpro, ct_hatpro], axlim, color=fig8.get_facecolor(),
				# # # label=dt.datetime.strftime(ct_hatpro, "%d.%m.%Y, %H:%M UTC"))
	# # # ax1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_mirac, linestyle='dashed', linewidth=2,
				# # # label="$\\bf{MiRAC-P}$")
	# # # for ct_mirac in calibration_times_MiRAC: 
		# # # if ct_mirac <= date_range_end and ct_mirac >= date_range_start:
			# # # # same color as background to be invisible
			# # # ax1[1].plot([ct_mirac, ct_mirac], axlim, color=fig8.get_facecolor(),
				# # # label=dt.datetime.strftime(ct_mirac, "%d.%m.%Y, %H:%M UTC"))

	# # # cal_ti_handles, cal_ti_labels = ax1[1].get_legend_handles_labels()
	# # # lo = ax1[1].legend(handles=cal_ti_handles, labels=cal_ti_labels, loc='upper left', 
						# # # fontsize=fs+2, title="Calibration")		 # bbox_to_anchor=(0.0,1.0), 
	# # # lo.get_title().set_fontsize(fs+2)
	# # # lo.get_title().set_fontweight('bold')

	if save_figures:
		iwv_name_base = "LWP_time_series_total_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_arm_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_arm_mirac_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig8.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_LWP_overview_boxplot and plot_option == 5:
	# LWP box plot HATPRO (and later: + ARM ?)
	fig11, ax1 = plt.subplots(1,2)
	fig11.set_size_inches(22,10)

	axlim = [0, 750]		# axis limits for LWP plot in g m^-2

	labels_bp = ['Oct 19', 'Nov 19', 'Dec 19', 'Jan 20', 'Feb 20', 'Mar 20',
				'Apr 20', 'May 20', 'Jun 20', 'Jul 20', 'Aug 20', 'Sep 20']

	whis_lims = [1,99]

	def make_boxplot_great_again(bp, col):	# change color and set linewidth to 1.5
		plt.setp(bp['boxes'], color=col, linewidth=1.5)
		plt.setp(bp['whiskers'], color=col, linewidth=1.5)
		plt.setp(bp['caps'], color=col, linewidth=1.5)
		plt.setp(bp['medians'], color=col, linewidth=1.5)


	n_ins_plus = 4					# number of instruments in the box plot + 1
	n_groups = len(LWP_MWR_DS_grouped)		# e.g. number of months in data set
	n_boxes = n_ins_plus*n_groups

	# positions of boxes for each instrument:
	pos_a = [n_ins_plus*k + 1 for k in range(n_groups)]		# ARM
	pos_h = [n_ins_plus*k + 2 for k in range(n_groups)]		# HATPRO
	pos_m = [n_ins_plus*k + 3 for k in range(n_groups)]		# MiRAC-P

	# BOXPLOT: whis = [5, 95] # to have 5th and 95th percentile as whiskers
	# in g m^-2 --> requires multiplication by 1000
	LWP_MWR_grouped_hatpro_plot = [lwp_hatpro*1000 for lwp_hatpro in LWP_MWR_grouped_hatpro]
	LWP_MWR_grouped_mirac_plot = [lwp_mirac*1000 for lwp_mirac in LWP_MWR_grouped_mirac]
	LWP_MWR_grouped_arm_plot = [lwp_arm*1000 for lwp_arm in LWP_MWR_grouped_arm]
	# ARM:
	bp_plot_a = ax1[0].boxplot(LWP_MWR_grouped_arm_plot, sym='', positions=pos_a, whis=whis_lims, widths=0.5)
	# HATPRO:
	bp_plot_h = ax1[0].boxplot(LWP_MWR_grouped_hatpro_plot, sym='', positions=pos_h, whis=whis_lims, widths=0.5)
	# MiRAC-P:
	bp_plot_m = ax1[0].boxplot(LWP_MWR_grouped_mirac_plot, sym='', positions=pos_m, whis=whis_lims, widths=0.5)

	make_boxplot_great_again(bp_plot_a, col=c_ARM)
	make_boxplot_great_again(bp_plot_h, col=c_H)
	make_boxplot_great_again(bp_plot_m, col=c_M)

	# create dummy plots for legend:
	ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=3.0, color=c_ARM, label='ARM')
	ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=3.0, color=c_H, label='HATPRO')
	ax1[0].plot([np.nan, np.nan], [np.nan, np.nan], linewidth=3.0, color=c_M, label='MiRAC-P')

	# legend:
	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=axlim[0], top=axlim[1])
	ax1[0].set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1[0].set_title("Liquid Water Path (LWP) during MOSAiC: October 2019 - October 2020", fontsize=fs)

	ax1[0].set_xticks(pos_h)
	ax1[0].set_xticklabels(labels_bp)

	ax1[0].tick_params(axis='both', labelsize=fs-2)

	ax1[0].grid(which='major', axis='y', color=(0.5,0.5,0.5), alpha=0.5)

	# hide grid lines when box plot is above:
	ax1[0].set_axisbelow(True)


	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	if save_figures:
		iwv_name_base = "LWP_overview_boxplot_"
		instruments_string = "_hatpro_mirac_arm_"
		if considered_period != 'user':
			iwv_name_suffix_def = instruments_string + considered_period
		else:
			iwv_name_suffix_def = instruments_string + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig11.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_LWP_time_series_daily_avg and plot_option == 6:

	# only consider steps outside EEZs
	hatpro_dict['time'] = hatpro_dict['time'][outside_eez['hatpro']]
	hatpro_dict['time_npdt'] = hatpro_dict['time_npdt'][outside_eez['hatpro']]
	hatpro_dict['clwvi'] = hatpro_dict['clwvi'][outside_eez['hatpro']]

	"""
	# compute daily averages manually:
	n_days = (date_range_end - date_range_start).days
	hatpro_dict['clwvi_daily'] = np.full((n_days+1,), np.nan)
	hatpro_dict['time_daily_npdt'] = np.full((n_days+1,), hatpro_dict['time'][0])

	# manually compute daily clwvi
	now_date = date_range_start
	kk = 0
	while now_date <= date_range_end:
		# boundaries:
		now_date_0 = datetime_to_epochtime(now_date)
		now_date_1 = datetime_to_epochtime(now_date + dt.timedelta(days=1))

		# find indices that mark the current day:
		ii = np.where((hatpro_dict['time'] >= now_date_0) & (hatpro_dict['time'] < now_date_1))[0]
		hatpro_dict['time_daily_npdt'][kk] = now_date_0

		if len(ii) > 0:
			hatpro_dict['clwvi_daily'][kk] = np.nanmean(hatpro_dict['clwvi'][ii])

		kk += 1
		now_date += dt.timedelta(days=1)

	hatpro_dict['time_daily_npdt'] = hatpro_dict['time_daily_npdt'].astype("datetime64[s]")
	pdb.set_trace()
	"""

	hatpro_clwvi = xr.DataArray(hatpro_dict['clwvi'], coords=[hatpro_dict['time_npdt']], dims=["time"])
	hatpro_clwvi_daily = hatpro_clwvi.resample(time="1D").mean()

	# LWP time series of daily averages of HATPRO
	fig1 = plt.figure(figsize=(16,8))
	ax1 = plt.axes()

	if include_data_availability:
		axlim = [-100, 400]		# axis limits for LWP plot in g m^-2

		# compute data availability: cycle through all days
		date_start_npdt = np.datetime64(date_start + "T00:00:00")
		date_end_npdt = np.datetime64(date_end + "T00:00:00")
		now_date_npdt = date_start_npdt
		n_days = (date_end_npdt - date_start_npdt).astype("timedelta64[D]").astype("int32")+1
		kk = 0
		date_array = np.full((n_days,), now_date_npdt)
		hatpro_data_flagged = np.zeros((n_days,), dtype="float32")		# fraction of available data
		while now_date_npdt <= date_end_npdt:
			# find indices of HATPRO in the range [now_date_npdt:now_date_npdt + 1 day)
			hatproix = np.where((hatpro_dict['time_npdt'] >= now_date_npdt) & 
								(hatpro_dict['time_npdt'] < now_date_npdt + np.timedelta64(1, "D")))[0]

			n_tot = len(hatproix)
			n_flagged = np.count_nonzero(np.isnan(hatpro_dict['clwvi'][hatproix]))

			date_array[kk] = now_date_npdt
			if n_tot > 0:
				hatpro_data_flagged[kk] = n_flagged / n_tot		# fraction of flagged data
			else:
				hatpro_data_flagged[kk] = 1.0			# then 100 % is flagged
			
			kk += 1
			now_date_npdt += np.timedelta64(1, "D")
		
	else:
		axlim = [-50, 450]		# axis limits for LWP plot in g m^-2


	HATPRO_IWV_plot = ax1.plot(hatpro_clwvi_daily.time.values, 1000*hatpro_clwvi_daily.values,
									color=c_H, linewidth=1.5)

	if include_data_availability:
		# scale the data availability to 0 - -100: 0 = 0% flagged; -100 = 100 % flagged:
		hatpro_data_flagged *= -100
		ax1.bar(date_array, hatpro_data_flagged, width=1.0, color=(0.9,0,0),
					label="HATPRO data flagged")

		lh, ll = ax1.get_legend_handles_labels()
		ax1.legend(handles=lh, labels=ll, loc="upper left", fontsize=fs)


	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1.plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1.plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)


	# set axis limits and labels:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax1.set_xlim(left=dt.datetime(2019,10,22,6,0), right=date_range_end)
	ax1.xaxis.set_major_formatter(dt_fmt)
	ax1.set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
	the_title = ax1.set_title("Daily average Liquid Water Path (LWP) during MOSAiC (" + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")", fontsize=fs, pad=50)
	the_title.set_position((0.5, 1.2))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1.set_xticks(x_ticks_dt)
		# tick_dt = np.asarray([date_range_start + (k+1)*x_tick_delta for k in range(int(date_range_delta/x_tick_delta)+1)])
		# xlabels = [dt.datetime.strftime(xtdt, "%Y-%m-%d %H:%M") for xtdt in tick_dt]
		# ax1[0].set_xticklabels(xlabels, fontsize=fs-2, rotation=45, ha='right')
		ax1.tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	if include_data_availability:
		ytick_labels = [yt for yt in ax1.get_yticks()]
		ax1.yaxis.set_ticklabels(["100$\,\%$"] + ytick_labels[1:])

	ax1.tick_params(axis='y', labelsize=fs-2)

	ax1.grid(which='major', axis='both')

	ax1_pos = ax1.get_position().bounds
	ax1.set_position([ax1_pos[0], ax1_pos[1]+0.2*ax1_pos[3], ax1_pos[2], ax1_pos[3]*0.8])

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1.text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs, ha='center', va='bottom')

	if save_figures:
		iwv_name_base = "LWP_daily_mean_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_LWP_IWV_time_series_daily_avg and plot_option == 6:
	# only consider steps outside EEZs
	hatpro_clwvi = xr.DataArray(hatpro_dict['clwvi'][outside_eez['hatpro']], 
								coords=[hatpro_dict['time_npdt'][outside_eez['hatpro']]], 
								dims=["time"])
	hatpro_clwvi_daily = hatpro_clwvi.resample(time="1D").mean()

	hatpro_prw = xr.DataArray(hatpro_dict['prw'][outside_eez['hatpro']],
								coords=[hatpro_dict['time_npdt'][outside_eez['hatpro']]],
								dims=['time'])
	hatpro_prw = hatpro_prw.resample(time="1D").mean()

	# LWP time series of daily averages of HATPRO
	fig1 = plt.figure(figsize=(16,8))
	ax1 = plt.axes()

	if include_data_availability:
		axlim = [-100, 400]		# axis limits for LWP plot in g m^-2
		ax2lim = [-10, 40]		# axis limits for IWV plot in kg m-2

		# compute data availability: cycle through all days
		date_start_npdt = np.datetime64(date_start + "T00:00:00")
		date_end_npdt = np.datetime64(date_end + "T00:00:00")
		now_date_npdt = date_start_npdt
		n_days = (date_end_npdt - date_start_npdt).astype("timedelta64[D]").astype("int32")+1
		kk = 0
		date_array = np.full((n_days,), now_date_npdt)
		hatpro_data_flagged = np.zeros((n_days,), dtype="float32")		# fraction of available data
		while now_date_npdt <= date_end_npdt:
			# find indices of HATPRO in the range [now_date_npdt:now_date_npdt + 1 day)
			hatproix = np.where((hatpro_dict['time_npdt'] >= now_date_npdt) & 
								(hatpro_dict['time_npdt'] < now_date_npdt + np.timedelta64(1, "D")))[0]

			n_tot = len(hatproix)
			n_flagged = np.count_nonzero(np.isnan(hatpro_dict['prw'][hatproix]))

			date_array[kk] = now_date_npdt
			if n_tot > 0:
				hatpro_data_flagged[kk] = n_flagged / n_tot		# fraction of flagged data
			else:
				hatpro_data_flagged[kk] = 1.0			# then 100 % is flagged
			
			kk += 1
			now_date_npdt += np.timedelta64(1, "D")
		
	else:
		axlim = [-50, 450]		# axis limits for LWP plot in g m^-2
		ax2lim = [0, 35]		# axis limits for IWV plot in kg m-2


	HATPRO_LWP_plot = ax1.plot(hatpro_clwvi_daily.time.values, 1000*hatpro_clwvi_daily.values,
									color=(0,0,0), linewidth=1.5)

	ax2 = ax1.twinx()

	# dummy plot of LWP for legend
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=2.0, label="LWP")
	ax2.plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, linestyle='dashed', label="IWV")

	HATPRO_IWV_plot = ax2.plot(hatpro_prw.time.values, hatpro_prw.values, color=c_H, linewidth=1.5,
								linestyle='dashed', zorder=-1)

	if include_data_availability:
		# scale the data availability to 0 - -100: 0 = 0% flagged; -100 = 100 % flagged:
		hatpro_data_flagged *= -100
		ax1.bar(date_array, hatpro_data_flagged, width=1.0, color=(0.9,0,0))

		# dummy for legend:
		ax2.bar([np.nan, np.nan], [np.nan, np.nan], width=1.0, color=(0.9,0,0),
					label="HATPRO data flagged")

	lh, ll = ax1.get_legend_handles_labels()
	ax2.legend(handles=lh, labels=ll, loc="upper left", fontsize=fs)


	for leg in [*MOSAiC_legs.keys()]:
		if MOSAiC_legs[leg][0] >= date_range_start and MOSAiC_legs[leg][0] <= date_range_end:
			ax1.plot([MOSAiC_legs[leg][0], MOSAiC_legs[leg][0]], axlim, color=(0,0,0), linewidth=2)
		if MOSAiC_legs[leg][1] >= date_range_start and MOSAiC_legs[leg][1] <= date_range_end:
			ax1.plot([MOSAiC_legs[leg][1], MOSAiC_legs[leg][1]], axlim, color=(0,0,0), linewidth=2)


	# set axis limits:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax1.set_xlim(left=dt.datetime(2019,10,22,6,0), right=date_range_end)
	ax1.xaxis.set_major_formatter(dt_fmt)
	ax2.set_ylim(bottom=ax2lim[0], top=ax2lim[1])

	# labels:
	ax1.set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax2.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, color=c_H)
	if with_titles:
		the_title = ax1.set_title(("Daily average of Liquid Water Path (LWP) and Integrated Water Vapour (IWV)\nduring MOSAiC (" + 
							dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
							" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")"), fontsize=fs, pad=40)
		the_title.set_position((0.5, 1.2))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1.set_xticks(x_ticks_dt)
		ax1.tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	if include_data_availability:
		ytick_labels = [yt for yt in ax1.get_yticks()]
		ax1.yaxis.set_ticklabels(["100$\,\%$"] + ytick_labels[1:])

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

	# include texts indicating the legs of MOSAiC:
	for MLT, x_pos in zip(MOSAiC_Legs_Text, MLT_x_pos):
		if x_pos >= date_range_start and x_pos <= date_range_end:
			ax1.text(x_pos, 1.01*axlim[1], MLT, fontweight='bold', fontsize=fs, ha='center', va='bottom')

	if save_figures:
		iwv_name_base = "LWP_IWV_daily_mean_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')

	elif save_figures_eps:
		iwv_name_base = "LWP_IWV_daily_mean_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "_hatpro_" + considered_period
		else:
			iwv_name_suffix_def = "_hatpro_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_option + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()
