import numpy as np
import datetime as dt
import copy
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import sys
import warnings
import netCDF4 as nc

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from met_tools import *
from data_tools import *
from scipy import stats
import xarray as xr
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


def import_mirac_level2a(
	filename,
	keys='basic',
	minute_avg=False):

	"""
	Importing MiRAC-P level 2a (integrated quantities, e.g. IWV, LWP).

	Parameters:
	-----------
	filename : str
		Path and filename of mwr data (level2a).
	keys : list of str or str, optional
		Specify which variables are to be imported. Another option is
		to import all keys (keys='all') or import basic keys
		that the author considers most important (keys='basic')
		or leave this argument out.
	minute_avg : bool
		If True: averages over one minute are computed and returned instead of False when all
		data points are returned (more outliers, higher memory usage).
	"""

	file_nc = nc.Dataset(filename)

	if keys == 'basic': 
		keys = ['time', 'lat', 'lon', 'zsl', 'azi', 'ele', 'flag']
		if 'clwvi_' in filename:
			for add_key in ['clwvi']: keys.append(add_key)
		if 'prw_' in filename:
			for add_key in ['prw']: keys.append(add_key)

	elif keys == 'all':
		keys = file_nc.variables.keys()

	elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
		raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

	mwr_dict = dict()
	for key in keys:
		if not key in file_nc.variables.keys():
			raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
		mwr_dict[key] = np.asarray(file_nc.variables[key])


	if 'time' in keys:	# avoid nasty digita after decimal point
		mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)

	return mwr_dict


def import_mirac_level2a_daterange(
	path_data,
	date_start,
	date_end,
	which_retrieval='both',
	vers='v01',
	minute_avg=False,
	verbose=0):

	"""
	Runs through all days between a start and an end date. It concats the level 2a data time
	series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
	for the entire date range period.

	Parameters:
	-----------
	path_data : str
		Base path of level 2a data. This directory contains subfolders representing the year, which,
		in turn, contain months, which contain day subfolders. Example path_data:
		"/data/obs/campaigns/mosaic/mirac-p/l2/"
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	which_retrieval : str, optional
		This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
		integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
		load both. Default: 'both'
	vers : str
		Indicates the mwr_pro output version number. Valid options: 'v01'
	minute_avg : bool
		If True: averages over one minute are computed and returned instead of False when all
		data points are returned (more outliers, higher memory usage).
	verbose : int
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	# check if the input of the retrieval variable is okay:
	if not isinstance(which_retrieval, str):
			raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
				"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
				"'both' will load both. Default: 'both'")

	else:
		if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
			raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
				"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
				"'both' will load both. Default: 'both'")

		else:
				if which_retrieval == 'iwv':
					which_retrieval = ['prw']
				elif which_retrieval == 'lwp':
					which_retrieval = ['clwvi']
				elif which_retrieval == 'both':
					which_retrieval = ['prw', 'clwvi']
				else:
					raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
						"'iwv' or 'prw' will load the " +
						"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
						"'both' will load both. Default: 'both'")
					

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
	date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


	# count the number of days between start and end date as max. array size:
	n_days = (date_end - date_start).days + 1
	n_ret = 86			# inquired from level 2a data, number of available elevation angles in retrieval

	# basic variables that should always be imported:
	mwr_time_keys = ['time', 'azi', 'ele', 'flag']				# keys with time as coordinate

	# mwr_master_dict (output) will contain all desired variables on specific axes:
	# e.g. level 2a and 2b have got the same time axis (according to pl_mk_nds.pro)
	# and azimuth and elevation angles.
	mwr_master_dict = dict()
	if minute_avg:	# max number of minutes: n_days*1440
		n_minutes = n_days*1440
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)

		if 'prw' in which_retrieval:
			mwr_master_dict['prw'] = np.full((n_minutes,), np.nan)

		if 'clwvi' in which_retrieval:
			mwr_master_dict['clwvi'] = np.full((n_minutes,), np.nan)

	else:			# max number of seconds: n_days*86400
		n_seconds = n_days*86400
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)

		if 'prw' in which_retrieval:
			mwr_master_dict['prw'] = np.full((n_seconds,), np.nan)

		if 'clwvi' in which_retrieval:
			mwr_master_dict['clwvi'] = np.full((n_seconds,), np.nan)


	# cycle through all years, all months and days:
	time_index = 0	# this index (for lvl 2a) will be increased by the length of the time
						# series of the current day (now_date) to fill the mwr_master_dict time axis
						# accordingly.
	day_index = 0	# same as above, but only increases by 1 for each day
	for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

		if verbose >= 1: print("Working on MiRAC-P Level 2a, ", now_date)

		yyyy = now_date.year
		mm = now_date.month
		dd = now_date.day

		day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

		if not os.path.exists(os.path.dirname(day_path)):
			continue

		# list of files:
		mirac_level2_nc = sorted(glob.glob(day_path + "*.nc"))
		# filter for i01 files:
		mirac_level2_nc = [lvl2_nc for lvl2_nc in mirac_level2_nc if vers in lvl2_nc]

		if len(mirac_level2_nc) == 0:
			if verbose >= 2:
				warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
			continue

		# identify level 2a files:
		mirac_level2a_nc = []
		for lvl2_nc in mirac_level2_nc:
			for wr in which_retrieval:
				if wr + '_' in lvl2_nc:
					mirac_level2a_nc.append(lvl2_nc)

		# load one retrieved variable after another from current day and save it into the mwr_master_dict
		for lvl2_nc in mirac_level2a_nc: 
			mwr_dict = import_mirac_level2a(lvl2_nc, minute_avg=minute_avg)

			n_time = len(mwr_dict['time'])
			cur_time_shape = mwr_dict['time'].shape

			# save to mwr_master_dict
			for mwr_key in mwr_dict.keys():
				mwr_key_shape = mwr_dict[mwr_key].shape
				if mwr_key_shape == cur_time_shape:	# then the variable is on time axis:
					mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

				elif mwr_key_shape == (): # lat, lon, zsl
					continue

				else:
					raise RuntimeError("Something went wrong in the " +
						"import_mirac_level2a_daterange routine. Unexpected MWR variable dimension. " + 
						"The length of one used variable ('%s') of level 2a data "%(mwr_key) +
							"neither equals the length of the time axis nor equals 1.")


		time_index = time_index + n_time
		day_index = day_index + 1

	for mwr_key in ['lat', 'lon', 'zsl']: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

	if time_index == 0 and verbose >= 1: 	# otherwise no data has been found
		raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
				dt.datetime.strftime(date_end, "%Y-%m-%d"))
	else:

		# truncate the mwr_master_dict to the last nonnan time index:
		last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
		time_shape_old = mwr_master_dict['time'].shape
		for mwr_key in mwr_master_dict.keys():
			if mwr_master_dict[mwr_key].shape == time_shape_old:
				mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

	return mwr_master_dict


def import_hatpro_level2a(
	filename,
	keys='basic',
	minute_avg=False):

	"""
	Importing HATPRO level 2a (integrated quantities, e.g. IWV, LWP).

	Parameters:
	-----------
	filename : str
		Path and filename of mwr data (level2a).
	keys : list of str or str, optional
		Specify which variables are to be imported. Another option is
		to import all keys (keys='all') or import basic keys
		that the author considers most important (keys='basic')
		or leave this argument out.
	minute_avg : bool
		If True: averages over one minute are computed and returned instead of False when all
		data points are returned (more outliers, higher memory usage).
	"""

	file_nc = nc.Dataset(filename)

	if keys == 'basic': 
		keys = ['time', 'lat', 'lon', 'zsl', 'flag']
		if 'clwvi_' in filename:
			for add_key in ['clwvi', 'clwvi_err', 'clwvi_offset']: keys.append(add_key)
		if 'prw_' in filename:
			for add_key in ['prw', 'prw_err', 'prw_offset']: keys.append(add_key)

	elif keys == 'all':
		keys = file_nc.variables.keys()

	elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
		raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

	mwr_dict = dict()
	for key in keys:
		if not key in file_nc.variables.keys():
			raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
		mwr_dict[key] = np.asarray(file_nc.variables[key])


	if 'time' in keys:	# avoid nasty digita after decimal point
		mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
	return mwr_dict


def import_hatpro_level2a_daterange(
	path_data,
	date_start,
	date_end,
	which_retrieval='both',
	minute_avg=False,
	vers='p00',
	verbose=0):

	"""
	Runs through all days between a start and an end date. It concats the level 2a data time
	series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
	for the entire date range period.

	Parameters:
	-----------
	path_data : str
		Base path of level 2a data. This directory contains subfolders representing the year, which,
		in turn, contain months, which contain day subfolders. Example path_data:
		"/data/obs/campaigns/mosaic/hatpro/l2/"
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	which_retrieval : str, optional
		This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
		integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
		load both. Default: 'both'
	minute_avg : bool
		If True: averages over one minute are computed and returned instead of False when all
		data points are returned (more outliers, higher memory usage).
	vers : str
		Indicates the mwr_pro output version number. Valid options: 'i01', 'v00', and 'v01'. Default: 'v01'
	verbose : int
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	if vers not in ['p00']:
		raise ValueError("In import_hatpro_level2a_daterange, the argument 'vers' must be one of the" +
							" following options: 'p00'")

	# check if the input of the retrieval variable is okay:
	if not isinstance(which_retrieval, str):
			raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
				"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
				"'both' will load both. Default: 'both'")

	else:
		if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
			raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
				"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
				"'both' will load both. Default: 'both'")

		else:
				if which_retrieval == 'iwv':
					which_retrieval = ['prw']
				elif which_retrieval == 'lwp':
					which_retrieval = ['clwvi']
				elif which_retrieval == 'both':
					which_retrieval = ['prw', 'clwvi']
				else:
					raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
						"'iwv' or 'prw' will load the " +
						"integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
						"'both' will load both. Default: 'both'")
					

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
	date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


	# count the number of days between start and end date as max. array size:
	n_days = (date_end - date_start).days + 1
	n_ret = 86			# inquired from level 2a data, number of available elevation angles in retrieval

	# basic variables that should always be imported:
	mwr_time_keys = ['time', 'flag']				# keys with time as coordinate

	# mwr_master_dict (output) will contain all desired variables on specific axes:
	# e.g. level 2a and 2b have got the same time axis (according to pl_mk_nds.pro)
	# and azimuth and elevation angles.
	mwr_master_dict = dict()
	if minute_avg:	# max number of minutes: n_days*1440
		n_minutes = n_days*1440
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)

		if 'prw' in which_retrieval:
			mwr_master_dict['prw'] = np.full((n_minutes,), np.nan)
			mwr_master_dict['prw_offset'] = np.full((n_minutes,), np.nan)							####### could be reduced to a daily value or even one value for the entire period
			# mwr_master_dict['prw_err'] = np.full((n_days, n_ret), np.nan)								####### could be reduced to one value array for the entire period
			mwr_master_dict['prw_err'] = np.full((n_ret,), np.nan)

		if 'clwvi' in which_retrieval:
			mwr_master_dict['clwvi'] = np.full((n_minutes,), np.nan)
			mwr_master_dict['clwvi_offset'] = np.full((n_minutes,), np.nan)							####### could be reduced to a daily value or even one value for the entire period
			# mwr_master_dict['clwvi_err'] = np.full((n_days, n_ret), np.nan)								###### could be reduced to one value for the entire period
			mwr_master_dict['clwvi_err'] = np.full((n_ret,), np.nan)

	else:			# max number of seconds: n_days*86400
		n_seconds = n_days*86400
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)

		if 'prw' in which_retrieval:
			mwr_master_dict['prw'] = np.full((n_seconds,), np.nan)
			mwr_master_dict['prw_offset'] = np.full((n_seconds,), np.nan)							####### could be reduced to a daily value or even one value for the entire period
			# mwr_master_dict['prw_err'] = np.full((n_days, n_ret), np.nan)								####### could be reduced to one value array for the entire period
			mwr_master_dict['prw_err'] = np.full((n_ret,), np.nan)

		if 'clwvi' in which_retrieval:
			mwr_master_dict['clwvi'] = np.full((n_seconds,), np.nan)
			mwr_master_dict['clwvi_offset'] = np.full((n_seconds,), np.nan)							####### could be reduced to a daily value or even one value for the entire period
			# mwr_master_dict['clwvi_err'] = np.full((n_days, n_ret), np.nan)								###### could be reduced to one value for the entire period
			mwr_master_dict['clwvi_err'] = np.full((n_ret,), np.nan)


	# cycle through all years, all months and days:
	time_index = 0	# this index (for lvl 2a) will be increased by the length of the time
						# series of the current day (now_date) to fill the mwr_master_dict time axis
						# accordingly.
	day_index = 0	# same as above, but only increases by 1 for each day
	for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

		if verbose >= 1: print("Working on HATPRO Level 2a, ", now_date)

		yyyy = now_date.year
		mm = now_date.month
		dd = now_date.day

		day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

		if not os.path.exists(os.path.dirname(day_path)):
			continue

		# list of files:
		hatpro_level2_nc = sorted(glob.glob(day_path + "*.nc"))
		# filter for v01 files:
		hatpro_level2_nc = [lvl2_nc for lvl2_nc in hatpro_level2_nc if vers in lvl2_nc]

		if len(hatpro_level2_nc) == 0:
			if verbose >= 2:
				warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
			continue

		# identify level 2a files:
		hatpro_level2a_nc = []
		for lvl2_nc in hatpro_level2_nc:
			for wr in which_retrieval:
				if wr + '_' in lvl2_nc:
					hatpro_level2a_nc.append(lvl2_nc)

		# load one retrieved variable after another from current day and save it into the mwr_master_dict
		for lvl2_nc in hatpro_level2a_nc: 
			mwr_dict = import_hatpro_level2a(lvl2_nc, minute_avg=minute_avg)

			n_time = len(mwr_dict['time'])
			cur_time_shape = mwr_dict['time'].shape

			# save to mwr_master_dict
			for mwr_key in mwr_dict.keys():
				mwr_key_shape = mwr_dict[mwr_key].shape
				if mwr_key_shape == cur_time_shape:	# then the variable is on time axis:
					mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

				elif mwr_key == 'prw_err' or mwr_key == 'clwvi_err': 	# these variables are nret x 1 arrays
					# mwr_master_dict[mwr_key][day_index:day_index + 1, :] = mwr_dict[mwr_key]			## for the case that we leave _err a daily value
					mwr_master_dict[mwr_key][:] = mwr_dict[mwr_key]

				elif mwr_key_shape == (): # lat, lon, zsl
					continue

				else:
					raise RuntimeError("Something went wrong in the " +
						"import_hatpro_level2a_daterange routine. Unexpected MWR variable dimension. " + 
						"The length of one used variable ('%s') of level 2a data "%(mwr_key) +
							"neither equals the length of the time axis nor equals 1.")


		time_index = time_index + n_time
		day_index = day_index + 1

	for mwr_key in ['lat', 'lon', 'zsl']: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

	if time_index == 0 and verbose >= 1: 	# otherwise no data has been found
		raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
				dt.datetime.strftime(date_end, "%Y-%m-%d"))
	else:

		# truncate the mwr_master_dict to the last nonnan time index:
		last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
		time_shape_old = mwr_master_dict['time'].shape
		for mwr_key in mwr_master_dict.keys():
			if mwr_master_dict[mwr_key].shape == time_shape_old:
				mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

	return mwr_master_dict


################################################################


# Paths:
path_hatpro_level2 = "/data/obs/site/jue/tophat/l2/"				# path of hatpro derived products
path_mirac = "/data/obs/site/jue/mirac-p/l2/"						# path of mirac-p derived products
path_plots = "/net/blanc/awalbroe/Plots/mirac_p_jue/"


plot_IWV_time_series = True				# simple IWV time series
plot_IWV_scatterplots = False		# scatterplot of IWV: HATPRO vs. sonde, MiRAC-P vs. sonde for entire period
plot_IWV_diff = False				# IWV difference plot
plot_IWV_diff_histogram = False		# histogram of IWV differences
save_figures = False
save_figures_eps = False		# save figures as vector graphics (pdf or eps)
with_titles = False				# if True, plots will have titles (False for publication plots)

which_retrievals = 'iwv'		# which data is to be imported: 'iwv' only
mirac_version = 'v01'					# version of outout: currently available: "v01"


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
considered_period = 'user'
daterange_options = {'user': ["2021-12-01", "2022-02-10"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"


# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

# load MiRAC-P IWV values:
mirac_dict = import_mirac_level2a_daterange(path_mirac, date_start, date_end, which_retrieval=which_retrievals, 
											vers=mirac_version, minute_avg=False, verbose=1)

# import sonde and hatpro level 2a data and mirac IWV data:
# and create a datetime variable for plotting and apply running mean, if specified:
hatpro_dict = import_hatpro_level2a_daterange(path_hatpro_level2, date_start, date_end, 
											which_retrieval=which_retrievals, minute_avg=False, verbose=1)


# Create datetime out of the MWR times:x
hatpro_dict['time_npdt'] = hatpro_dict['time'].astype("datetime64[s]")
mirac_dict['time_npdt'] = mirac_dict['time'].astype("datetime64[s]")


# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")


import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
dt_fmt = mdates.DateFormatter("%b %d") # (e.g. "Feb 23")
datetick_auto = False
fs = 24		# fontsize

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P

# create x_ticks depending on the date range: roughly 20 x_ticks are planned
# round off to days if number of days > 15:
date_range_delta = (date_range_end - date_range_start)
if date_range_delta < dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=6)
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta < dt.timedelta(days=11) and date_range_delta >= dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=12)
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta >= dt.timedelta(days=11) and date_range_delta < dt.timedelta(21):
	x_tick_delta = dt.timedelta(days=1)
else:
	x_tick_delta = dt.timedelta(days=(date_range_delta/20).days)

x_ticks_dt = mdates.drange(date_range_start, date_range_end, x_tick_delta)	# alternative if the xticklabel is centered



########## Plotting ##########

# Flag stuff:
hatpro_dict['flag'][hatpro_dict['prw'] == -999.] = 1.
hatpro_dict['prw'][hatpro_dict['flag'] > 0] = np.nan
mirac_dict['prw'][mirac_dict['flag'] > 0] = np.nan


if plot_IWV_time_series:

	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(22,10)
	
	axlim = [0, 35]			# axis limits for IWV plot in kg m^-2

	HATPRO_IWV_plot = ax1.plot(hatpro_dict['time_npdt'], hatpro_dict['prw'],
									color=c_H, linewidth=1.2)
	MIRAC_IWV_plot = ax1.plot(mirac_dict['time_npdt'], mirac_dict['prw'],
									color=c_M, linewidth=1.2)


	# legend + dummy lines for thicker lines in the legend:
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, label='HATPRO')
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linewidth=2.0, label='MiRAC-P')
	iwv_leg_handles, iwv_leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0, markerscale=1.5)

	# set axis limits and labels:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax1.set_xlim(left=date_range_start, right=date_range_end)
	ax1.xaxis.set_major_formatter(dt_fmt)
	ax1.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles:
		the_title = ax1.set_title("Integrated Water Vapour (IWV) TOPHAT vs. MiRAC-P (" + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
							" - " + dt.datetime.strftime(date_range_end-dt.timedelta(days=1), "%Y-%m-%d") + ")", fontsize=fs, pad=50)
		the_title.set_position((0.5, 1.2))

	ax1.set_xticks(x_ticks_dt)
	ax1.tick_params(axis='x', labelsize=fs-2, labelrotation=90)

	ax1.tick_params(axis='y', labelsize=fs-2)
	ax1.grid(which='major', axis='both')

	ax1_pos = ax1.get_position().bounds
	ax1.set_position([ax1_pos[0], ax1_pos[1]+0.1, ax1_pos[2], ax1_pos[3]*0.9])


	if save_figures:
		iwv_name_base = "IWV_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "hatpro_mirac_joyce" + considered_period
		else:
			iwv_name_suffix_def = "hatpro_mirac_joyce" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')

	elif save_figures_eps:
		iwv_name_base = "IWV_time_series_"
		if considered_period != 'user':
			iwv_name_suffix_def = "hatpro_mirac_joyce" + considered_period
		else:
			iwv_name_suffix_def = "hatpro_mirac_joyce" + date_start.replace("-","") + "-" + date_end.replace("-","")
		fig1.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_IWV_diff:
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


if plot_IWV_scatterplots:
	# IWV scatterplot for entire period

	# bring on mutual time axis:
	mirac_dict['prw_ip'] = np.interp(hatpro_dict['time'], mirac_dict['time'], mirac_dict['prw'], left=np.nan, right=np.nan)
	mirac_dict['time_ip'] = hatpro_dict['time']
	mirac_dict['time_npdt_ip'] = hatpro_dict['time_npdt']
	

	fig6 = plt.figure(figsize=(10,10))
	ax1 = plt.axes(box_aspect=1)
	
	# handle axis limits:
	axlim = np.asarray([0, 35])

	# compute retrieval statistics:
	ret_stat_dict = compute_retrieval_statistics(hatpro_dict['prw'], mirac_dict['prw_ip'],
													compute_stddev=True)

	# plotting:
	ax1.plot(hatpro_dict['prw'][::60], mirac_dict['prw_ip'][::60], 
			markerfacecolor=c_H, markeredgecolor=(0,0,0),
			linestyle='none', marker='.', markersize=9.0, label="obs")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	nonnan_hatson = np.argwhere(~np.isnan(hatpro_dict['prw']) &
						~np.isnan(mirac_dict['prw_ip'])).flatten()
	y_fit = mirac_dict['prw_ip'][nonnan_hatson]
	x_fit = hatpro_dict['prw'][nonnan_hatson]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
		m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
		a = m_fit[0]
		b = m_fit[1]

		ds_fit = ax1.plot(axlim, a*axlim + b, color=c_H, linewidth=1.2, label="Best fit")

	# plot a line for orientation which would represent a perfect fit:
	ax1.plot(axlim, axlim, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")


	# add figure identifier of subplots: a), b), ...
	ax1.text(0.02, 0.98, "HATPRO and MiRAC-P at JOYCE", color=(0,0,0), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax1.transAxes)


	# add statistics:
	ax1.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nstd. = %.2f \nR = %.3f"%(ret_stat_dict['N'], 
			np.nanmean(np.concatenate((hatpro_dict['prw'], mirac_dict['prw_ip']), axis=0)),
			ret_stat_dict['bias'], ret_stat_dict['rmse'], ret_stat_dict['stddev'], ret_stat_dict['R']),
			ha='right', va='bottom', transform=ax1.transAxes, fontsize=fs-2)


	# Legends:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs)


	# set axis limits:
	ax1.set_ylim(bottom=axlim[0], top=axlim[1])
	ax1.set_xlim(left=axlim[0], right=axlim[1])


	# set axis ticks, ticklabels and tick parameters:
	ax1.minorticks_on()
	ax1.tick_params(axis='both', labelsize=fs-2)


	# # aspect ratio:
	# ax1.set_aspect('equal')


	# grid:
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)


	# labels:
	ax1.set_ylabel("IWV$_{\mathrm{HATPRO}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_xlabel("IWV$_{\mathrm{MiRAC-P}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	if with_titles: ax1.set_title("IWV comparison of HATPRO and\nMiRAC-P at JOYCE", fontsize=fs)


	if save_figures:
		iwv_name_base = "IWV_scatterplot_jue_"
		iwv_name_suffix_def = "hatpro_vs_mirac"
		fig6.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400, bbox_inches='tight')
		# plt.show()

	elif save_figures_eps:
		iwv_name_base = "IWV_scatterplot_jue_"
		iwv_name_suffix_def = "hatpro_vs_mirac"
		fig6.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".pdf", bbox_inches='tight')

	else:
		plt.show()


if plot_IWV_diff_histogram:
	
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

