import numpy as np
import copy
import datetime as dt
import xarray as xr
import os
import glob
import pdb
import warnings


def datetime_to_epochtime(dt_array):
	
	"""
	This tool creates a 1D array (or of seconds since 1970-01-01 00:00:00 UTC
	(type: float) out of a datetime object or an array of datetime objects.

	Parameters:
	-----------
	dt_array : array of datetime objects or datetime object
		Array (1D) that includes datetime objects. Alternatively, dt_array is directly a
		datetime object.
	"""

	reftime = dt.datetime(1970,1,1)

	try:
		sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in dt_array])
	except TypeError:	# then, dt_array is no array
		sec_epochtime = (dt_array - reftime).total_seconds()

	return sec_epochtime


def numpydatetime64_to_epochtime(npdt_array):

	"""
	Converts numpy datetime64 array to array in seconds since 1970-01-01 00:00:00 UTC (type:
	float).
	Alternatively, just use "some_array.astype(np.float64)".

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	return sec_epochtime


def numpydatetime64_to_reftime(
	npdt_array, 
	reftime):

	"""
	Converts numpy datetime64 array to array in seconds since a reftime as type:
	float. Reftime could be for example: "2017-01-01 00:00:00" (in UTC)

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	reftime : str
		Specification of the reference time in "yyyy-mm-dd HH:MM:SS" (in UTC).
	"""

	time_dt = numpydatetime64_to_datetime(npdt_array)

	reftime = dt.datetime.strptime(reftime, "%Y-%m-%d %H:%M:%S")

	try:
		sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in time_dt])
	except TypeError:	# then, time_dt is no array
		sec_epochtime = (time_dt - reftime).total_seconds()

	return sec_epochtime


def numpydatetime64_to_datetime(npdt_array):

	"""
	Converts numpy datetime64 array to a datetime object array.

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	# sec_epochtime can be an array or just a float
	if sec_epochtime.ndim > 0:
		time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in sec_epochtime])

	else:
		time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)

	return time_dt


def break_str_into_lines(
	le_string,
	n_max,
	split_at=' ',
	keep_split_char=False):

	"""
	Break a long strings into multiple lines if a certain number of chars may
	not be exceeded per line. String will be split into two lines if its length
	is > n_max but <= 2*n_max.

	Parameters:
	-----------
	le_string : str
		String that will be broken into several lines depending on n_max.
	n_max : int
		Max number of chars allowed in one line.
	split_at : str
		Character to look for where the string will be broken. Default: space ' '
	keep_split_char : bool
		If True, the split char indicated by split_at will not be removed (useful for "-" as split char).
		Default: False
	"""

	n_str = len(le_string)
	if n_str > n_max:
		# if string is > 2*n_max, then it has to be split into three lines, ...:
		n_lines = (n_str-1) // n_max		# // is flooring division

		# look though the string in backwards direction to find the first space before index n_max:
		le_string_bw = le_string[::-1]
		new_line_str = "\n"

		for k in range(n_lines):
			space_place = le_string_bw.find(split_at, n_str - (k+1)*n_max)
			if keep_split_char:
				le_string_bw = le_string_bw[:space_place].replace("\n","") + new_line_str + le_string_bw[space_place:]
			else:
				le_string_bw = le_string_bw[:space_place] + new_line_str + le_string_bw[space_place+1:]

		# reverse the string again
		le_string = le_string_bw[::-1]

	return le_string


def Gband_double_side_band_average(
	TB,
	freqs,
	xarray_compatibility=False,
	freq_dim_name=""):

	"""
	Computes the double side band average of TBs that contain both
	sides of the G band absorption line. Returns either only the TBs
	or both the TBs and frequency labels with double side band avg.
	If xarray_compatibility is True, also more dimensional TB arrays
	can be included. Then, also the frequency dimension name must be
	supplied.

	Parameters:
	-----------
	TB : array of floats or xarray data array
		Brightness temperature array. Must have the following shape
		(time x frequency). More dimensions and other shapes are only
		allowed if xarray_compatibility=True.
	freqs : array of floats or xarray data array
		1D Array containing the frequencies of the TBs. The array must be
		sorted in ascending order.
	xarray_compatibility : bool
		If True, xarray utilities can be used, also allowing TBs of other
		shapes than (time x frequency). Then, also freq_dim_name must be
		provided.
	freq_dim_name : str
		Name of the xarray frequency dimension. Must be specified if 
		xarray_compatibility=True.
	"""

	if xarray_compatibility and not freq_dim_name:
		raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

	# Double side band average for G band if G band frequencies are available, which must first be clarified:
	# Determine, which frequencies are around the G band w.v. absorption line:
	g_upper_end = 183.31 + 15
	g_lower_end = 183.31 - 15
	g_freq = np.where((freqs > g_lower_end) & (freqs < g_upper_end))[0]
	non_g_freq = np.where(~((freqs > g_lower_end) & (freqs < g_upper_end)))[0]

	TB_dsba = copy.deepcopy(TB)

	if g_freq.size > 0: # G band within frequencies
		g_low = np.where((freqs <= 183.31) & (freqs >= g_lower_end))[0]
		g_high = np.where((freqs >= 183.31) & (freqs <= g_upper_end))[0]

		assert len(g_low) == len(g_high)
		if not xarray_compatibility:
			for jj in range(len(g_high)):
				TB_dsba[:,jj] = (TB[:,g_low[-1-jj]] + TB[:,g_high[jj]])/2.0

		else:
			for jj in range(len(g_high)):
				TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0


	# Indices for sorting:
	idx_have = np.concatenate((g_high, non_g_freq), axis=0)
	idx_sorted = np.argsort(idx_have)

	# truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
	if not xarray_compatibility:
		TB_dsba = TB_dsba[:,:len(g_low)]
		TB_dsba = np.concatenate((TB_dsba, TB[:,non_g_freq]), axis=1)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[:,idx_sorted]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = np.concatenate((freqs[g_high], freqs[non_g_freq]))[idx_sorted]

	else:
		TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(g_low))}]
		TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_g_freq}]], dim=freq_dim_name)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = xr.concat([freqs[g_high], freqs[non_g_freq]], dim=freq_dim_name)[idx_sorted]


	return TB_dsba, freq_dsba


def Fband_double_side_band_average(
	TB,
	freqs,
	xarray_compatibility=False,
	freq_dim_name=""):

	"""
	Computes the double side band average of TBs that contain both
	sides of the F band absorption line. Returns either only the TBs
	or both the TBs and frequency labels with double side band avg.

	Parameters:
	-----------
	TB : array of floats
		Brightness temperature array. Must have the following shape
		(time x frequency).
	freqs : array of floats
		1D Array containing the frequencies of the TBs. The array must be
		sorted in ascending order.
	xarray_compatibility : bool
		If True, xarray utilities can be used, also allowing TBs of other
		shapes than (time x frequency). Then, also freq_dim_name must be
		provided.
	freq_dim_name : str
		Name of the xarray frequency dimension. Must be specified if 
		xarray_compatibility=True.
	"""

	if xarray_compatibility and not freq_dim_name:
		raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

	# Double side band average for F band if F band frequencies are available, which must first be clarified:
	# Determine, which frequencies are around the F band w.v. absorption line:
	upper_end = 118.75 + 10
	lower_end = 118.75 - 10
	f_freq = np.where((freqs > lower_end) & (freqs < upper_end))[0]
	non_f_freq = np.where(~((freqs > lower_end) & (freqs < upper_end)))[0]

	TB_dsba = copy.deepcopy(TB)
	
	if f_freq.size > 0: # F band within frequencies
		low = np.where((freqs <= 118.75) & (freqs >= lower_end))[0]
		high = np.where((freqs >= 118.75) & (freqs <= upper_end))[0]

		assert len(low) == len(high)
		if not xarray_compatibility:
			for jj in range(len(high)):
				TB_dsba[:,jj] = (TB[:,low[-1-jj]] + TB[:,high[jj]])/2.0

		else:
			for jj in range(len(high)):
				TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: low[-1-jj]}] + TB[{freq_dim_name: high[jj]}])/2.0


	# Indices for sorting:
	idx_have = np.concatenate((high, non_f_freq), axis=0)
	idx_sorted = np.argsort(idx_have)

	# truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
	if not xarray_compatibility:
		TB_dsba = TB_dsba[:,:len(low)]
		TB_dsba = np.concatenate((TB_dsba, TB[:,non_f_freq]), axis=1)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[:,idx_sorted]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = np.concatenate((freqs[high], freqs[non_f_freq]))[idx_sorted]

	else:
		TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(low))}]
		TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_f_freq}]], dim=freq_dim_name)

		# Now, the array just needs to be sorted correctly:
		TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

		# define freq_dsba (usually, the upper side of the G band is then used as
		# frequency label:
		freq_dsba = xr.concat([freqs[high], freqs[non_f_freq]], dim=freq_dim_name)[idx_sorted]

	return TB_dsba, freq_dsba


def select_MWR_channels(
	TB,
	freq,
	band,
	return_idx=0):

	"""
	This function selects certain frequencies (channels) of brightness temperatures (TBs)
	from a given set of TBs. The output will therefore be a subset of the input TBs. Single
	frequencies cannot be selected but only 'bands' (e.g. K band, V band, ...). Combinations
	are also possible.

	Parameters:
	-----------
	TB : array of floats
		2D array (i.e., time x freq; freq must be the second dimension) of TBs (in K).
	freq : array of floats
		1D array of frequencies (in GHz).
	band : str
		Specify the frequencies to be selected. Valid options:
		'K': 20-40 GHz, 'V': 50-60 GHz, 'W': 85-95 GHz, 'F': 110-130 GHz, 'G': 170-200 GHz,
		'243/340': 240-350 GHz
		Combinations are also possible: e.g. 'K+V+W' = 20-95 GHz
	return_idx : int
		If 0 the frq_idx list is not returned and merely TB and freq are returned.
		If 1 TB, freq, and frq_idx are returned. If 2 only frq_idx is returned.
	"""

	# define dict of band limits:
	band_lims = {	'K': [20, 40],
					'V': [50, 60],
					'W': [85, 95],
					'F': [110, 130],
					'G': [170, 200],
					'243/340': [240, 350]}

	# split band input:
	band_split = band.split('+')

	# cycle through all bands:
	frq_idx = list()
	for k, baba in enumerate(band_split):
		# find the right indices for the appropriate frequencies:
		frq_idx_temp = np.where((freq >= band_lims[baba][0]) & (freq <= band_lims[baba][1]))[0]
		for fit in frq_idx_temp: frq_idx.append(fit)

	# sort the list and select TBs:
	frq_idx = sorted(frq_idx)
	TB = TB[:, frq_idx]
	freq = freq[frq_idx]

	if return_idx == 0:
		return TB, freq

	elif return_idx == 1:
		return TB, freq, frq_idx

	elif return_idx == 2:
		return frq_idx

	else:
		raise ValueError("'return_idx' in function 'select_MWR_channels' must be an integer. Valid options: 0, 1, 2")


def compute_retrieval_statistics(
	x_stuff,
	y_stuff,
	compute_stddev=False):

	"""
	Compute bias, RMSE and Pearson correlation coefficient (and optionally the standard deviation)
	from x and y data.

	Parameters:
	x_stuff : float or array of floats
		Data that is to be plotted on the x axis.
	y_stuff : float or array of floats
		Data that is to be plotted on the y axis.
	compute_stddev : bool
		If True, the standard deviation is computed (bias corrected RMSE).
	"""

	where_nonnan = np.argwhere(~np.isnan(y_stuff) & ~np.isnan(x_stuff)).flatten()
					# -> must be used to ignore nans in corrcoef
	stat_dict = {	'N': np.count_nonzero(~np.isnan(x_stuff) & ~np.isnan(y_stuff)),
					'bias': np.nanmean(y_stuff - x_stuff),
					'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
					'R': np.corrcoef(x_stuff[where_nonnan], y_stuff[where_nonnan])[0,1]}

	if compute_stddev:
		stat_dict['stddev'] = np.sqrt(np.nanmean((x_stuff - (y_stuff - stat_dict['bias']))**2))

	return stat_dict