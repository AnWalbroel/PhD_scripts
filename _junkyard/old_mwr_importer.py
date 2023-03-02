def import_mirac_IWV_RPG_daterange(
	path_data,
	date_start,
	date_end,
	minute_avg=True,
	verbose=0):

	"""
	Runs through all days between a start and an end date. It concats the IWV time
	series of each day so that you'll have one dictionary that will contain the IWV
	for the entire date range period.

	Parameters:
	-----------
	path_data : str
		Base path of MiRAC-P level 1 data. This directory contains subfolders representing the 
		year, which, in turn, contain months, which contain day subfolders. Example:
		path_data = "/data/obs/campaigns/mosaic/mirac-p/l1/"
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	minute_avg : bool
		If True: averages over one minute are computed and returned instead of False when all
		data points are returned (more outliers, higher memory usage).
	verbose : int
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""


	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
	date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


	# count the number of days between start and end date as max. array size:
	n_days = (date_end - date_start).days + 1

	# basic variables that should always be imported:
	# mwr_time_keys = ['time', 'ElAng', 'AziAng', 'RF', 'IWV']	# keys with time as coordinate
	mwr_time_keys = ['time', 'RF', 'IWV']	# keys with time as coordinate

	# mwr_master_dict (output) will contain all desired variables on time axis for entire date range:
	mwr_master_dict = dict()
	if minute_avg:	# max number of minutes: n_days*1440
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_days*1440,), np.nan)
	else:			# max number of seconds: n_days*86400
		for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_days*86400,), np.nan)


	# cycle through all years, all months and days:
	time_index = 0	# this index will be increased by the length of the time
						# series of the current day (now_date) to fill the mwr_master_dict time axis
						# accordingly.
	day_index = 0	# will increase for each day
	for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

		if verbose >= 1: print("Working on RPG retrieval, MiRAC-P IWV, ", now_date)

		yyyy = now_date.year
		mm = now_date.month
		dd = now_date.day

		day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

		if not os.path.exists(os.path.dirname(day_path)):
			continue

		# list of .IWV.NC files: Sorting is important as this will
		# ensure automatically that the time series of each hour will
		# be concatenated appropriately!
		mirac_iwv_nc = sorted(glob.glob(day_path + "*.IWV.NC"))

		if len(mirac_iwv_nc) == 0:
			if verbose >= 2:
				warnings.warn("No .IWV.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
			continue


		# load one retrieved variable after another from current day and save it into the mwr_master_dict
		for iwv_file in mirac_iwv_nc: 
			mwr_dict = import_mirac_IWV_RPG(iwv_file, minute_avg=minute_avg)

			n_time = len(mwr_dict['time'])
			time_shape = mwr_dict['time'].shape

			# save to mwr_master_dict
			for mwr_key in mwr_dict.keys():
				if mwr_dict[mwr_key].shape == time_shape:
					mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

				elif mwr_dict[mwr_key].shape == ():
					mwr_master_dict[mwr_key][day_index] = mwr_dict[mwr_key]

				else:
						raise ValueError("The length of one used variable ('%s') of MiRAC-P .IWV.NC data "%(mwr_key) +
							"neither equals the length of the time axis nor equals 1.")

			time_index = time_index + n_time
		day_index = day_index + 1

	if time_index == 0 and verbose >= 1: 	# otherwise no data has been found
		raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
				dt.datetime.strftime(date_end, "%Y-%m-%d"))
	else:
		# truncate the mwr_master_dict to the last nonnan time index:
		last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
		time_shape_old = mwr_master_dict['time'].shape
		for mwr_key in mwr_master_dict.keys():
			if mwr_master_dict[mwr_key].shape == time_shape_old:
				mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step]

	return mwr_master_dict