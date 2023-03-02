from __future__ import print_function, division
import numpy as np
import os
import datetime
import glob
import netCDF4 as nc
import pdb



def run_concat_mwr(day_folders, mwr_out_path):

	# Check if path for concatenated mwr files exists:
	mwr_out_path_dir = os.path.dirname(mwr_out_path)
	if not os.path.exists(mwr_out_path_dir):
		os.makedirs(mwr_out_path_dir)

	for folder in day_folders:
		folder_modules = {										##################
			'KV': folder + "/radiometer/KV/",					##################
			'11990': folder + "/radiometer/11990/",				##################
			'183': folder + "/radiometer/183/"					##################
			}
		modules = ['KV', '11990', '183']	# available modules within HAMP
		# pdb.set_trace()
		folder_date = folder[-6:]																							##################

		print("Opening mwr file of '" + folder_date + "' \n")

		# check if the folders actually exist: if true: can be used for concatenating:
		module_exist = {module: False for module in modules}	# {KV: ..., 11990: ..., 183: ...}
		for module in modules:
			if os.path.exists(folder_modules[module] + folder_date + ".BRT.NC"):
				module_exist[module] = True

		# the relevant filenames are formatted like this: yymmdd.BRT.NC: If HALO landed after midnight another file with the next day's date
		# may exist.
		nextdayfile = (datetime.datetime.strptime(folder_date, "%y%m%d") + datetime.timedelta(days=1)).strftime("%y%m%d")
		mwr_files = dict()
		for mod_key, folder_module in folder_modules.items():
			if module_exist[mod_key]:
				mwr_files[mod_key] = folder_module + folder_date + ".BRT.NC"		# .brt.nc files of that day before midnight
		# check for additional files (after midnight):
		add_files = dict()
		for mod_key, aftermidnight in {mwr_key: folder_module + nextdayfile + ".BRT.NC" for mwr_key, folder_module in folder_modules.items()}.items():
			if os.path.exists(aftermidnight):
				add_files[mod_key] = folder_modules[mod_key] + nextdayfile + ".BRT.NC"

		# Read in keys and concatenate all frequencies that we can find:
		mwr_dict_all = dict()				# will contain all modules for a day
		mwr_module_dict = dict()			# contains each single module in a seperate dictionary
		mwr_module_dict_add = dict()		# contains the mwr measurements after midnight

		for mod_key, mwr_file in mwr_files.items(): # mod_key: either KV, 11990 or 183 ... depends on the currently read file
			mwr_nc = nc.Dataset(mwr_file)
			mwr_keys = mwr_nc.variables.keys()
			mwr_dims = [mwr_nc.dimensions['time'].size, mwr_nc.dimensions['number_frequencies'].size]		# ntimes x nfreq


			mwr_module_dict[mod_key] = dict()

			# read variables of mwr_file:
			for key in mwr_keys:
				if key == 'time':
					mwr_module_dict[mod_key][key] = np.asarray(mwr_nc.variables[key]).astype(float)			# necessary to covert to unixtime

					# and finally converting to unixtime:
					time_base = datetime.datetime.strptime(mwr_nc.variables['time'].units[14:], "%d.%m.%Y, %H:%M:%S") # time base given in the units attribute
					if not mwr_nc.variables['time'].units.lower().startswith('seconds since'):
						raise ValueError("Expected time units in seconds. Given '%s'" % mwr_nc.variables['time'].units)
					else:
						mwr_module_dict[mod_key][key] = (time_base - datetime.datetime(1970,1,1)).total_seconds() + mwr_module_dict[mod_key][key]

				else:
					mwr_module_dict[mod_key][key] = np.asarray(mwr_nc.variables[key])

		# Handling the measurements after midnight with the same procedure:
		for mod_key, add_file in add_files.items():	# mod_key: either KV, 11990 or 183 ... depends on the currently read file
			mwr_nc = nc.Dataset(add_file)
			mwr_keys = mwr_nc.variables.keys()
			mwr_dims = [mwr_nc.dimensions['time'].size, mwr_nc.dimensions['number_frequencies'].size]		# ntimes x nfreq


			mwr_module_dict_add[mod_key] = dict()		# will contain the 'after midnight' measurements

			# read variables of mwr_file:
			for key in mwr_keys:
				if key == 'time':
					mwr_module_dict_add[mod_key][key] = np.asarray(mwr_nc.variables[key]).astype(float)			# necessary to covert to unixtime

					# and finally converting to unixtime:
					time_base = datetime.datetime.strptime(mwr_nc.variables['time'].units[14:], "%d.%m.%Y, %H:%M:%S") # time base given in the units attribute#
					if not mwr_nc.variables['time'].units.lower().startswith('seconds since'):
						raise ValueError("Expected time units in seconds. Given '%s'" % mwr_nc.variables['time'].units)
					else:
						mwr_module_dict_add[mod_key][key] = (time_base - datetime.datetime(1970,1,1)).total_seconds() + mwr_module_dict_add[mod_key][key]
				else:
					mwr_module_dict_add[mod_key][key] = np.asarray(mwr_nc.variables[key])


		# Now all variables for the current day have been read in: Now they have to be glued together accordingly:
		# Thus we need to find the correct timestamps first: Find earliest and latest time
		used_modules = mwr_module_dict.keys()
		add_used_modules = mwr_module_dict_add.keys()

		# concatenate frequencies of all modules:
		mwr_dict_all['frequencies'] = np.array([22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,31.4000,50.3000,51.7600,52.8000,53.7500,54.9400,56.6600,58.0000,90.0000,120.150,121.050,122.950,127.250,183.910,184.810,185.810,186.810,188.310,190.810,195.810])	# the 195.81 seems to be missing in module '183'


		# create time axis: find earliest and latest measurement (latest measurement will be taken from add_used_modules if available):
		time_0 = np.min(np.asarray([mwr_module_dict[mod]['time'][0] for mod in used_modules]))		# earliest measurement of the day
		time_end = np.max(np.asarray([mwr_module_dict[mod]['time'][-1] for mod in used_modules]))	# latest measurement before midnight

		if add_used_modules:
			time_end = np.max(np.asarray([mwr_module_dict_add[mod]['time'][-1] for mod in add_used_modules]))		# absolute latest measurement

		# create a 4 Hz time line:
		time_axis = np.arange(time_0, time_end + 1, 0.25).astype(int)			# must be integer to avoid decimal point in timestamp


		# now we have to loop through the time axis and insert TB measurements if the module recorded something at that time
		# make sure the 4 Hz measurements remain (take 4 lines):
		n_time = len(time_axis)
		n_frq = 26
		mwr_dict_all['TBs'] = np.empty((n_time, n_frq))		# HAMP has got 26 channels... although 183 is missing one frequency (the outermost)
		mwr_dict_all['TBs'][:] = np.nan				# first assume that there is no measurement for the current time step: it'll be filled afterwards
		mwr_dict_all['rain_flag'] = np.zeros((n_time, 1))		# rain flag for each time stamp
		mwr_dict_all['elevation_angle'] = np.zeros((n_time, 1))
		mwr_dict_all['azimuth_angle'] = np.zeros((n_time, 1))

		time_axis = time_axis.astype(float)
		for k in range(0, n_time-3, 4):

			# unfortunately, all modules (and after midnight files) have to be treated seperately because the number of non-zero entries may differ!
			# -> may be a bit slow, but safe.


			# check all modules because frequencies also have to be concatenated:
			# K and V band:
			if 'KV' in used_modules:
				# for as many time values as you can find in mwr_module_dict[...]['time']: do stuff
				# --> necessary because mwr measurements aren't always 4 Hz ... sometimes there were e.g. 3 per second, sometimes 5, sometimes 8
				for j in range(np.count_nonzero(mwr_module_dict['KV']['time'] == time_axis[k])): # count entries where mwr_module time matches the current time axis
					idx_KV = np.argwhere(mwr_module_dict['KV']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 0:len(mwr_module_dict['KV']['frequencies'])] = mwr_module_dict['KV']['TBs'][idx_KV, :]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict['KV']['rain_flag'][idx_KV]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict['KV']['elevation_angle'][idx_KV]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict['KV']['azimuth_angle'][idx_KV]

			if 'KV' in add_used_modules:	# same for the after midnight measurements:
				for j in range(np.count_nonzero(mwr_module_dict_add['KV']['time'] == time_axis[k])):
					idx_KV_add = np.argwhere(mwr_module_dict_add['KV']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 0:len(mwr_module_dict_add['KV']['frequencies'])] = mwr_module_dict_add['KV']['TBs'][idx_KV_add, :]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict_add['KV']['rain_flag'][idx_KV_add]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict_add['KV']['elevation_angle'][idx_KV_add]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict_add['KV']['azimuth_angle'][idx_KV_add]


			# W and F band:
			if '11990' in used_modules:
				for j in range(np.count_nonzero(mwr_module_dict['11990']['time'] == time_axis[k])):
					idx_WF = np.argwhere(mwr_module_dict['11990']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 14] = mwr_module_dict['11990']['TBs'][idx_WF, 0]
					mwr_dict_all['TBs'][k+j, 15:14+len(mwr_module_dict['11990']['frequencies'])] = mwr_module_dict['11990']['TBs'][idx_WF, 1:]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict['11990']['rain_flag'][idx_WF]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict['11990']['elevation_angle'][idx_WF]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict['11990']['azimuth_angle'][idx_WF]

			if '11990' in add_used_modules:
				for j in range(np.count_nonzero(mwr_module_dict_add['11990']['time'] == time_axis[k])):
					idx_WF_add = np.argwhere(mwr_module_dict_add['11990']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 14] = mwr_module_dict_add['11990']['TBs'][idx_WF_add, 0]
					mwr_dict_all['TBs'][k+j, 15:14+len(mwr_module_dict_add['11990']['frequencies'])] = mwr_module_dict_add['11990']['TBs'][idx_WF_add, 1:]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict_add['11990']['rain_flag'][idx_WF_add]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict_add['11990']['elevation_angle'][idx_WF_add]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict_add['11990']['azimuth_angle'][idx_WF_add]


			# G band:
			if '183' in used_modules:
				for j in range(np.count_nonzero(mwr_module_dict['183']['time'] == time_axis[k])):
					idx_G = np.argwhere(mwr_module_dict['183']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 19:19+len(mwr_module_dict['183']['frequencies'])] = mwr_module_dict['183']['TBs'][idx_G, :]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict['183']['rain_flag'][idx_G]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict['183']['elevation_angle'][idx_G]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict['183']['azimuth_angle'][idx_G]

			if '183' in add_used_modules:
				for j in range(np.count_nonzero(mwr_module_dict_add['183']['time'] == time_axis[k])):
					idx_G_add = np.argwhere(mwr_module_dict_add['183']['time'] == time_axis[k])[j]
					mwr_dict_all['TBs'][k+j, 19:19+len(mwr_module_dict_add['183']['frequencies'])] = mwr_module_dict_add['183']['TBs'][idx_G_add, :]
					mwr_dict_all['rain_flag'][k+j] = mwr_module_dict_add['183']['rain_flag'][idx_G_add]
					mwr_dict_all['elevation_angle'][k+j] = mwr_module_dict_add['183']['elevation_angle'][idx_G_add]
					mwr_dict_all['azimuth_angle'][k+j] = mwr_module_dict_add['183']['azimuth_angle'][idx_G_add]


			# WARNING: WITH THIS SETTING, SOME MEASUREMENTS MAY BE LOST: E.g. if there are 3 measurements for time stamp 601118570 in mwr_module_dict[...][time] but 5 measurements in time
			# stamp 601118571: then it would look like this:
			# 601118570: TB[601118570]
			# 601118570: TB[601118570]
			# 601118570: TB[601118570]
			# 601118570: nan
			# 601118571: TB[601118571]
			# 601118571: TB[601118571]
			# 601118571: TB[601118571]
			# 601118571: TB[601118571]
			# 601118572: TB[601118572] instead of TB[601118571]
			# 601118572: TB[601118572]
			# 601118572: TB[601118572]
			# 601118572: TB[601118572]


		# Save glued MWR modules into a new .nc file:
		mwr_outname = mwr_out_path + folder_date + "_v01.nc"
		with nc.Dataset(mwr_outname, "w", format="NETCDF4") as new_nc:

			# dimensions: time x frequency:
			new_nc.createDimension("time", n_time)
			new_nc.createDimension("number_frequencies", n_frq)

			# Global attributes:
			new_nc.description = "v01: All HAMP Microwave Radiometer modules (KV, 11990, 183) concatenated and brought to a time line from the earliest to the latest measurement of the day."
			new_nc.history = "Created: " + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
			new_nc.author = "Andreas Walbroel (Mail: a.walbroel@.uni-koeln.de)"

			# Create variables:
			freq = new_nc.createVariable("frequencies", "f8", ("number_frequencies"))
			time = new_nc.createVariable("time", "f8", ("time"))
			rain_flag = new_nc.createVariable("rain_flag", "i4", ("time"))
			elev_angle = new_nc.createVariable("elevation_angle", "f4", ("time"))
			azim_angle = new_nc.createVariable("azimuth_angle", "f4", ("time"))
			TB = new_nc.createVariable("TBs", "f8", ("time", "number_frequencies"))
			integr_time = new_nc.createVariable("integration_time_per_sample", "i4")
			ultra_sampling_factor = new_nc.createVariable("ultra_sampling_factor", "i4")

			# variable attributes:
			freq.long_name = "Channel frequency"
			freq.units = "GHz"
			time.long_name = "sample time"
			time.units = "seconds since 1970-01-01 00:00:00"
			time.comment = "reference time zone indicated in field time_reference"
			rain_flag.description = "0 = no rain, 1 = raining"
			elev_angle.long_name = "viewing_elevation angle"
			elev_angle.units = "degrees (-90 - +180)"
			elev_angle.comment = "-90 is blackbody view, 0 is horizontal view (red arrow), 90 is zenith view, 180 is horizontal view (2nd quadrant)"
			azim_angle.units = "degrees (0 - 360)"
			TB.description = "Brightness temperature for K, V, W, F and G band. F and G band are double side band averaged."
			TB.units = "K"
			integr_time.long_name = "integration time for each sample"
			integr_time.units = "seconds"
			ultra_sampling_factor.long_name = "flag indicating the time zone reference"
			ultra_sampling_factor.units = "unitless"
			ultra_sampling_factor.comment = "0 = local time, 1 = UTC"

			# write into variables:
			freq[:] = mwr_dict_all['frequencies']
			time[:] = time_axis
			rain_flag[:] = mwr_dict_all['rain_flag']
			elev_angle[:] = mwr_dict_all['elevation_angle']
			azim_angle[:] = mwr_dict_all['azimuth_angle']
			TB[:,:] = mwr_dict_all['TBs']
			integr_time[:] = [mwr_module_dict[mod_key]['integration_time_per_sample'] for mod_key, module_active in module_exist.items() if module_active][0]
			ultra_sampling_factor[:] = [mwr_module_dict[mod_key]['ultra_sampling_factor'] for mod_key, module_active in module_exist.items() if module_active][0]
