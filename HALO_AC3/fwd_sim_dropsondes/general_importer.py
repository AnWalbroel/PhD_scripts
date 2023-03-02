from __future__ import print_function, division
import numpy as np
import datetime as dt
import netCDF4 as nc
import pandas as pd
import xarray as xr
import copy
import pdb
import sys

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_epochtime


# Reading Geet George's dropsonde files (netcdf): (only reads the basic variables; hard coded)
# Returns dictionary including: pressure, temperature, relative humidity, timestamp, launch altitude, height levels ############
# All meteorological variables will be converted to SI units
def readNC(filename):
	file_nc = nc.Dataset(filename)

	# temp in K; press in Pa; relhum in %, launch_time in unixtime (must be treated specially, use its UNITS),
	# height in m, lat in deg north, lon in deg east, u (eastward: > 0) & v (northward: > 0) in m/s, w (vertical wind: upward: > 0),
	#
	dropsonde_dict = {\
	'tdry': np.asarray(file_nc.variables['tdry']) + 273.15, \
	'pres': np.asarray(file_nc.variables['pres'])*100, \
	'rh': np.asarray(file_nc.variables['rh']), \
	'launch_time': np.asarray(file_nc.variables['launch_time']), \
	'alt': np.asarray(file_nc.variables['alt']), \
	'lat': np.asarray(file_nc.variables['lat']), \
	'lon': np.asarray(file_nc.variables['lon']), \
	'u_wind': np.asarray(file_nc.variables['u_wind']), \
	'v_wind': np.asarray(file_nc.variables['v_wind']), \
	'w_wind': np.asarray(file_nc.variables['w_wind']) \
	}

	# handling the launch time: converting it to unixtime: seconds since 1970-01-01 00:00:00 UTC
	time_base = dt.datetime.strptime(file_nc.variables['launch_time'].units[14:], "%Y-%m-%d %H:%M:%S") # time base given in the units attribute
	dropsonde_dict['launch_time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['launch_time']

	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = dropsonde_dict['tdry']
	dropsonde_dict['P'] = dropsonde_dict['pres']
	dropsonde_dict['RH'] = dropsonde_dict['rh']
	dropsonde_dict['Z'] = dropsonde_dict['alt']

	return dropsonde_dict


# This function works similar to readNC but loads all variables into the dictionary that will be returned:
# Variables with a designated unit that is not an SI unit will be converted to SI units. Rel. humidity will be in % though.
# Timestamps will be given in seconds since 1970-01-01 00:00:00 UTC
def readNCraw(filename, verbose=False):
	file_nc = nc.Dataset(filename)

	dropsonde_dict = dict()
	for nc_keys in file_nc.variables.keys():
		nc_var = file_nc.variables[nc_keys]
		dropsonde_dict[nc_keys] = np.asarray(nc_var)

		# converting units: time stamps will be handled seperately.
		if hasattr(nc_var, 'units'):
			if nc_var.units == 'degC':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys] + 273.15
				if verbose: print("From degC to K: " + str(nc_keys))
			elif nc_var.units == 'hPa':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]*100
				if verbose: print("From hPa to Pa: " + str(nc_keys))
			elif nc_var.units == 'gram/kg':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]/1000
				if verbose: print("From g/kg to kg/kg: " + str(nc_keys))

	time_base = dt.datetime.strptime(file_nc.variables['launch_time'].units[14:], "%Y-%m-%d %H:%M:%S") # time base given in the units attribute
	dropsonde_dict['launch_time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['launch_time']
	dropsonde_dict['reference_time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['reference_time']
	if verbose: print("\n")

	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = dropsonde_dict['tdry']
	dropsonde_dict['P'] = dropsonde_dict['pres']
	dropsonde_dict['RH'] = dropsonde_dict['rh']
	dropsonde_dict['Z'] = dropsonde_dict['alt']

	return dropsonde_dict


# This function works similar to readNCraw but suits the raw dropsonde data with ending PRAW.nc:
# Variables with a designated unit that is not an SI unit will be converted to SI units. Rel. humidity will be in % though.
# Timestamps will be given in seconds since 1970-01-01 00:00:00 UTC
def readrawNCraw(filename, verbose=False):
	file_nc = nc.Dataset(filename)

	dropsonde_dict = dict()
	dropsonde_dict['fillValues'] = dict()
	for nc_keys in file_nc.variables.keys():
		nc_var = file_nc.variables[nc_keys]
		dropsonde_dict[nc_keys] = np.asarray(nc_var)

		if hasattr(nc_var, 'missing_value'):
			dropsonde_dict['fillValues'][nc_keys] = nc_var.missing_value

			# type of the nc_var:
			ncvar_type = type(dropsonde_dict[nc_keys][0])

			# find where the current variable has missing values and set them to nan:
			missing_idx = np.argwhere(dropsonde_dict[nc_keys] == dropsonde_dict['fillValues'][nc_keys])

			if ((ncvar_type == np.float32) or (ncvar_type == np.float64)):
				dropsonde_dict[nc_keys][missing_idx] = float('nan')


		# converting units: time stamps will be handled seperately.
		if hasattr(nc_var, 'units'):
			if nc_var.units == 'degC':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys] + 273.15
				if verbose: print("From degC to K: " + str(nc_keys))
			elif nc_var.units == 'hPa':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]*100
				if verbose: print("From hPa to Pa: " + str(nc_keys))
			elif nc_var.units == 'gram/kg':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]/1000
				if verbose: print("From g/kg to kg/kg: " + str(nc_keys))

	time_base = dt.datetime.strptime(file_nc.variables['launch_time'].units[14:-4], "%Y-%m-%d %H:%M:%S") # time base given in the units attribute
	dropsonde_dict['launch_time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['launch_time']
	dropsonde_dict['reference_time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['reference_time']
	dropsonde_dict['time'] = (time_base - dt.datetime(1970,1,1)).total_seconds() + dropsonde_dict['time']
	if verbose: print("\n")

	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = dropsonde_dict['tdry']
	dropsonde_dict['P'] = dropsonde_dict['pres']
	dropsonde_dict['RH'] = dropsonde_dict['rh']
	dropsonde_dict['Z'] = dropsonde_dict['gpsalt']

	return dropsonde_dict


def import_dropsonde_QC(filename):

	"""
	Imports level 1 dropsondes from HALO-AC3, dropped from HALO. Indicated as level 1 files with
	"QC" in the filename. Dictionary with data is returned.

	Parameters:
	----------
	filename : str
		Path and filename of netCDF files of dropsondes.
	"""

	FILE_DS = xr.open_dataset(filename)

	dropsonde_dict = dict()
	dropsonde_dict['fillValues'] = dict()

	# loop through data vars to convert to SI units
	for data_var in FILE_DS.data_vars:
		# convert units:
		if hasattr(FILE_DS[data_var], "units"):
			if FILE_DS[data_var].units == "degC":
				FILE_DS[data_var] += 273.15
			elif FILE_DS[data_var].units == "hPa":
				FILE_DS[data_var] *= 100
			elif FILE_DS[data_var].units == "gram/kg":
				FILE_DS[data_var] /= 1000

	dropsonde_dict['reference_time'] = numpydatetime64_to_epochtime(FILE_DS.reference_time.values[0])
	dropsonde_dict['time'] = numpydatetime64_to_epochtime(FILE_DS.time.values)
	dropsonde_dict['launch_time'] = numpydatetime64_to_epochtime(FILE_DS.launch_time.values)

	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = FILE_DS.tdry.values
	dropsonde_dict['P'] = FILE_DS.pres.values
	dropsonde_dict['RH'] = FILE_DS.rh.values
	dropsonde_dict['Z'] = FILE_DS.gpsalt.values
	dropsonde_dict['u_wind'] = FILE_DS.u_wind.values
	dropsonde_dict['v_wind'] = FILE_DS.v_wind.values
	dropsonde_dict['lat'] = FILE_DS.lat.values
	dropsonde_dict['lon'] = FILE_DS.lon.values
	dropsonde_dict['reference_lat'] = FILE_DS.reference_lat.values
	dropsonde_dict['reference_lon'] = FILE_DS.reference_lon.values
	dropsonde_dict['reference_rh'] = FILE_DS.reference_rh.values
	dropsonde_dict['reference_tdry'] = FILE_DS.reference_tdry.values
	dropsonde_dict['reference_pres'] = FILE_DS.reference_pres.values
	dropsonde_dict['reference_alt'] = FILE_DS.reference_alt.values
	
	return dropsonde_dict


# This importer routine is designed for JOANNE2.0 data to convert T to Kelvin, P to Pa, ...
# Timestamp is already in Unixtime (seconds since 1970-01-01 00:00:00 UTC) and will be converted
# to seconds since 1970-01-01 00:00:00 UTC:
def readNCrawJOANNE2(filename, verbose=False):
	file_nc = nc.Dataset(filename)

	if verbose: print("Working on '" + filename + "'.")

	dropsonde_dict = dict()
	dropsonde_dict['fillValues'] = dict()
	for nc_keys in file_nc.variables.keys():
		nc_var = file_nc.variables[nc_keys]
		# Convert any time units to seconds since 1970-01-01 00:00:00.

		if (
			'time' in nc_keys.lower() or
			(hasattr(nc_var, 'standard_name') and 'time' in nc_var.standard_name.lower()) or
			(hasattr(nc_var, 'long_name') and 'time' in nc_var.long_name.lower())
		):
			dates = nc.num2date(nc_var[:], nc_var.units)
			dropsonde_dict[nc_keys] = nc.date2num(dates, 'seconds since 1970-01-01 00:00:00')
		else:
			dropsonde_dict[nc_keys] = np.asarray(nc_var).astype(np.float64)

		if hasattr(nc_var, '_FillValue'):
			dropsonde_dict['fillValues'][nc_keys] = nc_var._FillValue

			# type of the nc_var:
			ncvar_type = type(dropsonde_dict[nc_keys][0])

			# find where the current variable has missing values and set them to nan:
			missing_idx = np.argwhere(dropsonde_dict[nc_keys] == dropsonde_dict['fillValues'][nc_keys])

			if ((ncvar_type == np.float32) or (ncvar_type == np.float64)):
				dropsonde_dict[nc_keys][missing_idx] = float('nan')


		# convert units:
		if hasattr(nc_var, 'units'):
			if nc_var.units == 'degree_Celsius':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys] + 273.15		# deg C to Kelvin
			elif nc_var.units == 'hPa':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]*100

	dropsonde_dict['time'] = np.rint(dropsonde_dict['time']).astype(float)

	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = dropsonde_dict['T']
	dropsonde_dict['P'] = dropsonde_dict['p']
	dropsonde_dict['RH'] = dropsonde_dict['rh']
	dropsonde_dict['Z'] = dropsonde_dict['height']
	dropsonde_dict['u_wind'] = np.sin(np.pi*dropsonde_dict['wdir']/180) * dropsonde_dict['wspd']
	dropsonde_dict['v_wind'] = np.cos(np.pi*dropsonde_dict['wdir']/180) * dropsonde_dict['wspd']

	return dropsonde_dict


# This importer routine is designed for JOANNE3.0 data.
# Timestamp is already in Unixtime (seconds since 1970-01-01 00:00:00 UTC) and will be converted
# to seconds since 1970-01-01 00:00:00 UTC:
def readNCrawJOANNE3(	filename,
						keys=['alt', 'launch_time', 'lat', 'lon',
							'p', 'ta', 'rh', 'u', 'v', 'flight_lat',
							'flight_lon', 'flight_height', 'platform'],
						verbose=False):
	file_nc = nc.Dataset(filename)

	if verbose: print("Working on '" + filename + "'.")

	if 'alt' in keys and 'alt_bnds' not in keys:
		keys.append('alt_bnds')

	if keys == '':
		keys = file_nc.variables.keys()

	dropsonde_dict = dict()
	dropsonde_dict['fillValues'] = dict()
	for nc_keys in keys:

		if not nc_keys in file_nc.variables.keys():
			raise KeyError("I have no memory of this key '%s'. Key not found in JOANNE V3 dropsonde file." % nc_keys)

		nc_var = file_nc.variables[nc_keys]

		# Convert any time units to seconds since 1970-01-01 00:00:00.
		if (
			'time' in nc_keys.lower() or
			(hasattr(nc_var, 'standard_name') and 'time' in nc_var.standard_name.lower()) or
			(hasattr(nc_var, 'long_name') and 'time' in nc_var.long_name.lower())
		):
			dates = nc.num2date(nc_var[:], nc_var.units)
			dropsonde_dict[nc_keys] = nc.date2num(dates, 'seconds since 1970-01-01 00:00:00')
		elif nc_keys != 'platform': # because that key is a string
			dropsonde_dict[nc_keys] = np.asarray(nc_var).astype(np.float64)
		else:
			dropsonde_dict[nc_keys] = np.asarray(nc_var)

		if hasattr(nc_var, '_FillValue'):
			dropsonde_dict['fillValues'][nc_keys] = nc_var._FillValue
			dropsonde_dict[nc_keys][dropsonde_dict[nc_keys] == nc_var._FillValue] = float('nan')


	# Get real altitude from alt boundaries (at least necessary in version Level_3_v0.9.2)
	dropsonde_dict['alt'] = np.mean(dropsonde_dict['alt_bnds'], -1)
	dropsonde_dict.pop('alt_bnds')

	dropsonde_dict['launch_time'] = np.rint(dropsonde_dict['launch_time']).astype(float)


	# Convert to internal convention: Temperature = T, Pressure = P, relative humidity = RH, altitude = Z
	dropsonde_dict['T'] = dropsonde_dict['ta']
	dropsonde_dict['P'] = dropsonde_dict['p']
	dropsonde_dict['rh'] = dropsonde_dict['rh']*1.06													###### multiplication by 1.06, suggested by Geet George (2021-05-14)
	dropsonde_dict['rh'][dropsonde_dict['rh'] > 1.0] = 1.0												###### limit RH to 100 %
	dropsonde_dict['RH'] = dropsonde_dict['rh']*100
	dropsonde_dict['Z'] = dropsonde_dict['alt']
	dropsonde_dict['u_wind'] = dropsonde_dict['u']
	dropsonde_dict['v_wind'] = dropsonde_dict['v']

	launch_datetime64 = np.datetime64('1970-01-01') + np.timedelta64(1, 's') * dropsonde_dict['launch_time']

	# faulty temperature measurements at top (suspicious temperature inversion)
	faulty_mask = launch_datetime64 == np.datetime64('2020-01-24T10:35:05')
	assert faulty_mask.sum() == 1, 'There should me one match in version Level_3_v0.9.2'
	faulty_index = np.where(faulty_mask)[0]

	if verbose: print("Remove faulty temperature measurements at top in i_time =", faulty_index)
	mask = dropsonde_dict['alt'] > 8680
	dropsonde_dict['T'][faulty_index, mask] = np.nan
	dropsonde_dict['P'][faulty_index, mask] = np.nan
	dropsonde_dict['RH'][faulty_index, mask] = np.nan
	dropsonde_dict['u_wind'][faulty_index, mask] = np.nan
	dropsonde_dict['v_wind'][faulty_index, mask] = np.nan

	# humidity profile too dry
	faulty_mask = launch_datetime64 == np.datetime64('2020-01-31T17:57:36')
	assert faulty_mask.sum() == 1, 'There should me one match in version Level_3_v0.9.2'
	faulty_index = np.where(faulty_mask)[0]

	if verbose: print("Remove sonde with too dry humidity in i_time =", faulty_index)
	dropsonde_dict['T'][faulty_index, :] = np.nan
	dropsonde_dict['P'][faulty_index, :] = np.nan
	dropsonde_dict['RH'][faulty_index, :] = np.nan
	dropsonde_dict['u_wind'][faulty_index, :] = np.nan
	dropsonde_dict['v_wind'][faulty_index, :] = np.nan

	return dropsonde_dict


# This function works similar to readNCraw but is adapted to work for v01 dropsonde files:
def readNCraw_V01(filename, verbose=False):
	file_nc = nc.Dataset(filename)

	dropsonde_dict = dict()
	for nc_keys in file_nc.variables.keys():
		nc_var = file_nc.variables[nc_keys]
		dropsonde_dict[nc_keys] = np.asarray(nc_var)

		# converting units: time stamps will be handled seperately.
		if hasattr(nc_var, 'units'):
			if nc_var.units == 'degC':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys] + 273.15
				if verbose: print("From degC to K: " + str(nc_keys))
			elif nc_var.units == 'hPa':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]*100
				if verbose: print("From hPa to Pa: " + str(nc_keys))
			elif nc_var.units == 'gram/kg':
				dropsonde_dict[nc_keys] = dropsonde_dict[nc_keys]/1000
				if verbose: print("From g/kg to kg/kg: " + str(nc_keys))


	if verbose: print("\n")

	return dropsonde_dict


def import_GHRSST(filename, keys=''):	# imports SST data from GHRSST data base. keys can be assigned; otherwise all keys will be read in:
	GHRSST_dict = dict()
	file_nc = nc.Dataset(filename)

	if keys == '':
		keys = file_nc.variables.keys()

	for key in keys:
		if key == 'lon' or key == 'lat':
			GHRSST_dict[key] = np.asarray(file_nc.variables[key]).astype(float)		# because lat, lon are single float; will be converted to double float

		elif key == 'analysed_sst':	# just renaming the array because analysed_sst sounds too fancy
			GHRSST_dict['SST'] = np.asarray(file_nc.variables[key])

		elif key == 'time':	# convert to unixtime:
			GHRSST_dict[key] = np.asarray(file_nc.variables[key]).astype(float)
			time_base = dt.datetime.strptime(file_nc.variables[key].units[14:], "%Y-%m-%d %H:%M:%S") # time base given in the units attribute
			GHRSST_dict[key] = (time_base - dt.datetime(1970,1,1)).total_seconds() + GHRSST_dict[key]

		else:
			GHRSST_dict[key] = np.asarray(file_nc.variables[key])

	return GHRSST_dict


def import_BAHAMAS_unified(filename, keys=''):	# import data from BAHAMAS measurements on the unified grid from Heike Konow.
	BAHAMAS_dict = dict()
	file_nc = nc.Dataset(filename)

	if keys == '':
		keys = file_nc.variables.keys()

	for key in keys:
		# DID NOT THINK ABOUT ANY SPECIAL CASES FOR IMPORT YET because for now I'll most likely use 'time' and 'altitude' only
		BAHAMAS_dict[key] = np.asarray(file_nc.variables[key])		# luckily, all variables are float and time is in unixtime

	return BAHAMAS_dict


def import_BAHAMAS_raw(filename):

	"""
	Import data from BAHAMAS measurements.

	Parameters:
	-----------
	filename : str
		Path and file name of raw BAHAMAS data (as netCDF).
	"""

	BAHAMAS_dict = dict()
	FILE_DS = xr.open_dataset(filename)

	BAHAMAS_dict['time'] = numpydatetime64_to_epochtime(FILE_DS.TIME.values)
	BAHAMAS_dict['altitude'] = FILE_DS.IRS_ALT.values
	# BAHAMAS_dict['ta'] = FILE_DS.TAT.values 
	BAHAMAS_dict['ta'] = FILE_DS.TS.values		# in K
	BAHAMAS_dict['p'] = FILE_DS.PS.values*100	# in Pa
	BAHAMAS_dict['rh'] = FILE_DS.RELHUM.values	# in %

	return BAHAMAS_dict


def import_radar_nc(filename, verbose=False): # imports stuff from the uniform radar files
	radar_dict = dict()
	import xarray
	ds = xarray.open_dataset(filename)

	keys = ['dBZ', 'height', 'time']

	if verbose:
		print("Importing variables " + str(keys) + " from file '" + filename + "'.\n")
		print("You may ignore the warning: 'RuntimeWarning: invalid value encountered in greater' ... it just occurs because there are nan values in the mwr TBs.\n")

	for key in keys:
		radar_dict[key] = np.asarray(ds[key].values)

	if not radar_dict['height'][0] < 30:
		raise ValueError('Height does not start at surface as expected.')

	# avoid surface clutter
	radar_dict['dBZ'][:, :5] = np.nan
	# convert time to unix time. (time and its units are decoded by xarray)
	assert np.issubdtype(radar_dict['time'].dtype, np.datetime64)
	radar_dict['time'] = (radar_dict['time'] - np.datetime64('1970-01-01 00:00:00'))/np.timedelta64(1, 's')

	return radar_dict


def import_mwr_nc(filename, keys='', verbose=False): # imports stuff from the concatenated mwr files (v01, ): keys to be imported can be assigned. Otherwise all variables will be read in:
	mwr_dict = dict()
	file_nc = nc.Dataset(filename)

	if keys == '':
		keys = file_nc.variables.keys()

	if verbose:
		print("Importing variables " + str(keys) + " from file '" + filename + "'.\n")
		print("You may ignore the warning: 'RuntimeWarning: invalid value encountered in greater' ... it just occurs because there are nan values in the mwr TBs.\n")

	for key in keys:
		mwr_dict[key] = np.asarray(file_nc.variables[key])

	if 'uniform' in filename or 'unified' in filename:
		if filename.endswith('v0.8.nc'):
			correct_time_offsets(mwr_dict)
		# add empty channel 26:
		ch26 = np.nan + np.empty(mwr_dict['tb'].shape[0])
		mwr_dict['tb'] = np.concatenate([mwr_dict['tb'], ch26[:, np.newaxis]], axis=1)
		# rename from uniform scheme to RPG scheme
		mwr_dict['TBs'] = mwr_dict['tb']

	return mwr_dict


def correct_time_offsets(mwr_dict):
	"""in-place correct time offsets that were in the uniform data until version v0.8"""
	band_KV = np.s_[0:14]
	band_WF = np.s_[14:19]
	band_G = np.s_[19:]

	def shift(tb, o, s, band):
		"""
		shift tb timeseriesinplace.
		o: offset in time (1st dimension)
		s: slice that should be shifted
		band: frequency band that is affected (2nd dimension)
		"""
		roll = np.roll(tb[s, band], o, axis=0)
		if o > 0:
			roll[:o, :] = np.nan # mask the elements that roll beyond the last position wre re-introduced at the first.
		elif o < 0:
			roll[o:, :] = np.nan # mask the elements that roll beyond the first position wre re-introduced at the end.

		tb[s, band] = roll

	date = dt.datetime.utcfromtimestamp(mwr_dict['time'][0]).strftime('%Y%m%d')

	if date == '20200119':
		o = -141
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_WF)

	elif date == '20200122':
		pass

	elif date == '20200124':
		o = -1
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_KV)
		o = -2
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -7
		s = np.s_[:21000]
		shift(mwr_dict['tb'], o, s, band_G)
		o = -3
		s = np.s_[21000:29400]
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200126':
		s = np.s_[0: 23350]
		o = -2
		shift(mwr_dict['tb'], o, s, band_KV)
		o = 1
		shift(mwr_dict['tb'], o, s, band_WF)

		s = np.s_[23700:]
		o=-3
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200128':
		pass

	elif date == '20200130':
		pass

	elif date == '20200131':
		o = -1
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -3
		s = np.s_[21200:]
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200202':
		o = -2
		s = np.s_[4000:6210]
		shift(mwr_dict['tb'], o, s, band_KV)

		o = -3
		s = np.s_[:2000]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -1
		s = np.s_[2000:4000]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -3
		s = np.s_[4000:6255]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -2
		s = np.s_[6255:6720]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -1
		s = np.s_[6720:]
		shift(mwr_dict['tb'], o, s, band_WF)

		o = -2
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200205':
		o = -3
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_KV)
		o = -4
		shift(mwr_dict['tb'], o, s, band_WF)


	elif date == '20200207':
		o = -2
		s = np.s_[21300:23260]
		shift(mwr_dict['tb'], o, s, band_KV)
		o = 2
		s = np.s_[:22400]
		shift(mwr_dict['tb'], o, s, band_WF)

	elif date == '20200209':
		o = -2
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200211':
		o = -2
		s = np.s_[:22900]
		shift(mwr_dict['tb'], o, s, band_KV)
		o = -5
		s = np.s_[21000:23500]
		shift(mwr_dict['tb'], o, s, band_G)

	elif date == '20200213':
		pass

	elif date == '20200215':
		o = -2
		s = np.s_[2034:28757]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -1
		s = np.s_[20484:]
		shift(mwr_dict['tb'], o, s, band_KV)

	elif date == '20200218':
		o = -1
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_WF)
		o = -4
		s = np.s_[:]
		shift(mwr_dict['tb'], o, s, band_G)

	else:
		raise ValueError(f'Unknown time offsets on date "{date}".')


def import_DSpam_nc(filename, keys='', withDSBA=True, alldims=True):	# imports stuff from PAMTRA simualted dropsondes
	# keys to be imported may be assigned. Otherwise all variables will be read in.
	# withDSBA decides whether or not the double side bands (F and G band) will be averaged. If True: double side band averaging will be performed.

	DSpam_dict = dict()
	file_nc = nc.Dataset(filename)
	grid_x = 1		# should always be 1 because in HALO_dropsonde_to_TB, the dropsondes were aligned with y
	grid_y = file_nc.dimensions['grid_y'].size
	grid_z = file_nc.dimensions['outlevel'].size		# outlevel as z axis

	if keys == '':
		keys = file_nc.variables.keys()

	print("Importing variables " + str(keys) + " from file '" + filename + "'.\n")

	for key in keys:

		DSpam_dict[key] = np.asarray(file_nc.variables[key])

		if DSpam_dict[key].shape == (1,grid_y):	# 2D arrays
			DSpam_dict[key] = DSpam_dict[key][0,:]
		if DSpam_dict[key].shape == (1,grid_y,grid_z):	# 3D arrays
			DSpam_dict[key] = DSpam_dict[key][0,:,:]


	if 'tb' in keys:
		# filter the unnecessary dimensions (like angles, grid_x, passive polarization)
		# dimensions: grid_x, grid_y, outlevel, angles, frequency, passive polarization
		if alldims:
			DSpam_dict['tb'] = DSpam_dict['tb'][0,:,:,0,:].mean(axis=-1)
		else:
			DSpam_dict['tb'] = DSpam_dict['tb'][0,:,0,0,:].mean(axis=-1)

		if withDSBA:	# double side band averaging (because HAMP MWR also does this)
			TB_old = DSpam_dict['tb']
			TB_new = copy.deepcopy(TB_old)

			# F band:
			TB_new[...,15] = (TB_old[...,18] + TB_old[...,19])/2
			TB_new[...,16] = (TB_old[...,17] + TB_old[...,20])/2
			TB_new[...,17] = (TB_old[...,16] + TB_old[...,21])/2
			TB_new[...,18] = (TB_old[...,15] + TB_old[...,22])/2
			# G band:
			TB_new[...,19] = (TB_old[...,29] + TB_old[...,30])/2
			TB_new[...,20] = (TB_old[...,28] + TB_old[...,31])/2
			TB_new[...,21] = (TB_old[...,27] + TB_old[...,32])/2
			TB_new[...,22] = (TB_old[...,26] + TB_old[...,33])/2
			TB_new[...,23] = (TB_old[...,25] + TB_old[...,34])/2
			TB_new[...,24] = (TB_old[...,24] + TB_old[...,35])/2
			TB_new[...,25] = (TB_old[...,23] + TB_old[...,36])/2

			# overwrite the edited entries only:
			TB_new = TB_new[...,0:26]
			DSpam_dict['tb'] = TB_new

			# create another frequency array that respects the double side band averaging:
			old_freq = np.asarray(file_nc.variables['frequency'])
			new_freq = copy.deepcopy(old_freq)
			new_freq[15:19] = old_freq[19:23]		# to match the indicated frequencies of HAMP MWR
			new_freq[19:26] = old_freq[30:37]
			DSpam_dict['frequency_dsba'] = new_freq[0:26]

	return DSpam_dict


def import_DSpam_nc_XR(filename, keys='', withDSBA=True, alldims=True):	# imports stuff from PAMTRA simualted dropsondes
	# keys to be imported may be assigned. Otherwise all variables will be read in.
	# withDSBA decides whether or not the double side bands (F and G band) will be averaged. If True: double side band averaging will be performed.

	# Create new xarray DATASET from netcdf file:
	DSpam_ds = xr.open_dataset(filename)

	if keys == '':
		keys = DSpam_ds.keys()

	print("Importing variables " + str(keys) + " from file '" + filename + "'.\n")


	# eliminate redundant dimension (grid_x):
	# only squeeze along grid_x because otherwise 'outlevel' may also be squeezed unintentionally if only
	# one outlevel exists
	DSpam_ds = DSpam_ds.squeeze(dim='grid_x').drop('grid_x')

	# convert time to seconds since 1970-01-01 00:00:00 UTC:
	DSpam_ds['datatime'].values = (DSpam_ds.datatime.values - np.datetime64(0,'s')) / np.timedelta64(1,'s')


	if 'tb' in keys:
		# filter the unnecessary dimensions (like angles, grid_x, passive polarization)
		# dimensions: grid_y, outlevel, angles, frequency, passive polarization
		# remaining: grid_y, outlevel, frequency
		if alldims:
			# datatime = DSpam_ds.datatime			# NECESSARY IF datatime is a numpy.datetime64 object
			DSpam_ds = DSpam_ds.isel(angles=0).mean('passive_polarisation')
				# -> removes the redundant dimensions from the whole data set at once
				# (but angles is still accessible if you want to know which angle was used)
			# DSpam_ds['datatime'] = datatime		# NECESSARY IF datatime is a numpy.datetime64 object

		else:
			# # datatime = DSpam_ds.datatime		# NECESSARY IF datatime is a numpy.datetime64 object
			DSpam_ds = DSpam_ds.isel(angles=0, outlevel=0).mean('passive_polarisation')
			# # DSpam_ds['datatime'] = datatime		# NECESSARY IF datatime is a numpy.datetime64 object

		if withDSBA:	# double side band averaging (because HAMP MWR also does this)
			TB_old = DSpam_ds.tb
			TB_new = copy.deepcopy(TB_old)

			# F band:
			TB_new[...,15] = (TB_old[...,18] + TB_old[...,19])/2
			TB_new[...,16] = (TB_old[...,17] + TB_old[...,20])/2
			TB_new[...,17] = (TB_old[...,16] + TB_old[...,21])/2
			TB_new[...,18] = (TB_old[...,15] + TB_old[...,22])/2
			# G band:
			TB_new[...,19] = (TB_old[...,29] + TB_old[...,30])/2
			TB_new[...,20] = (TB_old[...,28] + TB_old[...,31])/2
			TB_new[...,21] = (TB_old[...,27] + TB_old[...,32])/2
			TB_new[...,22] = (TB_old[...,26] + TB_old[...,33])/2
			TB_new[...,23] = (TB_old[...,25] + TB_old[...,34])/2
			TB_new[...,24] = (TB_old[...,24] + TB_old[...,35])/2
			TB_new[...,25] = (TB_old[...,23] + TB_old[...,36])/2

			# overwrite the edited entries only:
			TB_new = TB_new[...,0:26]

			# generate new frequency as coordinate:
			frq_dsba = DSpam_ds.frequency
			frq_dsba = np.append(np.append(frq_dsba[0:15], frq_dsba[19:23]), frq_dsba[30:37])

			# use the new frequency to reindex the DataArray TB_new:
			TB_new['frequency'] = frq_dsba

			# and reindex the Dataset:
			DSpam_ds = DSpam_ds.reindex({'frequency': frq_dsba})

			# and finally replace the TB values with the double side band averaged ones:
			DSpam_ds['tb'].loc[dict(frequency=frq_dsba)] = TB_new


			# # # # # #Another option how to replace the TBs:
			# # # # # # generate new DataArray and replace 'tb' in the DataSet DSpam_ds:
			# # # # # TB_new_DA = xr.DataArray(TB_new.values,
				# # # # # dims=('grid_y', 'outlevel', 'frequency_dsba'),
				# # # # # coords={'grid_y': DSpam_ds.coords['grid_y'].values, 'outlevel': DSpam_ds.coords['outlevel'].values, 'frequency_dsba': frq_dsba})

			# # # # # DSpam_ds['tb'] = TB_new_DA

			# # # # # # Remove the non-dsba frequency:
			# # # # # DSpam_ds = DSpam_ds.drop('frequency')

			# # # # # # RENAME:
			# # # # # DSpam_ds = DSpam_ds.rename({'frequency_dsba': 'frequency'})

	return DSpam_ds


def import_TB_stat(filename, keys=''):	# import variables saved for the TB statistics:
	# keys to be imported may be assigned. Otherwise all variables will be read in.

	TB_stat_dict = dict()
	file_nc = nc.Dataset(filename)

	if keys == '':
		keys = file_nc.variables.keys()

	for key in keys:
		if key == 'date':
			continue													###			DOES NOT WORK YET 			###

		else:
			TB_stat_dict[key] = np.asarray(file_nc.variables[key])

	return TB_stat_dict


def import_sonde_raw_P(filename): # import raw dropsonde files with ending _P.<some number>
	# returns the data and variable information (name + units) in dictionaries

	headersize = 4		# 4 lines: line 2 and 3: for variable names; line 4: units
	footersize = 19

	fileHandler = open(filename, 'r')
	listOfLines = list()				# will contain all lines as list

	for line in fileHandler:
		current_line = line.strip().split(' ')		# split by spaces ... but there are sometimes more than one spaces. So, remove all of them

		while '' in current_line:
			current_line.remove('')

		listOfLines.append(current_line)

	# define the footer (end of file information):
	footer = listOfLines[-footersize:]

	# delete footer from listOfLines:
	del listOfLines[-footersize:]

	# get some more auxiliary information: e.g. sonde ID from the third column or from footer:
	sonde_ID = float(listOfLines[headersize+1][2])
	launch_time = (dt.datetime.strptime(footer[3][5] + footer[3][6], "%Y-%m-%d,%H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
	launch_time_string = dt.datetime.strptime(footer[3][5] + footer[3][6], "%Y-%m-%d,%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")

	# pre-launch meteorological conditions:
	pre_launch = {
	'T0': float(footer[10][7]),		# temperature in deg C
	'T0_units': footer[10][8][:-1],
	'P0': float(footer[10][5]),		# air pressure in hPa or mb, respectively
	'P0_units': footer[10][6][:-1],
	'D0': float(footer[10][9]),		# dewpoint in deg C
	'D0_units': footer[10][10][:-1],
	'RH0': float(footer[10][11]), 	# relative humidity in %
	'RH0_units': footer[10][12],
	'WindDir0': float(footer[11][5]),	# wind direction in deg (whyever it uses a convention that allows negative wind direction)
	'WindDir0_units': footer[11][6][:-1],
	'WindSpeed0': float(footer[11][7]), # wind speed in m s^-1
	'WindSpeed0_units': footer[11][8],
	'lon': float(footer[12][5]), 	# longitude in deg east
	'lon_units': footer[12][6][:-1],
	'lat': float(footer[12][7]),	# latitude in deg north
	'lat_units': footer[12][8][:-1],
	'aircraft_alt': float(footer[12][9]), 	# aircraft altitude in m
	'aircraft_alt_units': footer[12][10][:-1]
	}

	# check for non-existing pre-launch values:
	for prekey in pre_launch.keys():
		if pre_launch[prekey] == -999.0:
			pre_launch[prekey] = float('nan')


	# now listOfLines is a list that contains a list for each single line.
	# elements 0, 1 and of each line are merely sonde name & sonde status & sonde ID. No need to take them. So... delete them
	for line in listOfLines:
		# need to make an exception for line 1 because it contains a blank spot
		if listOfLines.index(line) != 1:
			del line[0:3]

		else:
			del line[0:2]


	# define the header out of which we'll extract variable names
	header = listOfLines[0:headersize]

	# from the header, line 2 + 3 [1, 2 in python] we can create variable names: from line 4 (3 in python) we create units:
	variable_names = [header_1 + '_' + header_2 for header_1, header_2 in zip(header[1], header[2])]
	raw_units = [uni for uni in header[3]]


	# check if all variable names have got attributes or if something hasn't been caught well:
	assert len(variable_names) == len(raw_units)

	# composed into dictionary:
	aux_dict = {
		'variable_names': variable_names,
		'raw_units': raw_units,
		'pre_launch': pre_launch,
		'launch_time': launch_time,
		'launch_time_string': launch_time_string,
		'sondeID': int(sonde_ID)
		}


	# actual data:
	data_block = listOfLines[headersize+2:]

	ndata = len(data_block)

	# assign meteorological variables:
	# handle non-existent measurements (fill value = ...):
	fillVal_dict = {
		'Air_Press': 9999.00,
		'Air_Temp': 99.00,
		'Geopoten_Altitude': 99999.00,
		'GPS_Wnd': 0,
		'Rel_Humid': 999.00,
		'GPS_Latitude': 99.000000,
		'GPS_Altitude': 99999.00,
		'Sonde_RH2': 999.00,
		'Sonde_RH1': 999.00,
		'Wind_Error': 99.00,
		'GPS_Snd': 0,
		'Vert_Veloc': 99.00,
		'GPS_Longitude': 999.000000,
		'Wind_Spd': 999.00,
		'Wind_Dir': 999.00
		}

	data_dict = dict()
	for varname in variable_names:
		loc = variable_names.index(varname)

		if (varname == 'UTC_Date') or (varname == 'UTC_Time'):
			data_dict[varname] = np.asarray([data_block[k][loc] for k in range(ndata)])

		else:

			data_dict[varname] = np.asarray([float(data_block[k][loc]) for k in range(ndata)])

			# find indices of non-existent measurements:
			nan_idx = np.argwhere(data_dict[varname] == fillVal_dict[varname])
			data_dict[varname][nan_idx] = float('nan')

	# glue date and time together and convert it to unixtime (seconds since 1970-01-01 00:00:00 UTC):
	data_dict['sonde_time'] = np.asarray([(dt.datetime.strptime(date + " " + time + "0000", "%y%m%d %H%M%S.%f") -
		dt.datetime(1970,1,1)).total_seconds() for date, time in zip(data_dict['UTC_Date'], data_dict['UTC_Time'])])


	return data_dict, aux_dict