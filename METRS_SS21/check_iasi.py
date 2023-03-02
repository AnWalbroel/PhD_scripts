import numpy as np
import xarray as xr
import netCDF4 as nc
import glob
import pdb
import datetime as dt


def cut_useless_variables(DS):
	"""
	Preprocessing the IASI dataset before concatenation.
	Removing undesired dimensions and variables. 

	Parameters:
	-----------
	ds : xarray dataset
		Dataset of IASI data.
	"""

	# Remove some nasty variables:
	"""
	Remaining variables are:
		pressure_levels_temp
		pressure_levels_humidity
		record_start_time
		record_stop_time
		lat
		lon
		atmospheric_temperature
		atmospheric_water_vapor
		surface_temperature
		surface_pressure
		instrument_mode
		flag_cldnes
		flag_iasibad
		flag_itconv
		flag_landsea
		error_data_index
			# temperature_error
			# water_vapour_error
		surface_z				(useful in combination with surface_pressure)
		co_qflag
		co_bdiv

	Remaining dimensions:
		nlt
		nlq
		along_track
		across_track
			# nerr
			# nerrt
			# nerrw
	"""
	useless_vars = ['cloud_formation', 'pressure_levels_ozone', 'surface_emissivity_wavelengths', 
					'degraded_ins_MDR', 'degraded_proc_MDR', 'solar_zenith', 'satellite_zenith', 
					'solar_azimuth', 'satellite_azimuth', 'fg_atmospheric_temperature', 
					'fg_atmospheric_water_vapor', 'fg_atmospheric_ozone', 'fg_surface_temperature', 
					'atmospheric_ozone', 'integrated_water_vapor', 'integrated_ozone', 'integrated_n2o', 
					'integrated_co', 'integrated_ch4', 'integrated_co2', 'surface_emissivity', 
					'number_cloud_formations', 'fractional_cloud_cover', 'cloud_top_temperature',
					'cloud_top_pressure', 'cloud_phase', 'spacecraft_altitude', 'flag_amsubad', 
					'flag_avhrrbad', 'flag_cdlfrm', 'flag_cdltst', 'flag_daynit', 'flag_dustcld', 
					'flag_fgcheck', 'flag_initia', 'flag_mhsbad', 'flag_numit', 'flag_nwpbad', 
					'flag_physcheck', 'flag_retcheck', 'flag_satman', 'flag_sunglnt', 'flag_thicir', 
					'nerr_values', 'ozone_error', 'co_npca', 'co_nfitlayers', 'co_nbr_values', 
					'co_cp_air', 'co_cp_co_a', 'co_x_co', 'co_h_eigenvalues', 'co_h_eigenvectors',
					'temperature_error', 'water_vapour_error']
	DS = DS.drop_vars(useless_vars)

	# useless_dims = ['npct', 'npcw', 'npco', 'nl_co', 'nl_hno3', 'nl_o3', 'nl_so2', 'new', 'nlo', 
					# 'cloud_formations', 'nerro', 'co_nbr', 'neva_co', 'neve_co']
	# DS = DS.squeeze(useless_dims, drop=True)

	return DS

def remove_ETIM(DS):
	DS = DS.drop_vars('ETIM')

	return DS

def create_launch_time(DS):
	
	time_dif = np.diff(DS.time.values)
	where_jump = np.argwhere(np.abs(time_dif) > 3600).flatten()
	launch_time = np.concatenate((np.array([DS.time.values[0]]), DS.time.values[where_jump+1]))

	return xarray.DataArray(launch_time)


def import_single_PS122_mosaic_radiosonde_level2(
	filename,
	keys='all',
	verbose=0):

	"""
	Imports single level 2 radiosonde data created with PANGAEA_tab_to_nc.py 
	('PS122_mosaic_radiosonde_level2_yyyymmdd_hhmmssZ.nc'). Converts to SI units
	and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

	Parameters:
	-----------
	filename : str
		Name (including path) of radiosonde data file.
	keys : list of str or str, optional
		This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
		Specifying 'basic' will load the variables the author consideres most useful for his current
		analysis.
		Default: 'all'
	verbose : int
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	"""
		Loaded values are imported in the following units:
		T: in deg C, will be converted to K
		P: in hPa, will be converted to Pa
		RH: in %, will be converted to [0-1]
		Altitude: in m
		q: in kg kg^-1 (water vapor specific humidity)
		time: in sec since 1970-01-01 00:00:00 UTC
	"""

	file_nc = nc.Dataset(filename)

	if (not isinstance(keys, str)) and (not isinstance(keys, list)):
		raise TypeError("Argument 'key' must be a list of strings or 'all'.")

	if keys == 'all':
		keys = file_nc.variables.keys()
	elif keys == 'basic':
		keys = ['time', 'T', 'P', 'RH', 'q', 'Altitude']

	sonde_dict = dict()
	for key in keys:
		if not key in file_nc.variables.keys():
			raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

		sonde_dict[key] = np.asarray(file_nc.variables[key])
		if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
			return None

		if key in ['Latitude', 'Longitude']:	# only interested in the first lat, lon position
			sonde_dict[key] = sonde_dict[key][0]
		if key == 'IWV':
			sonde_dict[key] = np.float64(sonde_dict[key])

	# convert units:
	if 'RH' in keys:	# from percent to [0, 1]
		sonde_dict['RH'] = sonde_dict['RH']*0.01
	if 'T' in keys:		# from deg C to K
		sonde_dict['T'] = sonde_dict['T'] + 273.15
	if 'P' in keys:		# from hPa to Pa
		sonde_dict['P'] = sonde_dict['P']*100
	if 'time' in keys:	# from int64 to float64
		sonde_dict['time'] = np.float64(sonde_dict['time'])
		sonde_dict['launch_time'] = sonde_dict['time'][0]

	keys = [*keys]		# converts dict_keys to a list
	for key in keys:
		if sonde_dict[key].shape == sonde_dict['time'].shape:
			if key not in ['time', 'Latitude', 'Longitude', 'ETIM', 'Altitude']:
				sonde_dict[key + "_ip"] = np.interp(np.arange(0,15001,5), sonde_dict['Altitude'], sonde_dict[key])
			elif key == 'Altitude':
				sonde_dict[key + "_ip"] = np.arange(0, 15001,5)


	# Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
	renaming = {'T': 'temp', 	'P': 'pres', 	'RH': 'rh',
				'Altitude': 'height', 'h_geom': 'height_geom',
				'Latitude': 'lat', 	'Longitude': 'lon',
				'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
				'Altitude_ip': 'height_ip', 'h_geom_ip': 'height_geom_ip',
				'IWV': 'iwv'}
	for ren_key in renaming.keys():
		if ren_key in sonde_dict.keys():
			sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

	return sonde_dict


def import_single_NYA_RS_radiosonde(
	filename,
	keys='all',
	verbose=0):

	"""
	Imports single NYA-RS radiosonde data for Ny Alesund. Converts to SI units
	and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

	Parameters:
	-----------
	filename : str
		Name (including path) of radiosonde data file.
	keys : list of str or str, optional
		This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
		Specifying 'basic' will load the variables the author consideres most useful for his current
		analysis.
		Default: 'all'
	verbose : int
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	"""
		Loaded values are imported in the following units:
		T: in K
		P: in hPa, will be converted to Pa
		RH: in [0-1]
		Altitude: in m
		time: will be converted to sec since 1970-01-01 00:00:00 UTC
	"""

	file_nc = nc.Dataset(filename)

	if (not isinstance(keys, str)) and (not isinstance(keys, list)):
		raise TypeError("Argument 'key' must be a list of strings or 'all'.")

	if keys == 'all':
		keys = file_nc.variables.keys()
	elif keys == 'basic':
		keys = ['time', 'temp', 'press', 'rh', 'alt']

	sonde_dict = dict()
	for key in keys:
		if not key in file_nc.variables.keys():
			raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

		sonde_dict[key] = np.asarray(file_nc.variables[key])
		if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
			return None

		if key in ['lat', 'lon']:	# only interested in the first lat, lon position
			sonde_dict[key] = sonde_dict[key][0]

	# convert units:
	if 'P' in keys:		# from hPa to Pa
		sonde_dict['P'] = sonde_dict['P']*100
	if 'time' in keys:	# from int64 to float64
		time_unit = file_nc.variables['time'].units
		time_offset = (dt.datetime.strptime(time_unit[-19:], "%Y-%m-%dT%H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
		sonde_dict['time'] = np.float64(sonde_dict['time']) + time_offset
		sonde_dict['launch_time'] = sonde_dict['time'][0]

	keys = [*keys]		# converts dict_keys to a list
	for key in keys:
		if sonde_dict[key].shape == sonde_dict['time'].shape:
			if key not in ['time', 'lat', 'lon', 'alt']:
				sonde_dict[key + "_ip"] = np.interp(np.arange(0,15001,5), sonde_dict['alt'], sonde_dict[key])
			elif key == 'alt':
				sonde_dict[key + "_ip"] = np.arange(0, 15001,5)


	# Renaming variables to a standard convention
	renaming = {'press': 'pres', 'alt': 'height', 'press_ip': 'pres_ip', 'alt_ip': 'height_ip'}
	for ren_key in renaming.keys():
		if ren_key in sonde_dict.keys():
			sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

	return sonde_dict


def import_radiosonde_daterange(
	path_data,
	date_start,
	date_end,
	s_version='level_2',
	with_wind=False,
	verbose=0):

	"""
	Imports radiosonde data 'mossonde-curM1' and concatenates the files into time series x height.
	E.g. temperature profile will have the dimension: n_sondes x n_height

	Parameters:
	-----------
	path_data : str
		Path of radiosonde data.
	date_start : str
		Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	date_end : str
		Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
	s_version : str, optional
		Specifies the radiosonde version that is to be imported. Possible options: 'mossonde',
		'psYYMMDDwHH', 'level_2'. Default: 'level_2' (published by Marion Maturilli)
	with_wind : bool, optional
		This describes if wind measurements are included (True) or not (False). Does not work with
		s_version='psYYMMDDwHH'. Default: False
	verbose : int, optional
		If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
		is printed.
	"""

	if not isinstance(s_version, str): raise TypeError("s_version in import_radiosonde_daterange must be a string.")

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
	date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

	if s_version == 'level_2':
		all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level2*.nc"))

		# inquire the number of radiosonde files (date and time of launch is in filename):
		# And fill a list which will include the relevant radiosonde files.
		radiosondes_nc = []
		for rs_nc in all_radiosondes_nc:
			rs_date = rs_nc[-19:-3]		# date of radiosonde from filename
			yyyy = int(rs_date[:4])
			mm = int(rs_date[4:6])
			dd = int(rs_date[6:8])
			rs_date_dt = dt.datetime(yyyy,mm,dd)
			if rs_date_dt >= date_start and rs_date_dt <= date_end:
				radiosondes_nc.append(rs_nc)

	elif s_version == 'nya-rs':
		all_radiosondes_nc = sorted(glob.glob(path_data + "NYA-RS_*.nc"))

		# inquire the number of radiosonde files (date and time of launch is in filename):
		# And fill a list which will include the relevant radiosonde files.
		radiosondes_nc = []
		for rs_nc in all_radiosondes_nc:
			rs_date = rs_nc[-15:-3]		# date of radiosonde from filename
			yyyy = int(rs_date[:4])
			mm = int(rs_date[4:6])
			dd = int(rs_date[6:8])
			rs_date_dt = dt.datetime(yyyy,mm,dd)
			if rs_date_dt >= date_start and rs_date_dt <= date_end:
				radiosondes_nc.append(rs_nc)


	# number of sondes:
	n_sondes = len(radiosondes_nc)

	# count the number of days between start and end date as max. array size:
	n_days = (date_end - date_start).days

	# basic variables that should always be imported:
	if s_version == 'level_2':
		geoinfo_keys = ['lat', 'lon', 'launch_time', 'iwv']
		time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']
		if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']

	elif s_version == 'nya-rs':
		geoinfo_keys = ['lat', 'lon', 'launch_time']
		time_height_keys = ['pres', 'temp', 'rh', 'height']
		if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
	else:
		raise ValueError("s_version in import_radiosonde_daterange must be 'nya-rs' or 'level_2'.")
	all_keys = geoinfo_keys + time_height_keys

	# sonde_master_dict (output) will contain all desired variables on specific axes:
	# Time axis (one sonde = 1 timestamp) = axis 0; height axis = axis 1
	n_height = len(np.arange(0,15001,5))	# length of the interpolated height grid
	sonde_master_dict = dict()
	for gk in geoinfo_keys: sonde_master_dict[gk] = np.full((n_sondes,), np.nan)
	for thk in time_height_keys: sonde_master_dict[thk] = np.full((n_sondes, n_height), np.nan)

	if s_version == 'level_2':
		all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'Altitude', 'rho_v', 'q', 'time', 'IWV']
		if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']


		# cycle through all relevant sonde files:
		for rs_idx, rs_nc in enumerate(radiosondes_nc):
			
			if verbose >= 1:
				# rs_date = rs_nc[-19:-3]
				print("Working on Radiosonde, " + rs_nc)

			sonde_dict = import_single_PS122_mosaic_radiosonde_level2(rs_nc, keys=all_keys_import)
			
			# save to sonde_master_dict:
			for key in all_keys:
				if key in geoinfo_keys:
					sonde_master_dict[key][rs_idx] = sonde_dict[key]

				elif key in time_height_keys:
					sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]		# must use the interpolated versions!

				else:
					raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
									"import_single_PS122_mosaic_radiosonde_level2")

	if s_version == 'nya-rs':
		all_keys_import = ['lat', 'lon', 'press', 'temp', 'rh', 'alt', 'time']
		if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']


		# cycle through all relevant sonde files:
		for rs_idx, rs_nc in enumerate(radiosondes_nc):
			
			if verbose >= 1:
				# rs_date = rs_nc[-19:-3]
				print("Working on Radiosonde, " + rs_nc)

			sonde_dict = import_single_NYA_RS_radiosonde(rs_nc, keys=all_keys_import)
			
			# save to sonde_master_dict:
			for key in all_keys:
				if key in geoinfo_keys:
					sonde_master_dict[key][rs_idx] = sonde_dict[key]

				elif key in time_height_keys:
					sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]		# must use the interpolated versions!

				else:
					raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
									"import_single_NYA_RS_radiosonde")
				
	return sonde_master_dict


###################################################################################################
###################################################################################################


"""
	Small script to check out IASI files that Giovanni Chellini converted to
	netCDF.
"""

path_data = "/net/blanc/awalbroe/Data/METRS_SS21/IASI/"

file_list = sorted(glob.glob(path_data + "*.nc"))


"""
Useful variables:
	pressure_levels_temp
	pressure_levels_humidity
	record_start_time
	record_stop_time
	lat
	lon
	atmospheric_temperature
	atmospheric_water_vapor
	surface_temperature
	surface_pressure
	instrument_mode
	flag_cldnes
	flag_iasibad
	flag_itconv
	flag_landsea
	error_data_index
	temperature_error
	water_vapour_error
	surface_z				(useful in combination with surface_pressure)
	co_qflag
	co_bdiv

From Global Attributes:
	start_sensing_data_time
	end_sensing_data_time
	subsat_track_start_lat
	subsat_track_start_lon
	subsat_track_end_lat
	subsat_track_end_lon
"""

# Use along track as time line and record start or end time as actual time reference.
# Then, e.g. run through all files and concat along record_start_time (as time coordinates).
# Only use useful variables.
# Once that's completed, search for time steps where lat, lon matches that of Polarstern or
# Ny Alesund.

# open one DS:
SINGLE_IASI_DS = xr.open_dataset(file_list[0])

# open all at once:
# pdb.set_trace()
IASI_DS = xr.open_mfdataset(file_list, concat_dim='along_track', combine='nested', preprocess=cut_useless_variables, decode_times=False)


# distance = np.full(IASI_DS.lat.shape, np.nan)
# for idx in range(len(NyAl_DS.lon.values)):
	# distance[idx] = geopy.distance(NyAl_DS.lon.values[idx]

# IASI_DS['record_start_time'] = 
record_start_time = np.zeros(IASI_DS['record_start_time'].shape)
time_diff = (dt.datetime(2000,1,1) - dt.datetime(1970,1,1)).total_seconds()
n_time = IASI_DS['record_start_time'].shape[0]
iasi_record_start_time = IASI_DS.record_start_time.values
for i in range(n_time):
	record_start_time[i] = iasi_record_start_time[i] + time_diff

IASI_DS['record_start_time'] = xr.DataArray(record_start_time, dims=['along_track'])



# # time_idx[0] shows which along_track coordinate of IASI_DS is within radiosonde launch 0
# # time_idx[1] shows which along_track coordinate of IASI_DS is within radiosonde launch 1

# from geopy.distance import geodesic

# coords_nyal = (78.924444, 11.928611)

n_along = len(IASI_DS.along_track)
n_across = len(IASI_DS.across_track)
IASI_lon = IASI_DS.lon.values
IASI_lat = IASI_DS.lat.values




# It may be better to NOT import Ny Alesund radiosondes with xr.open_mfdataset because that will
# make xarray concatenate them along "time" ... but then the time axis contains all sondes so that
# e.g. time[0:5000] will be sonde no. 1, time[5000:10000] will be sonde no. 2, ...
# Better: When importing, convert radiosonde data to (launch_time x n_height), so that e.g. 300 sondes
# will result in an array (300 x n_height). When importing, interpolate all relevant variables to a
# regular height grid: 0 - 15 km with 5 m spacing: np.interp(np.arange(0,15001,5), sonde_dict['Altitude'], sonde_dict[key])
path_data = "/data/radiosondes/raw/01004_ny-alesund/"
NY_sonde_dict = import_radiosonde_daterange(path_data, date_start="2020-08-01", date_end="2020-09-30", s_version='nya-rs', with_wind=False)
pdb.set_trace()


# Find time overlap of Polarstern radiosonde launches and IASI overpasses:
# --> Create Polarstern radiosonde 'launch_time' similar to Ny Alesund launch time: use function create_launch_time

# Import Polarstern Radiosondes:
path_data = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"
date_start = "2020-08-20"		# in yyyy-mm-dd
date_end = "2020-08-31"			# in yyyy-mm-dd
s_version = 'level_2'

PS_sonde_dict = import_radiosonde_daterange(path_data, date_start, date_end, s_version, with_wind=False)


n_time_ps = len(PS_sonde_dict['launch_time'])


from geopy.distance import geodesic
distance_iasi_ps = np.full((n_along, n_across), 999999.99)
record_start_time = IASI_DS.record_start_time.values
which_PS_sonde = np.full((n_along, n_across), np.nan)
for i in range(n_along):
	if i%1000 == 0: print(i)
	# find Polarstern radiosonde launch that has temporally closest to IASI along track scan (pixel):
	launch_time_dif = np.abs(PS_sonde_dict['launch_time'] - record_start_time[i])
	idx_ps_time = np.argmin(launch_time_dif)

	if np.abs(PS_sonde_dict['launch_time'][idx_ps_time] - record_start_time[i]) <= 3600:
		# which_PS_sonde[i] = idx_ps_time
		for j in range(n_across):

			if np.abs(IASI_lat[i,j] <= 90):
				disdis = geodesic((PS_sonde_dict['lat'][idx_ps_time], PS_sonde_dict['lat'][idx_ps_time]), (IASI_lat[i,j], IASI_lon[i,j])).km
				if disdis < 50:
					which_PS_sonde[i,j] = idx_ps_time
					distance_iasi_ps[i,j] = disdis

IASI_DS['distance_iasi_ps'] = xr.DataArray(distance_iasi_ps, dims=['along_track', 'across_track'])

# remove nans from which_PS_sonde:
which_PS_sonde = which_PS_sonde.astype(np.int32)
which_PS_sonde_nonnan = np.unique(which_PS_sonde[which_PS_sonde >= 0]) # explained below



# what do we have now:
# - distance_iasi_ps (n_along, n_across): distance of IASI to Polarstern for each IASI pixel (value is only non-nan if that pixel is temporally within 3600 sec of a 
# 	Polarstern radiosonde launch AND if it is within 50 km of Polarstern)
# - which_PS_sonde_nonnan (varying dimension): this now indicates which Polarstern sondes have IASI pixels that are within 3600 sec of a sonde launch and where IASI
# 	pixel is within 50 km of Polarstern

# Later, we want to know if there is a IASI pixel for a given Polarstern launch which is close enough (i.e. < 50 km) and within
# 3600 sec of that sonde launch. So, we would like to have an array with the shape (n_sondes_ps,2) (or (n_time_ps,2)) that tells
# us the exact along track and across track coordinate of IASI that fulfills these conditions.

# So, we can now run through all Polarstern sondes again, and check, if the indicated along track coordinate has one or more IASI_DS['distance_iasi_ps'] < 50 km
iasi_pixels_for_nya = np.full((n_time_ps,2), 0)

ps_iasi_overlap_pixels = list()
for idx in range(n_time_ps):
	if idx in which_PS_sonde_nonnan: # then it's not a fill value and a IASI pixel with time offset < 3600 sec exists for the current Polarstern sonde launch:

		# find lines where which_PS_sonde is equal to idx:
		ps_iasi_overlap_pixels.append(np.argwhere(which_PS_sonde == idx))

"""
ps_iasi_overlap_pixels must always be considered together with which_PS_sonde_nonnan:
Example: which_PS_sonde_nonnan[0] is 16: then Polarstern sonde number 16 of your array is within temporal and spatial range of IASI overpasses
Then, ps_iasi_overlap_pixels[0] tells you which coordinates of IASI_DS overlaps with Polarstern: For example, 
ps_iasi_overlap_pixels[0] can be 

array([[372, 108],
       [372, 109],
       [372, 110],
       [372, 111],
       [373, 108],
       [373, 111]])

--> first column: along_track coordinate that fulfills our requirements ; second column: across_track coordinate that fulfills requirements
--> 6 pixels of IASI fulfill our requirements in this case. You may now select the Polarstern temperature profile via:

PS_sonde_dict['temp'][which_PS_sonde_nonnan[0], :];

and the IASI temperature profile(s) (dimensions: along_track, across_track, height):

n_height_iasi = len(IASI_DS.nlt)
n_detected_pixels = len(ps_iasi_overlap_pixels[0])
IASI_T = np.zeros((n_detected_pixels, n_height_iasi))		# here, the T profiles for the current Polarstern sonde are saved to
for idx, ps_ol in enumerate(ps_iasi_overlap_pixels[0]):
	IASI_T[idx,:] = IASI_DS.atmospheric_temperature[ps_ol[0], ps_ol[1],:]

IASI_I might include many nans because not all pixels of IASI actually have a temperature profile!
You might now either average over the number of detected pixels (ignore nans):
IASI_T_avg = np.nanmean(IASI_T, axis=0)		# might produce warnings (RuntimeWarning)
"""






# time_idx for Ny Alesund radiosondes: Do this computation AFTER computing the IASI_DS['distance_iasi_nyal'].
# then, in the for loop running through all Ny Al soundings, check if any IASI_DS


pdb.set_trace()