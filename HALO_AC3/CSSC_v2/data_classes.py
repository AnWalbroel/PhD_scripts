import numpy as np
import xarray as xr
import datetime as dt

import matplotlib.pyplot as plt
import gc

import glob
import os

import pdb


class dropsondes:

	"""
		HALO dropsondes launched during the field campaign(s) HALO-(AC)3. Several versions are 
		supported (see dataset_type and version). All dropsondes will be merged into a
		(launch_time, height) grid. Variable names will be unified in the class attributes
		(also in self.DS).
		

		For initialisation, we need:
		path_data : str
			String indicating the path of the dropsonde data. Subfolders may exist, depending on the
			dropsonde data version.
		dataset_type : str
			Indicates the type of dropsonde data. Options: "raw", "unified"
		version : str
			Indicates the version of the dropsonde data type.

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
		height_grid : 1D array of floats
			1D array of floats indicating the new height grid (especially raw) dropsonde data
			is interpolated to. Units: m
	"""

	def __init__(self, path_data, dataset_type, version="", **kwargs):

		# init attributes:
		self.temp = np.array([])			# air temperature in K
		self.pres = np.array([])			# air pressure in Pa
		self.rh = np.array([])				# relative humidity in [0, 1]
		self.height = np.array([])			# height in m
		self.launch_time = np.array([])		# launch time in sec since 1970-01-01 00:00:00 (for HALO-AC3)
		self.time = np.array([])			# time since launch_time in seconds
		self.u = np.array([])				# zonal wind component in m s-1
		self.v = np.array([])				# meridional wind component in m s-1
		self.wspeed = np.array([])			# wind speed in m s-1
		self.wdir = np.array([])			# wind direction in deg
		self.lat = np.array([])				# latitude in deg N
		self.lon = np.array([])				# longitude in deg E
		self.DS = None						# xarray dataset
		self.height_grid = np.array([])		# height grid in m
		self.n_hgt = 0						# number of height levels

		if ('height_grid' in kwargs.keys()): 
			self.height_grid = kwargs['height_grid']
			self.n_hgt = len(self.height_grid)
		else:
			# create default height grid:
			self.height_grid = np.arange(0.0, 16000.001, 10.0)
			self.n_hgt = len(self.height_grid)


		# Importing dropsonde files, depending on the type and version:
		if dataset_type == 'raw':

			# dictionary to change variable names: (also filters for relevant variables)
			translator_dict = {	'launch_time': 'launch_time',
								'time': 'time',
								'pres': 'pres',
								'tdry': 'temp',
								'rh': 'rh',
								'u_wind': 'u',
								'v_wind': 'v',
								'wspd': 'wspeed',
								'wdir': 'wdir',
								'lat': 'lat',
								'lon': 'lon',
								'alt': 'height',
								}

			# dictionary for final units:
			unit_dict = {	'launch_time': "seconds since 1970-01-01 00:00:00",
							'time': "seconds since 1970-01-01 00:00:00",
							'pres': "Pa",
							'temp': "K",
							'rh': "[0,1]",
							'u': "m s-1",
							'v': "m s-1",
							'wspeed': "m s-1",
							'wdir': "deg",
							'lat': "deg N",
							'lon': "deg E",}

			# search for daily subfolders and in them for *QC.nc:
			path_contents = os.listdir(path_data)
			subfolders = []
			for subfolder in path_contents:

				joined_contents = os.path.join(path_data, subfolder)
				if os.path.isdir(joined_contents):
					subfolders.append(joined_contents + "/")

			subfolders = sorted(subfolders)

			# find ALL dropsonde data files:
			# check if subfolders contain "Level_1", which should exist for *QC.nc:
			files_nc = []					# will contain all dropsonde files
			for subfolder in subfolders:	# this loop basically loops over the daily dropsondes:

				subfolder_contents = os.listdir(subfolder)
				if "Level_1" in subfolder_contents:
					files_nc = files_nc + sorted(glob.glob(subfolder + "Level_1/D*QC.nc"))

				else:
					raise ValueError(f"Could not find Level_1 dropsonde data in {subfolder} :(")

			# check if nc files were detected:
			if len(files_nc) == 0: raise RuntimeError("Where's the dropsonde data?? I can't find it.\n")


			# import data: importing with mfdataset costs a lot of memory and is therefore discarded here:
			DS_dict = dict()		# keys will indicate the dropsonde number of that day
			for k, file in enumerate(files_nc): DS_dict[str(k)] = xr.open_dataset(file)

			# interpolate dropsonde data to new height grid for all sondes; initialise array
			self.n_sondes = len(DS_dict.keys())
			vars_ip = dict()
			for var in translator_dict.keys(): vars_ip[var] = np.full((self.n_sondes, self.n_hgt), np.nan)

			for k, key in enumerate(DS_dict.keys()):

				# need to neglect nans:
				idx_nonnan = np.where(~np.isnan(DS_dict[key].alt.values))[0]

				# interpolate to new grid:
				for var in translator_dict.keys():
					if var not in ['launch_time', 'time']:
						try:
							vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key].alt.values[idx_nonnan],
															DS_dict[key][var].values[idx_nonnan], left=np.nan, right=np.nan)
						except ValueError:
							continue	# then, array for interpolation seems empty --> just leave nans is it

					elif var == 'time':
						# catch errors (empty array):
						try:
							vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key].alt.values[idx_nonnan], 
															DS_dict[key][var].values[idx_nonnan].astype("float64")*(1e-09),
															left=np.nan, right=np.nan)
						except ValueError:
							continue	# then, array for interpolation seems empty --> just leave nans is it

				"""
				# Uncomment if you would like to plot raw and interpolated dropsonde data (i.e., to check for correct procedures):
				if k%15 == 0:		# test some samples

					f1, a1 = plt.subplots(1,3)
					a1 = a1.flatten()
					a1[0].plot(vars_ip['tdry'][k,:], self.height_grid, color=(0,0,0), label='new')
					a1[0].plot(DS_dict[key].tdry.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')
					a1[1].plot(vars_ip['pres'][k,:], self.height_grid, color=(0,0,0), label='new')
					a1[1].plot(DS_dict[key].pres.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')
					a1[2].plot(vars_ip['rh'][k,:], self.height_grid, color=(0,0,0), label='new')
					a1[2].plot(DS_dict[key].rh.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')

					for ax in a1:
						ax.legend()
						ax.set_ylabel("Height (m)")
					a1[0].set_xlabel("tdry (degC)")
					a1[1].set_xlabel("pres (hPa)")
					a1[2].set_xlabel("rh (\%)")
					a1[1].set_title(f"{DS_dict[key].launch_time.values.astype('datetime64[D]')}")

					f1.savefig(f"/net/blanc/awalbroe/Plots/HALO_AC3/CSSC/dropsonde_ip_vs_original_{str(DS_dict[key].launch_time.values.astype('datetime64[D]')).replace('-','')}_{k}.png", 
								dpi=300, bbox_inches='tight')
					plt.close()
					gc.collect()
				"""


			# convert units of vars_ip to SI units:
			vars_ip['tdry'] = vars_ip['tdry'] + 273.15
			vars_ip['pres'] = vars_ip['pres']*100.0
			vars_ip['rh'] = vars_ip['rh']*0.01

			# create launch_time array:
			launch_time = np.zeros((self.n_sondes,))
			launch_time_npdt = np.full((self.n_sondes,), np.datetime64("1970-01-01T00:00:00.000000000"))
			for kk, key in enumerate(DS_dict.keys()):
				launch_time[kk] = DS_dict[key].launch_time.values.astype(np.float64)*(1e-09)
				launch_time_npdt[kk] = DS_dict[key].launch_time.values

			# compute time difference between launch times and true dropsonde measured times
			vars_ip['time_delta'] = np.full((self.n_sondes, self.n_hgt), np.nan)
			for k in range(self.n_sondes):
				vars_ip['time_delta'][k,:] = vars_ip['time'][k,:] - launch_time[k]


			# set class attributes:
			self.temp = vars_ip['tdry']
			self.pres = vars_ip['pres']
			self.rh = vars_ip['rh']
			self.height = self.height_grid
			self.launch_time = launch_time
			self.launch_time_npdt = launch_time_npdt
			self.time = vars_ip['time_delta']
			self.u = vars_ip['u_wind']
			self.v =  vars_ip['v_wind']
			self.wspeed = vars_ip['wspd']
			self.wdir = vars_ip['wdir']
			self.lat = vars_ip['lat']
			self.lon =  vars_ip['lon']


			# build new dataset with (launch_time, height) grid:
			if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:

				DS = xr.Dataset(coords={'launch_time': (['launch_time'], self.launch_time_npdt),
										'height': (['height'], self.height_grid, {'units': "m"})})


				for key in unit_dict.keys():
					if key not in ['launch_time', 'time']:
						DS[key] = xr.DataArray(self.__dict__[key], dims=['launch_time', 'height'], 
																attrs={'units': unit_dict[key]})
					elif key == 'time':

						DS[key] = xr.DataArray(self.time, dims=['launch_time', 'height'],
																attrs={'units': "seconds since launch_time"})

				DS.attrs['title'] = "HALO-(AC)3 HALO dropsondes Level_1 interpolated to (launch_time, height) grid"

				self.DS = DS


	def update_meteo_attrs(self):

		"""
		Update meteorological profiles of dropsondes
		"""

		pdb.set_trace()


class BAHAMAS:
	"""
		BAHAMAS data from HALO for time axis (and eventually other stuff later).

		For initialisation we need:
		path : str
			Path where HALO BAHAMAS data is located.
		which_date : str
			Marks the flight day that shall be imported. To be specified in yyyymmdd (e.g. 20200213)
			or "all" to import all nc files! "all" is not available for 'nc_raw'.
		version : str
			Version of the BAHAMAS data. Options available: 'nc_raw', 'halo_ac3_raw', 'unified'

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, which_date, version='halo_ac3_raw', **kwargs):
		
		if version == 'nc_raw': 
			# Identify correct time:
			files = [file for file in sorted(glob.glob(path + "*.nc")) if which_date in file]

			if len(files) == 1:	# then the file is unambiguous
				files = files[0]

			elif len(files) == 0:
				raise RuntimeError(f"No BAHAMAS files found for {which_date} in {path}.")

			else:
				print(f"Multiple potential BAHAMAS files found for {which_date} in {path}. Choose wisely... " +
						"I'll choose the first file")
				files = files[0]

			# import data:
			DS = xr.open_dataset(files)

			# set attributes:
			if version == 'nc_raw':
				self.time_npdt = DS.TIME.values			# np.datetime64 array
				self.time = DS.TIME.values.astype("datetime64[s]").astype("float64") # in seconds since 1970-01-01 00:00:00 UTC

				class_attrs = ['time']		# list of relevant class attributes (needed for self.DS)


		elif version == 'unified':
			# Identify correct date range of files:
			if which_date == 'all':
				files = sorted(glob.glob(path + "*.nc"))

				# import data:
				DS = xr.open_mfdataset(files, combine='nested', concat_dim='time')

			else:
				files = [file for file in sorted(glob.glob(path + "*.nc")) if which_date in file]

				if len(files) == 1:	# then the file is unambiguous
					files = files[0]

				elif len(files) == 0:
					raise RuntimeError(f"No BAHAMAS files found for {which_date} in {path}.")

				else:
					print(f"Multiple potential BAHAMAS files found for {which_date} in {path}. Choose wisely... " +
							"I'll choose the first file")
					files = files[0]

				# import data:
				DS = xr.open_dataset(files)

			# set attributes:
			self.time_npdt = DS.time.values		# np.datetime64 array
			self.time = DS.time.values.astype("datetime64[s]").astype("float64") # in seconds since 1970-01-01 00:00:00 UTC
			self.alt = DS.alt.values			# flight altitude in m
			self.lat = DS.lat.values			# latitude in deg N
			self.lon = DS.lon.values			# longitude in deg E
			self.pres = DS.P.values*100.0		# static air pressure in Pa
			self.temp = DS.T.values				# air temperature in K
			self.rh = DS.RH.values*0.01			# relative humidity in [0,1]

			class_attrs = ['time', 'alt', 'lat', 'lon', 'pres', 'temp', 'rh']		# list of relevant class attributes (needed for self.DS)


		elif version == 'halo_ac3_raw':
			# Identify correct time: 
			if which_date == 'all':
				# search for daily subfolders:
				path_contents = os.listdir(path)
				subfolders = []
				for subfolder in path_contents:

					joined_contents = os.path.join(path, subfolder)
					if os.path.isdir(joined_contents):
						subfolders.append(joined_contents + "/")

				subfolders = sorted(subfolders)

				# find ALL dropsonde data files:
				# check if subfolders contain "Level_1", which should exist for *QC.nc:
				files = []					# will contain all dropsonde files
				for subfolder in subfolders:	# this loop basically loops over the daily dropsondes:

					subfolder_contents = os.listdir(subfolder)
					files = files + sorted(glob.glob(subfolder + "*BAHAMAS*_v1.nc"))

				# import data:
				DS = xr.open_mfdataset(files, combine='nested', concat_dim='tid')

			else:
				path += f"/HALO-AC3_HALO_BAHAMAS_{which_date}_{RF_dict[which_date]}/"
				files = [file for file in sorted(glob.glob(path + "*BAHAMAS*.nc")) if which_date in file]

				if len(files) == 1:	# then the file is unambiguous
					files = files[0]

				elif len(files) == 0:
					raise RuntimeError(f"No BAHAMAS files found for {which_date} in {path}.")

				else:
					print(f"Multiple potential BAHAMAS files found for {which_date} in {path}. Choose wisely... " +
							"I'll choose the first file")
					files = files[0]

				# import data:
				DS = xr.open_dataset(files)

			# set attributes:
			self.time_npdt = DS.TIME.values			# np.datetime64 array
			self.time = DS.TIME.values.astype("datetime64[s]").astype("float64") # in seconds since 2017-01-01 00:00:00 UTC
			self.alt = DS.IRS_ALT.values			# flight altitude in m
			self.lat = DS.IRS_LAT.values			# latitude in deg N
			self.lon = DS.IRS_LON.values			# longitude in deg E
			self.pres = DS.PS.values*100.0			# static air pressure in Pa
			self.temp = DS.TS.values				# air temperature in K
			self.rh = DS.RELHUM.values*0.01			# relative humidity in [0,1]

			class_attrs = ['time', 'alt', 'lat', 'lon', 'pres', 'temp', 'rh']		# list of relevant class attributes (needed for self.DS)


		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
				# creating a dataset manually allows to have unified variable names
				DS_self = xr.Dataset(coords={'time': (['time'], self.time_npdt)})

				# put variables into the dataset
				for key in class_attrs:
					if key not in ['time']:
						DS_self[key] = xr.DataArray(self.__dict__[key], dims=['time'])

				DS_self.attrs['title'] = "HALO-(AC)3 HALO BAHAMAS data"
				DS_self.attrs['units_comment'] = "SI units, see data_classes.py code"

				self.DS = DS_self