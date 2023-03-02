import numpy as np
import datetime as dt
from import_data import *
from met_tools import *
from data_tools import *
import copy


class radiometers:
	"""
		Microwave radiometers (MWRs) such as HATPRO or LHUMPRO (MiRAC-P).

		For initialisation, we need:
		path_r : str or list of str
			Base path of MWR data. This directory contains subfolders representing the year, which,
			in turn, contain months, which contain day subfolders. Example path_r =
			"/data/obs/campaigns/mosaic/hatpro/l1/". If instrument == 'synthetic' path_r can also be
			a list of strings where each entry contains path and filename.
		instrument : str
			Specifies the instrument (radiometer instance). Options: 'hatpro', 'mirac-p', 'synthetic',
			'syn_mwr_pro'. In case 'synthetic' or 'syn_mwr_pro' is chosen, version, date_start, 
			date_end, and truncate_flagged become irrelevant.
		
		**kwargs:
		version : str
			Specifies the data version. Valid option depends on the instrument.
		date_start : str
			Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
		date_end : str
			Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
		truncate_flagged : bool
			If True, the data arrays (time, flag, TB) will be truncated to values where flag=0. If
			False, no truncation will be performed.
		include_pres_sfc : bool
			Used for instrument = 'synthetic' for a Neural Network retrieval. If true, surface pressure
			will also be imported and saved as class attribute.
		add_TB_noise : bool
			If True, random noise can be added to the brightness temperatures using the built-in function
			add_TB_noise. Usually only used if instrument == 'synthetic'.
		noise_dict : dict
			Dictionary that has the frequencies (with a resolution of 0.01 (.2f)) as keys and the noise 
			strength (in K) as value. Only used in add_TB_noise == True. 
			Example: noise_dict = {'190.71': 3.0}
		subset : array of str
			String array indicating a set of years that will be used for importing. For example, data is
			available in 2001-2017, but you only need the subset 2001-2005. Then, subset = np.asarray(
			["2001", "2002", "2003", "2004", "2005"]).
		subset_months : list of int
			List of integer indicating the months that will be used. All other months will be discarded.
			This option might be useful, if only summer months should be considered. Example: [6,7,8]
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path_r, instrument, **kwargs):

		if instrument == 'hatpro':
			mwr_dict = import_hatpro_level1b_daterange(path_r, kwargs['date_start'], kwargs['date_end'], verbose=1)

			# Unify variable names by defining class attributes:
			self.freq = mwr_dict['freq_sb']	# in GHz
			self.time = mwr_dict['time']	# in sec since 1970-01-01 00:00:00 UTC
			self.TB = mwr_dict['tb']		# in K, time x freq
			self.flag = mwr_dict['flag']

			if kwargs['truncate_flagged']:
				self.time = self.time[self.flag == 0]
				self.TB = self.TB[self.flag == 0, :]
				self.flag = self.flag[self.flag == 0]

		if instrument == 'mirac-p' and kwargs['version'] in ['i00', 'i01', 'i02', 'i03', 'v00', 'v01']:
			raise ValueError("Daterange importer has not yet been implemented for MiRAC-P " +
								"TBs for version '%s'."%kwargs['version'])

			if kwargs['truncate_flagged']:
				self.flag[self.flag == 16] = 0				# because of sanity_receiver_band1
				self.time = self.time[self.flag == 0]
				self.TB = self.TB[self.flag == 0, :]
				self.flag = self.flag[self.flag == 0]

		elif instrument == 'mirac-p' and kwargs['version'] == 'RPG':
			mwr_dict = import_mirac_BRT_RPG_daterange(path_r, kwargs['date_start'], kwargs['date_end'], verbose=1)

			# Unify variable names by defining class attributes:
			self.freq = mwr_dict['Freq']	# in GHz
			self.time = mwr_dict['time']	# in sec since 1970-01-01 00:00:00 UTC
			self.TB = mwr_dict['TBs']		# in K, time x freq
			self.flag = mwr_dict['RF']

			if kwargs['truncate_flagged']:
				self.time = self.time[self.flag == 0]
				self.TB = self.TB[self.flag == 0, :]
				self.flag = self.flag[self.flag == 0]

		elif instrument == 'synthetic':
			MWR_DS = import_synthetic_TBs(path_r)

			# Unify variable names by defining class attributes:
			self.freq = MWR_DS.frequency.values					# in GHz
			self.time = MWR_DS.time.values						# in sec since 1970-01-01 00:00:00 UTC
			self.TB = MWR_DS.brightness_temperatures.values		# in K, time x freq
			self.flag = np.zeros((len(self.time),))

			# If desired, random noise can be added:
			if 'add_TB_noise' in kwargs.keys():
				if kwargs['add_TB_noise'] and ('noise_dict' in kwargs.keys()):
					self.add_TB_noise(kwargs['noise_dict'])

				elif kwargs['add_TB_noise'] and ('noise_dict' not in kwargs.keys()):
					raise KeyError("Class radiometers requires 'noise_dict' if 'add_TB_noise' is True.")

			if kwargs['include_pres_sfc']:
				self.pres = MWR_DS.atmosphere_pressure_sfc.values	# pressure in Pa

		elif instrument == 'syn_mwr_pro':	# synthetic TBs as uploaded on ZENODO

			if type(path_r) == str:
				path_r = sorted(glob.glob(path_r + "*.nc"))

			MWR_DS = xr.open_mfdataset(path_r, concat_dim='time', combine='nested')

			# Cut unwanted dimensions in variables:
			MWR_DS['ele'] = MWR_DS.ele[0]
			MWR_DS['freq_sb'] = MWR_DS.freq_sb[0,:]		# eliminate time dependency

			# Eventually, only subset is needed: Filter time stamps:
			# Find indices for each year in the subset:
			if "subset" in kwargs.keys():
				idx_y = list()
				for subset in kwargs['subset']:
					idx_y.append(np.where(MWR_DS['time.year'].values == int(subset))[0])

				# convert to an array or list containing ALL indices of each year
				idx_yall = np.array([])
				for subset in idx_y:
					idx_yall = np.concatenate((idx_yall, subset), axis=0)
				idx_yall = idx_yall.astype(np.int32)

				# limit dataset to selected subset:
				MWR_DS = MWR_DS.isel(time=idx_yall)


			# Eventually, limit to certain months:
			if "subset_months" in kwargs.keys() and len(kwargs['subset_months']) > 0:
				MWR_DS = MWR_DS.isel(time=(MWR_DS.time.dt.month.isin(kwargs['subset_months'])))


			# Unify variable names by defining class attributes:
			self.freq = MWR_DS.freq_sb.values					# in GHz
			self.time = numpydatetime64_to_epochtime(MWR_DS.time.values)	# in sec since 1970-01-01 00:00:00 UTC
			self.TB = MWR_DS.tb.values[:,0,:]					# in K, time x freq (elevation axis removed)
			self.flag = np.zeros((len(self.time),))

			# If desired, random noise can be added:
			if 'add_TB_noise' in kwargs.keys():
				if kwargs['add_TB_noise'] and ('noise_dict' in kwargs.keys()):
					self.add_TB_noise(kwargs['noise_dict'])

				elif kwargs['add_TB_noise'] and ('noise_dict' not in kwargs.keys()):
					raise KeyError("Class radiometers requires 'noise_dict' if 'add_TB_noise' is True.")

			if kwargs['include_pres_sfc']:
				raise KeyError("'include_pres_sfc' not implemented in the uploaded version on ZENODO.")

			# also possible to return the xarray dataset
			if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
				self.DS = MWR_DS

	def add_TB_noise(self, noise_dict):

		"""
		Adds random (un)correlated noise to the brightness temperatures, which must be
		in time x freq shape.

		Parameters:
		-----------
		noise_dict : dict
			Dictionary that has the frequencies (with .2f floating point precision) as keys
			and the noise strength (in K) as value. Example: '190.71': 3.0
		"""

		n_time = self.TB.shape[0]

		# Loop through frequencies. Find which frequency is currently addressed and
		# create respective noise:
		for freq_sel in noise_dict.keys():
			frq_idx = np.where(np.isclose(self.freq, float(freq_sel), atol=0.01))[0]
			if len(frq_idx) > 0:
				frq_idx = frq_idx[0]
				self.TB[:,frq_idx] = self.TB[:,frq_idx] + np.random.normal(0.0, noise_dict[freq_sel], size=n_time)

	def get_calibration_times(self, instrument, to_epochtime):

		"""
		Saves the calibration times of a specified instrument as array of 
		datetime objects into the instanced radiometer object. Calibration
		times will be converted to seconds since 1970-01-01 00:00:00 UTC if
		to_epochtime is True.

		Parameters:
		-----------
		instrument : str
			Specifies the instrument (radiometer instance). Options: 'hatpro', 'mirac-p'.
		to_epochtime : bool
			If True, calibration times will be converted to epochtime (seconds since
			1970-01-01 00:00:00 UTC) and 
		"""

		if instrument == 'hatpro':
			# calibration times of HATPRO: manually entered from MWR logbook
			calibration_times = np.asarray([dt.datetime(2019,10,19,6,0), dt.datetime(2019,12,14,18,30), 
											dt.datetime(2020,3,1,11,0), dt.datetime(2020,5,2,12,0),
											dt.datetime(2020,7,6,9,33), dt.datetime(2020,8,12,9,17)])

		elif instrument == 'mirac-p':
			# calibration times of MiRAC-P: manually entered from logbook
			calibration_times = np.asarray([dt.datetime(2019,10,19,6,30), dt.datetime(2019,10,22,5,40),
											dt.datetime(2020,7,6,12,19), dt.datetime(2020,8,12,9,37)])

		if to_epochtime: calibration_times = datetime_to_epochtime(calibration_times)
		self.calibration_times = calibration_times


class radiosondes:
	"""
		Radiosondes such as those used during the MOSAiC campaign.

		For initialisation, we need:
		path_r : str or list of str
			Path of radiosonde data if single == False. Path of radiosonde data + filename itself if
			single == True. If s_version == 'mwr_pro' path_r can be a list of strings where each 
			entry contains path and filename.
		s_version : str
			Specifies the radiosonde version that is to be imported. Possible options: 'mossonde',
			'psYYMMDDwHH', 'level_2', 'mwr_pro'. Default: 'level_2' (published by Marion Maturilli)
		single : bool
			If True only one radiosonde will be loaded and not, as in the case single == False, an entire
			date range of radiosondes.
		with_wind : bool
			This describes if wind measurements are included (True) or not (False). Does not work with
			s_version='psYYMMDDwHH'. Default: False

		**kwargs:
		date_start : str
			Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
		date_end : str
			Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
		with_lwp : bool
			If True, liquid water path is also loaded from data. Make sure that this quantity exists.
		remove_failed : bool
			If True, failed sondes with unrealistic IWV values will be removed (currently only implmented
			for s_version == 'level_2').
	"""

	def __init__(self, path_r, s_version='level_2', single=False, with_wind=False, **kwargs):

		if single:
			if s_version == 'level_2':
				sonde_dict = import_single_PS122_mosaic_radiosonde_level2(path_r)
			elif s_version == 'mossonde':
				sonde_dict = import_single_mossonde_curM1(path_r)
			elif s_version == 'psYYMMDDwHH':
				sonde_dict = import_single_psYYMMDD_wHH_sonde(path_r)
			elif s_version == 'mwr_pro':
				raise ValueError("Argument 'single' must be False when s_version is 'mwr_pro', trying to " +
									"call the class 'radiosondes'.")

		else:
			if s_version == 'mwr_pro':

				with_lwp = False
				if "with_lwp" in kwargs.keys():
					with_lwp = kwargs['with_lwp']

				sonde_dict = import_mwr_pro_radiosondes(path_r, with_lwp=with_lwp)

			else:
				if (not kwargs['date_start']) and (not kwargs['date_end']):
					raise ValueError("If multiple radiosondes shall be imported (single=False) a date range " +	
									"(date_start and date_end) must be specified.")
				if not type(path_r) == str:
					raise TypeError("Argument 'path_r' must be a string when s_version != 'mwr_pro'. " +
									"Path of radiosonde data if single == False. Path of radiosonde " + 
									"data + filename itself if single == True. If s_version == 'mwr_pro' " +
									"path_r can be a list of strings where each entry contains path and filename.")

				sonde_dict = import_radiosonde_daterange(path_r, kwargs['date_start'], kwargs['date_end'], s_version=s_version,
															remove_failed=kwargs['remove_failed'], with_wind=with_wind, verbose=1)

		# Unification of variable names already done in the importing routine:
		if single:

			# also need to convert to a 2D array (time x height) to handle it just like single == False:
			n_hgt = len(sonde_dict['height_ip'])
			self.pres = np.reshape(sonde_dict['pres_ip'], (1, n_hgt))		# in Pa
			self.temp = np.reshape(sonde_dict['temp_ip'], (1, n_hgt))		# in K
			self.rh = np.reshape(sonde_dict['rh_ip'], (1, n_hgt))			# between 0 and 1
			self.height = np.reshape(sonde_dict['height_ip'], (1, n_hgt))	# in m
			self.rho_v = np.reshape(sonde_dict['rho_v_ip'], (1, n_hgt))		# in kg m^-3
			self.q = np.reshape(sonde_dict['q_ip'], (1, n_hgt))				# in kg kg^-1

			if with_wind:
				self.wspeed = np.reshape(sonde_dict['wspeed_ip'], (1, n_hgt))	# in m s^-1
				self.wdir = np.reshape(sonde_dict['wdir_ip'], (1, n_hgt))		# in deg

			self.lat = np.reshape(sonde_dict['lat'], (1,))		# in deg N
			self.lon = np.reshape(sonde_dict['lon'], (1,))		# in deg E
			self.launch_time = np.reshape(sonde_dict['launch_time'], (1,)) # in sec since 1970-01-01 00:00:00 UTC
			self.iwv = np.reshape(sonde_dict['iwv'], (1,))		# in kg m^-2

		else:
			self.pres = sonde_dict['pres']		# in Pa
			self.temp = sonde_dict['temp']		# in K
			self.rh = sonde_dict['rh']			# between 0 and 1
			self.height = sonde_dict['height']	# in m
			self.rho_v = sonde_dict['rho_v']	# in kg m^-3
			self.q = sonde_dict['q']			# in kg kg^-1

			if with_wind:
				self.wspeed = sonde_dict['wspeed']	# in m s^-1
				self.wdir = sonde_dict['wdir']		# in deg

			self.lat = sonde_dict['lat']		# in deg N
			self.lon = sonde_dict['lon']		# in deg E
			self.launch_time = sonde_dict['launch_time'] # in sec since 1970-01-01 00:00:00 UTC
			self.iwv = sonde_dict['iwv']		# in kg m^-2

			if with_lwp: self.lwp = sonde_dict['lwp']

	def fill_gaps_easy(self, nan_threshold=0.33):

		"""
		Simple function to quickly fill small gaps in the measurements.
		Runs through all radiosonde launches and checks which altitudes
		show gaps. If the number of gaps is less than 33% of the height
		level number, the holes will be filled.
		Wind is not respected here because I am currently only interested
		in surface winds and therefore don't care about wind measurement
		gaps in higher altitudes.

		Parameters:
		-----------
		nan_threshold : float, optional
			Threshold describing the fraction of nan values of the total height level
			number that is still permitted for computation.
		"""

		# Dictionary is handy here because we can address the variable
		# with the hole easier. xarray or pandas would also work but is usually
		# slower.
		sonde_dict = {'pres':self.pres,
						'temp':self.temp,
						'rh': self.rh,
						'height': self.height,
						'rho_v': self.rho_v,
						'q': self.q}

		n_height = len(sonde_dict['height'][0,:])
		max_holes = int(nan_threshold*n_height)	# max permitted number of missing values in a column
		for k, lt in enumerate(self.launch_time):
			# count nans in all default meteorol. measurements:
			n_nans = {'pres': np.count_nonzero(np.isnan(self.pres[k,:])),
						'temp': np.count_nonzero(np.isnan(self.temp[k,:])),
						'rh': np.count_nonzero(np.isnan(self.rh[k,:])),
						'height': np.count_nonzero(np.isnan(self.height[k,:])),
						'rho_v': np.count_nonzero(np.isnan(self.rho_v[k,:])),
						'q': np.count_nonzero(np.isnan(self.q[k,:]))}

			all_nans = np.array([n_nans['pres'], n_nans['temp'], n_nans['rh'], n_nans['height'],
						n_nans['rho_v'], n_nans['q']])

			if np.any(all_nans >= max_holes):
				print("Too many gaps in this launch: %s"%(dt.datetime.
						utcfromtimestamp(lt).strftime("%Y-%m-%d %H:%M:%S")))
				continue

			elif np.any(all_nans > 0):
				# which variables have got holes:
				ill_keys = [key for key in n_nans.keys() if n_nans[key] > 0]

				# Repair illness:
				for ill_key in ill_keys:
					nan_mask = np.isnan(sonde_dict[ill_key][k,:])
					nan_mask_diff = np.diff(nan_mask)		# yields position of holes
					where_diff = np.where(nan_mask_diff)[0]
					n_holes = int(len(where_diff) / 2)

					if len(where_diff) % 2 > 0:	# then the hole is at the bottom or top of the column
						continue
					else:
						# indices of bottom and top boundary of each hole:
						hole_boundaries = np.asarray([[where_diff[2*jj], where_diff[2*jj+1]+1] for jj in range(n_holes)])
						
						# use the values of the hole boundaries as interpolation targets:
						temp_var = copy.deepcopy(sonde_dict[ill_key][k,:])

						# cycle through holes:
						for hole_b, hole_t in zip(hole_boundaries[:,0], hole_boundaries[:,1]):
							rpl_idx = np.arange(hole_b, hole_t + 1)	# +1 because of python indexing
							bd_idx = np.array([rpl_idx[0], rpl_idx[-1]])
							bd_val = np.array([temp_var[hole_b], temp_var[hole_t]])

							bridge = np.interp(rpl_idx, bd_idx, bd_val)

							# fill the whole hole:
							sonde_dict[ill_key][k,rpl_idx] = bridge


		# save changes to class attributes:
		self.pres = sonde_dict['pres']
		self.temp = sonde_dict['temp']
		self.rh = sonde_dict['rh']
		self.height = sonde_dict['height']
		self.rho_v = sonde_dict['rho_v']
		self.q = sonde_dict['q']


class era_i:
	"""
		ERA-I reanalysis as published on ZENODO. Only for NN_retrieval_miracp.py. Time will be
		converted to epochtime (seconds since 1970-01-01 00:00:00 UTC).

		For initialisation, we need:
		file : str or list of str
			List with one entry containing path + filename of ERA-I data.

		**kwargs:
		subset : array of str
			String array indicating a set of years that will be used for importing. For example, data is
			available in 2001-2017, but you only need the subset 2001-2005. Then, subset = np.asarray(
			["2001", "2002", "2003", "2004", "2005"]).
		subset_months : list of int
			List of integer indicating the months that will be used. All other months will be discarded.
			This option might be useful, if only summer months should be considered. Example: [6,7,8]
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, file, **kwargs):

		DS = xr.open_dataset(file[0])

		# Eventually, only subset is needed: Filter time stamps:
		# Find indices for each year in the subset:
		if "subset" in kwargs.keys():
			idx_y = list()
			for subset in kwargs['subset']:
				idx_y.append(np.where(DS['time.year'].values == int(subset))[0])

			# convert to an array or list containing ALL indices of each year
			idx_yall = np.array([])
			for subset in idx_y:
				idx_yall = np.concatenate((idx_yall, subset), axis=0)
			idx_yall = idx_yall.astype(np.int32)

			# limit dataset to selected subset:
			DS = DS.isel(time=idx_yall)

		# Eventually, limit to certain months:
		if "subset_months" in kwargs.keys() and len(kwargs['subset_months']) > 0:
			DS = DS.isel(time=(DS.time.dt.month.isin(kwargs['subset_months'])))

		# assign attributes:
		self.launch_time = numpydatetime64_to_epochtime(DS.time.values) # in sec since 1970-01-01 00:00:00 UTC
		self.iwv = DS.prw.values		# in kg m^-2

		# also possible to return the xarray dataset
		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = DS

