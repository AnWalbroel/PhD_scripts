import numpy as np
import datetime as dt
import xarray as xr
import pdb
import sys
import glob
from major_importer import *

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')    # so that functions and modules from MOSAiC can be used
from data_tools import *
from met_tools import convert_rh_to_spechum, convert_rh_to_abshum


class MWR:
	"""
		Unified microwave radiometer (during EUREC4A) onboard HALO (part of HAMP).

		For initialisation we need:
		path : str
			Path of unified HALO HAMP-MWR data from EUREC4A campaign.
		version : str
			Specifies the data version. Valid option depends on the instrument.
		which_date : str
			Marks the EUREC4A flight day that shall be imported. To be specified in 
			yyyymmdd (e.g. 20200213)!

		**kwargs:
		cut_low_altitude : bool
			If True, time steps with altitude below 6000 m are removed. If False,
			this won't be done. Default: True
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, version, which_date, **kwargs):

		# import data: Identify correct data (version, dates):
		filename = sorted(glob.glob(path + f"radiometer_{which_date}_{version}*.nc"))
		if len(filename) == 0:
			raise RuntimeError(f"Could not find and import {filename}.")
		else:
			filename = filename[0]
		if 'cut_low_altitude' in kwargs.keys():
			data_DS = import_HAMP_unified_product(filename, cut_low_altitude=kwargs['cut_low_altitude'])
		else:
			data_DS = import_HAMP_unified_product(filename)
		
		# Unify variable names by defining class attributes:
		self.freq = data_DS.frequency.values			# in GHz
		self.time = data_DS.time.values					# in sec since 1970-01-01 00:00:00 UTC
		self.TB = data_DS.tb.values						# in K, time x freq

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS


class radar:
	"""
		Unified cloud radar (during EUREC4A) onboard HALO (part of HAMP).

		For initialisation we need:
		path : str
			Path of unified HALO HAMP radar data from EUREC4A campaign.
		version : str
			Specifies the data version. Valid option depends on the instrument.
		which_date : str
			Marks the EUREC4A flight day that shall be imported. To be specified in 
			yyyymmdd (e.g. 20200213)!

		**kwargs:
		cut_low_altitude : bool
			If True, time steps with altitude below 6000 m are removed. If False,
			this won't be done. Default: True
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, version, which_date, **kwargs):

		# import data: Identify correct data (version, dates):
		filename = sorted(glob.glob(path + f"radar_{which_date}_{version}*.nc"))
		if len(filename) == 0:
			raise RuntimeError(f"Could not find and import {filename}.")
		else:
			filename = filename[0]
		if 'cut_low_altitude' in kwargs.keys():
			data_DS = import_HAMP_unified_product(filename, cut_low_altitude=kwargs['cut_low_altitude'])
		else:
			data_DS = import_HAMP_unified_product(filename)
		
		# Unify variable names by defining class attributes:
		self.time = data_DS.time.values				# in sec since 1970-01-01 00:00:00 UTC
		self.height = data_DS.height.values			# in m above ground
		self.dBZ = data_DS.dBZ.values				# equivalent reflectivity factor in dBZ
		self.Z = data_DS.Z.values					# radar reflectivity im mm^6 m^-3

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS


class dropsondes:
	"""
		Dropsondes (during EUREC4A) launched from HALO.

		For initialisation we need:
		path : str
			Path of the dropsonde data from the EUREC4A campaign.
		version : str
			Specifies the data version. Valid option depends on the instrument.
            For unified dropsonde data, 'v0.9' is a valid option. Geet George's
            JOANNE dropsonde dataset is called if version == "joanne_level_3".
		which_date : str
			Marks the EUREC4A flight day that shall be imported. To be specified in
			yyyymmdd (i.e., 20200213)!

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
		save_time_height_matrix : bool
			If True, the matrix of time x height will be used for the attributes instead
			of False: sonde_number x height where each sonde number merely has a launch_time.
			Only usable on unified dropsonde data.
	"""

	def __init__(self, path, version, which_date, **kwargs):

		if version == 'joanne_level_3':
			filename = sorted(glob.glob(path + "EUREC4A_JOANNE_Dropsonde*.nc"))

			if len(filename) == 0:
				raise RuntimeError(f"Could not find and import {filename}.")
			else:
				filename = filename[0]
			data_DS = xr.open_dataset(filename)

			# Get real altitude from alt boundaries (at least necessary in version Level_3_v0.9.2)
			data_DS['alt'] = data_DS.alt_bnds.mean(dim='nv')

			# Pick correct date and convert time:
			time_idx = np.where((data_DS.launch_time.values >= np.datetime64("%s-%s-%sT00"%(which_date[:4], which_date[4:6],
								which_date[6:]))) & (data_DS.launch_time.values < np.datetime64("%s-%s-%sT00"%(
								which_date[:4], which_date[4:6], which_date[6:])) + np.timedelta64(1, 'D')) & 
								(data_DS.platform.values == 'HALO'))[0]

			if len(time_idx) > 0:
				data_DS = data_DS.isel(sounding=time_idx)

			if 'time' in data_DS:
				# convert time and remove nasty after decimal point values:
				data_DS['time_npdt'] = data_DS.time				# to keep the numpy datetime 64
				data_DS['time'] = np.rint(numpydatetime64_to_epochtime(data_DS.time.values)).astype(float)

			if 'launch_time' in data_DS:
				# convert time and remove nasty after decimal point values:
				data_DS['launch_time_npdt'] = data_DS.launch_time				# to keep the numpy datetime 64
				data_DS = data_DS.drop_vars('launch_time')
				data_DS['launch_time'] = np.rint(numpydatetime64_to_epochtime(data_DS.launch_time_npdt.values)).astype(float)


			# save attributes and convert to SI units:
			self.pres = data_DS.p.values						# in Pa
			self.temp = data_DS.ta.values						# in K
			self.rh = data_DS.rh.values							# between 0 and 1
			self.height = data_DS.alt.values					# in m
			self.q = data_DS.q.values							# in kg kg^-1
			self.rho_v = convert_rh_to_abshum(self.temp, self.rh)			# in kg m^-3
			self.lat = data_DS.lat.values						# in deg N
			self.lon = data_DS.lon.values						# in deg E
			self.launch_time = data_DS.launch_time.values		# in sec since 1970-01-01 00:00:00 UTC


		else:   # unified dropsonde data
			# import data: Identify correct data (version, dates):
			filename = sorted(glob.glob(path + f"dropsondes_{which_date}_{version}*.nc"))
			if len(filename) == 0:
				raise RuntimeError(f"Could not find and import {filename}.")
			else:
				filename = filename[0]
			data_DS = xr.open_dataset(filename)

			if 'time' in data_DS:
				# convert time and remove nasty after decimal point values:
				data_DS['time_npdt'] = data_DS.time				# to keep the numpy datetime 64
				data_DS['time'] = np.rint(numpydatetime64_to_epochtime(data_DS.time.values)).astype(float)

				# make sure that time sampling is 1 second:
				assert np.all(np.diff(data_DS.time.values) == 1.0)

			if 'launch_time' in data_DS:
				# convert time and remove nasty after decimal point values:
				data_DS['launch_time_npdt'] = data_DS.launch_time				# to keep the numpy datetime 64
				data_DS = data_DS.drop_vars('launch_time')
				data_DS['launch_time'] = np.rint(numpydatetime64_to_epochtime(data_DS.launch_time_npdt.values)).astype(float)


			# save attributes and convert to SI units:
			if ('save_time_height_matrix' in kwargs.keys()) and kwargs['save_time_height_matrix']:
				self.pres = data_DS.p_mat.values*100				# in Pa
				self.temp = data_DS.ta_mat.values + 273.15			# in K
				self.rh = data_DS.rh_mat.values*0.01				# between 0 and 1
				self.height = data_DS.height.values					# in m
				self.q = convert_rh_to_spechum(self.temp, self.pres, self.rh)	# in kg kg^-1
				self.rho_v = convert_rh_to_abshum(self.temp, self.rh)			# in kg m^-3
				self.lat = data_DS.lat_mat.values					# in deg N
				self.lon = data_DS.lon_mat.values					# in deg E
				self.launch_time = data_DS.launch_time.values		# in sec since 1970-01-01 00:00:00 UTC
				self.time = data_DS.time.values						# in sec since 1970-01-01 00:00:00 UTC

			else:
				self.pres = data_DS.p.values*100					# in Pa
				self.temp = data_DS.ta.values + 273.15				# in K
				self.rh = data_DS.rh.values*0.01					# between 0 and 1
				self.height = data_DS.height.values					# in m
				self.q = convert_rh_to_spechum(self.temp, self.pres, self.rh)	# in kg kg^-1
				self.rho_v = convert_rh_to_abshum(self.temp, self.rh)			# in kg m^-3
				self.lat = data_DS.lat.values						# in deg N
				self.lon = data_DS.lon.values						# in deg E
				self.launch_time = data_DS.launch_time.values		# in sec since 1970-01-01 00:00:00 UTC

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS

	def fill_gaps_easy(self):

		"""
		Simple function to quickly fill small gaps in the measurements.
		Runs through all radiosonde launches and checks which altitudes
		show gaps. If the number of gaps is less than 33% of the height
		level number, the holes will be filled.
		Wind is not respected here because I am currently only interested
		in surface winds and therefore don't care about wind measurement
		gaps in higher altitudes.
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
		max_holes = int(0.33*n_height)	# max permitted number of missing values in a column
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
				print("Too many gaps to be filled in this launch: %s"%(dt.datetime.
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


class lidar:
	"""
		Lidar (during EUREC4A) onboard HALO.

		For initialisation we need:
		path : str
			Path of the lidar data. For HALO (WALES), the path depends on the date.
		version : str
			Version of the lidar data. Valid option depends on the instrument.

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, version, **kwargs):

		# import data:
		filename = glob.glob(path + f"EUREC4A_HALO_WALES_cloudtop_*a_{version}.nc")
		if len(filename) == 0:
			raise RuntimeError(f"Could not find and import {filename}.")
		else:
			filename = filename[0]
		data_DS = xr.open_dataset(filename)

		# convert time and remove nasty after decimal point values:
		data_DS['time'] = np.rint(numpydatetime64_to_epochtime(data_DS.time.values)).astype(float)

		# Unify variable names by defining class attributes:
		self.time = data_DS.time.values					# in sec since 1970-01-01 00:00:00 UTC
		self.cloud_flag = data_DS.cloud_flag.values		# 0: clear sky, 1: cloudy

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS
		

class cloudmask:
	"""
		Cloud mask object (EUREC4A HALO HAMP was used for its derivation). The products have
		been created by Marek Jacob (Marek.Jacob@dwd.de).

		For initialisation we need:
		path : str
			Path of the cloud mask data.
		instrument : str
			Specifies the instrument from which the cloud mask was derived. Valid options:
			'mwr', 'radar'
		version : str
			Version of the cloud mask data. Valid option depends on the product.
		which_date : str
			Marks the EUREC4A flight day that shall be imported. To be specified in 
			yyyymmdd (e.g. 20200213)!

		**kwargs:
		cut_low_altitude : bool
			If True, time steps with altitude below 6000 m are removed. If False,
			this won't be done. Default: True
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, instrument, version, which_date, **kwargs):

		# import data:
		if instrument == 'mwr':
			cm_file = glob.glob(path + 
						f"EUREC4A_HALO_HAMP-MWR_cloud_mask_{which_date}_{version}.nc")
		elif instrument == 'radar':
			cm_file = glob.glob(path + 
						f"EUREC4A_HALO_HAMP-Radar_cloud_mask_{which_date}_{version}.nc")
		else:
			raise ValueError("Argument 'instrument' of the initializer function of class " +
								"cloudmask must be in ['mwr', 'radar'].")

		if len(cm_file) == 0:
			raise RuntimeError(f"Could not find and import {cm_file}.")
		else:
			cm_file = cm_file[0]

		if 'cut_low_altitude' in kwargs.keys():
			data_DS = import_HAMP_unified_product(cm_file, cut_low_altitude=kwargs['cut_low_altitude'])
		else:
			data_DS = import_HAMP_unified_product(cm_file, cut_low_altitude=False)

		# Unify variable names by defining class attributes:
		self.time = data_DS.time.values						# in sec since 1970-01-01 00:00:00 UTC
		self.cloud_mask = data_DS.cloud_mask.values			# 0: no cloud detected, 1: probably cloudy, 2: most likely cloudy

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS

	def find_cloudy_periods(self, t_spacing):

		"""
		Find and identify cloudy periods. Two cloudy periods must be separated by at least t_spacing seconds
		of clear sky.

		Run through the cloud mask time series and always count the clear sky seconds (and/or compute the time difference 
		between the last cloudy and the current time step). The counter is incremented when a clear sky second was found 
		and reset to 0 if another cloudy one was found.

		cloudy_idx will save the cloudy scenes and make the periods separable. It will be a 2D list of 'shape'
		(n_cloudy_periods, idx_of_cloudy_scenes_in_nth_cloudy_period).

		Parameters:
		-----------
		t_spacing : int or float
			Defines the time of clear sky conditions that separate two cloudy periods.
		"""

		clearsky_counter = 0
		time0 = self.time[0]
		cloudy_idx = list()
		cloudy_idx_temp = list()		# will save cloudy idx for the current cloudy period
		for k, tt in enumerate(self.time):

			if self.cloud_mask[k] > 1.5:		# most likely cloudy
				clearsky_counter = 0
				time0 = tt
				cloudy_idx_temp.append(k)

			# elif self.cloud_mask[k] > 0 and self.cloud_mask[k] <= 1.5:	# probably cloudy
				# # clearsky_counter = 0
				# # time0 = tt
				# # cloudy_idx_temp.append(k)

				# clearsky_counter += 1

			else:							# clear sky
				clearsky_counter += 1
			
			time_diff = tt - time0		# time difference is computed

			# # # # # # # # # # # count cloudy pixels in k:k+t_spacing

			if clearsky_counter >= t_spacing and time_diff >= t_spacing:		# end of cloudy period was found
				# if cloudy seconds were found in this period, save the indices
				# and reset the temporary variable
				if len(cloudy_idx_temp) > 0:
					cloudy_idx.append(cloudy_idx_temp)
					cloudy_idx_temp = list()

		"""
		number of cloudy periods: len(cloudy_idx)
		total number of cloudy pixels:
		n_cloudy_pixels = 0
		for k in cloudy_idx: n_cloudy_pixels += len(k)
		"""
		
		self.cloudy_idx = cloudy_idx

			