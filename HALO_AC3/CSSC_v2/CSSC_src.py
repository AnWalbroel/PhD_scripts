import numpy as np
import xarray as xr
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gc
import datetime as dt
import pandas as pd
import glob

import multiprocessing
import urllib
import sys
import os
import warnings

from data_classes import *
from data_tools_cssc import *
import pdb

os.environ['OPENBLAS_NUM_THREADS'] = "1"


# some general settings for plots:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


class CSSC:
	"""
	This is a code package to find offsets of microwave radiometer channels onboard the HALO 
	research aircraft by comparing measured with simulated brightness temperatures in clear sky 
	conditions. Information about executing the code package can be found below.

	Guide to generate clear sky sonde comparison netCDF4 file that shows the bias, RMSE and 
	correlation. Lines that are most likely to be edited by the user in the code are marked 
	with "##################".

	For initialisation, we need:
	path_data : dict of str
		Dictionary containing required and optional data paths for the clear sky sonde comparison,
		dropsonde repair, SST download, ...
	path_plot : str
		Path indicating where to save plots to.
	set_dict : dict
		Dictionary containing various settings required for CSSC (also includes data versions,
		time and space boundaries for SST).
	"""

	def __init__(self, path_data, path_plot, set_dict):

		# set attributes: paths and other settings:
		self.path_BAH = path_data['BAH']
		self.path_mwr = path_data['mwr_concat']
		self.path_radar = path_data['radar']
		self.path_dropsonde = path_data['dropsonde']
		self.path_dropsonde_rep = path_data['dropsonde_rep']
		self.path_sst = path_data['sst']
		self.path_dropsonde_sim = path_data['dropsonde_sim']
		self.path_cssc_output = path_data['cssc_output']

		# plot path and other settings:
		for key in set_dict.keys(): self.__dict__[key] = set_dict[key]
		self.path_plot = path_plot

		# check status of some optional measurements: first assume they are not available
		self.status_radar = 0
		if self.path_radar: self.status_radar = 1


		# convert dates to pandas datetime:
		self.start_date = pd.to_datetime(self.start_date, format="%Y-%m-%d")		# in YYYY-MM-DD
		self.end_date = pd.to_datetime(self.end_date, format="%Y-%m-%d")			# in YYYY-MM-DD
		self.daterange = pd.date_range(self.start_date, self.end_date, freq='D')		# daterange with days as increment


	def save_repaired_dropsondes(self, out_filename):

		"""
		Saves the dropsonde data as an nc file named out_filename into path_dropsonde_rep. Units will
		be SI units, and time will be converted to seconds since 2017-01-01 00:00:00 UTC
		(agreed HALO-(AC)3 convention).

		Parameters:
		-----------
		out_filename : str
			Path and name of the output file.
		"""


		# Set attributes:
		long_names = {	'launch_time': "dropsonde launch time",
						'height': 'dropsonde altitude (or height grid)',
						'time': 'time since sonde launch',
						'pres': 'air pressure',
						'temp': 'air temperature',
						'rh': 'relative humidity',
						'u': 'zonal wind speed',
						'v': 'meridional wind speed',
						'wspeed': 'total wind speed',
						'wdir': 'wind direction',
						'lat': 'latitude',
						'lon': 'longitude'}
		for key in long_names: self.DSDS_j[key].attrs['long_name'] = long_names[key]
		self.DSDS_j['height'].attrs['units'] = "m"


		# set global attributes:
		self.DSDS_j.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
		self.DSDS_j.attrs['conventions'] = "CF-1.7"
		self.DSDS_j.attrs['python_version'] = f"python version: {sys.version}"
		self.DSDS_j.attrs['python_packages'] = (f"numpy: {np.__version__}, xarray: {xr.__version__}, " +
												f"matplotlib: {mpl.__version__}")
		datetime_utc = dt.datetime.utcnow()
		self.DSDS_j.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")

		# time encoding
		launch_time = self.DSDS_j.launch_time
		reftime = np.datetime64("2017-01-01T00:00:00").astype("datetime64[s]").astype("float64")
		self.DSDS_j['launch_time'] = self.DSDS_j.launch_time.values.astype("datetime64[s]").astype(np.float64) - reftime
		self.DSDS_j['launch_time'].attrs['units'] = "seconds since 2017-01-01 00:00:00"
		self.DSDS_j['launch_time'].encoding['units'] = 'seconds since 2017-01-01 00:00:00'
		self.DSDS_j['launch_time'].encoding['dtype'] = 'double'
		self.DSDS_j['time'].attrs['units'] = f"seconds since {launch_time.dt.strftime('%Y-%m-%d %H:%M:%S').values}"
		self.DSDS_j['time'].encoding['units'] = f"seconds since {launch_time.dt.strftime('%Y-%m-%d %H:%M:%S').values}"
		self.DSDS_j['time'].encoding['dtype'] = "double"

		self.DSDS_j.to_netcdf(out_filename, mode='w', format='NETCDF4')
		self.DSDS_j.close()

		# clear memory:
		del self.DSDS_j


	def dropsonde_gap_filler(self):

		"""
		Repair dropsonde data by filling gaps, extrapolate to the surface and to the aircraft altitude
		when necessary, regrid and remove outliers (where the gaps will then be filled again). This 
		step is necessary because PAMTRA simulations do not allow nans in the meteo profiles.
		"""

		def fill_gaps(old_var):

			"""
			Old variable (old_var) gets linearly interpolated over data gaps. The function is 
			ignoring nan values at the surface and above the launch altitude.

			Parameters:
			-----------
			old_var : array of floats
				Old variable (1D (height dimension)) whose gaps will be filled by linear interpolation.
			"""

			new_var = deepcopy(old_var)

			# create flag variable indicating if an entry of old_var has been changed: if == 0: not interpol.
			interp_flag = np.zeros(old_var.shape)


			# identify regions of nan values in the middle of the drop. Extrapolation will be handled 
			# in a separate function:
			# identify the highest non-nan entry so we can cut the values above that highest entry:
			# identify the lowest non-nan entry for similar reasons:
			non_nan_idx = np.where(~np.isnan(old_var))[0]
			limits = np.array([non_nan_idx[0], non_nan_idx[-1]])

			temp_var = deepcopy(old_var)
			temp_var = temp_var[limits[0]:limits[1]+1]		# will be the variable where the gaps are filled

			interp_flag_temp = np.zeros(temp_var.shape)

			# identify mid-drop-nan-values: need values after and before the nan:
			nan_idx = np.argwhere(np.isnan(temp_var))
			interp_flag_temp[nan_idx] = 1

			if nan_idx.size == 0:
				return new_var, interp_flag

			else: # then interpolate over nan values: find the hole size via subtraction of subsequent indices

				hole_size = np.zeros((len(nan_idx)+1,)).astype(int)
				k = 0		# index to address a hole ('hole number')
				for m in range(len(temp_var)-1):

					if not np.isnan(temp_var[m+1] - temp_var[m]):
						hole_size[k] = 0

					elif np.isnan(temp_var[m+1] - temp_var[m]): # k shall only be incremented if an END of a hole has been identified:
						if len(nan_idx) == 1: 	# must be handled seperately in case that merely one nan value exists in temp_var
							hole_size[k] = 1
							break

						else:
							if (not np.isnan(temp_var[m+1])) & (np.isnan(temp_var[m])): # END of a hole
								k = k + 1
								continue
							hole_size[k] = hole_size[k] + 1		# k won't be incremented until m finds another non-nan value

					else:
						print("\n Something unexpected happened when trying to find the nan values in the " +
								"middle of the dropsonde launch... Contact 'a.walbroel@uni-koeln.de'. \n")

				# holes have been identified: edit the FIRST hole (editing depends on the size of the hole...)
				c = 0 		# dummy variable needed for the right jumps in hole_size and nan_idx. c is used to address nan_idx and therefore new_var...

				# meanwhile 'hs' just runs through the array hole_size:
				for hs in hole_size:
					for L in range(hs):		# range(0, 1): L = 0
						temp_var[nan_idx[c] + L] = (temp_var[nan_idx[c] - 1] + (L + 1)*(temp_var[int(nan_idx[c] + hs)] - 
													temp_var[nan_idx[c]-1]) / (hs + 1))

					c = c + int(hs)
					if c > len(hole_size)-1:
						break


			# overwrite the possibly holey section and update interp_flag:
			new_var[limits[0]:limits[1]+1] = temp_var
			interp_flag[limits[0]:limits[1]+1] = interp_flag_temp

			return new_var, interp_flag


		def std_extrapol_BAH(DSDS_old, ill_keys, DS_BAH, old_ipflag_dict=dict()):

			"""
			Will extrapolate some atmospheric variables to the ceiling of the dropsonde; old_ipflag will be updated.
			BAHAMAS data serves as extrapolation target.

			Parameters:
			-----------
			DSDS_old : xarray dataset
				Dropsonde dataset with height dimension and variables which require extrapolation to the
				aircraft altitude.
			ill_keys : list of str
				List indicating the variables where the surface values are extrapolated.
			DS_BAH : xarray dataset
				BAHAMAS dataset selected for the launch time of the currently considered dropsonde.
			old_ipflag_dict : dict
				Dictionary with keys being identical to ill_keys indicating if a data variable
				has been interpolated (1) or not (0) at certain heights.
			"""

			DSDS_new = DSDS_old
			new_ipflag_dict = old_ipflag_dict
			n_alt = len(DSDS_new.height.values)
			ceiling = DSDS_new.height.values[-1]	# last entry of altitude


			# loop over variables to be extrapolated:
			launch_time = str(DSDS_new.launch_time.dt.strftime("%Y-%m-%d %H:%M:%SZ").values) # for printing
			alt = DSDS_new.height.values
			for key in ill_keys:

				new_var = DSDS_new[key].values

				# find highest non nan value if it lies below the ceiling:
				idx = np.where(~np.isnan(new_var))[0][-1]

				if not new_ipflag_dict: # in case fill_gaps(...) wasn't called before this one, it's assumed that nothing has been interpolated yet.
					new_ipflag_dict[key] = np.zeros(new_var.shape)


				if key == 'temp':
					# If BAHAMAS Temperature measurement is available use it as target.
					if alt[idx] < 0.75*ceiling:
						print("Insufficient amount of temperature obs for extrapolation at the top of the dropsonde grid. " + 
							f"There are no temperature measurements above {alt[idx]} m. \n")
						continue

					else:
						new_var[idx+1:] = new_var[idx] + (DS_BAH.temp.values - new_var[idx]) / (DS_BAH.alt.values - alt[idx]) * (alt[idx+1:] - alt[idx])
						new_ipflag_dict[key][idx+1:] = 1		# setting the interpol flag


				elif key == 'pres':
					# Pressure: use BAHAMAS data as extrapol. target:
					if alt[idx] < 0.5*ceiling:
						print("Insufficient amount of obs for pressure extrapolation at the top of the dropsonde grid. " + 
							f"There are no pressure measurements above {alt[idx]} m. \n")
						continue

					else:
						new_var[idx+1:] = new_var[idx] + (DS_BAH.pres.values - new_var[idx]) / (DS_BAH.alt.values - alt[idx]) * (alt[idx+1:] - alt[idx])
						new_ipflag_dict[key][idx+1:] = 1		# setting the interpol flag


				elif key in ['u', 'v']:
					# Wind: idea: fill nan values with the mean wind gradient of the highest 20 (non-nan)measurents. It will only be extrapolated if the the last non-nan entry
					# is higher than 0.90*ceiling:
					# other idea: just keep the same wind value
					if alt[idx] < 0.9*ceiling:
						print("Insufficient amount of measurements for wind extrapolation at the top of the dropsonde grid. " + 
							f"There are no wind measurements above {alt[idx]} m. \n")
						continue

					else:
						# # # extra_speed_length = 20		# amount of indices used for wind speed gradient calculation
						# # # for k in range(idx, n_alt_new):
							# # # new_var[n,k] = new_var[n,idx] + (k-idx)*(new_var[n,idx] - new_var[n,idx-extra_speed_length]) / (extra_speed_length + (k-idx))

						# alternative: just use the latest value for higher altitudes:
						new_var[idx+1:] = new_var[idx]
						new_ipflag_dict[key][idx+1:] = 1		# setting the interpol flag


				elif key == 'rh':
					# Relative humidity (RH): Linearly interpolate to the BAHAMAS value
					if alt[idx] < 0.75*ceiling:
						print("Insufficient amount of measurements for relative humidity extrapolation at the top of the dropsonde grid. " + 
							f"There are no rel. hum. measurements above {alt[idx]} m. \n")
						continue

					else:
						new_var[idx+1:] = new_var[idx] + (DS_BAH.rh.values - new_var[idx]) / (DS_BAH.alt.values - alt[idx]) * (alt[idx+1:] - alt[idx])
						new_var[np.argwhere(new_var[:] < 0)] = 0.0		# avoid negative values
						new_ipflag_dict[key][idx+1:] = 1		# setting the interpol flag

			return DSDS_new, new_ipflag_dict


		def repair_surface(old_DS, ill_keys, old_ipflag_dict=dict()):

			"""
			Filling nan values at the surface if the gap to the surface isn't too large (e.g. 
			measurements below 200 m must exist (roughly 10-15 seconds before splash).

			Parameters:
			-----------
			old_DS = xarray dataset
				Dataset containing dropsonde data with non-repaired surface.
			ill_keys : list of str
				List indicating the variables where the surface values are extrapolated.
			old_ipflag_dict : dict
				Dictionary with keys being identical to ill_keys indicating if a data variable
				has been interpolated (1) or not (0) at certain heights.
			"""

			new_DS = old_DS
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # alt = DS.height.values['Z']
			new_ipflag_dict = old_ipflag_dict
			launch_time = str(new_DS.launch_time.dt.strftime("%Y-%m-%d %H:%M:%S").values)

			lim = 200		# if there are no measurements below this altitude then the extrapolation at the surface won't be performed

			if ill_keys == ['Z']:
				threshold_list = [ill_keys, [200], ['m']]
			else:
				threshold_list = [ill_keys, [5.0, 5000.0, 0.5, 0.1, 0.1, 5.0, 5.0],
					['K', 'Pa', '[]', 'deg', 'deg', 'm s-1', 'm s-1']]		# used to check if surface value deviates siginificantly from lowest measurement

			# loop over variables to be repaired:
			for key in ill_keys:

				new_var = new_DS[key].values

				# in case fill_gaps(...) wasn't called before this one, it's assumed that nothing has been interpolated yet.
				if not new_ipflag_dict:
					new_ipflag_dict[key] = np.zeros(new_var.shape)


				# find the first non-nan entry
				idx = np.where(~np.isnan(new_var[:]))[0][0]

				if new_DS.height.values[idx] < lim:
					sfc_gap = np.arange(0,idx)

					if len(sfc_gap) == 0:
						continue
					else:

						# create mean gradient of the variable of 10 measurements above the lowest measurement, or, 
						# if grid is too coarse, take lim-2*lim m average (after identifying height idx of 2*lim):
						if new_DS.height.values[idx+10] > 2*lim:
							idx2 = np.argmin(np.abs(new_DS.height.values - 2*lim))

						else: # take mean grad. of 10 measurem. above lowest measurement:
							idx2 = idx+10

						# mean gradient over idx:idx2+1:
						mean_grad = np.mean(np.diff(new_var[idx:idx2+1]))

						# repair surface:
						for j in sfc_gap:
							new_var[idx-j-1] = new_var[idx] - mean_grad*(j+1)


						# check if sfc value not too far off the lowest measurement:
						if key == 'RH':
							if np.any(new_var[sfc_gap] < 0):
								new_var[sfc_gap] = 0.0
								print(f"Caution, '{key}' surface repair resulted in negative values. " +
										f"Manually set the missing values at the ground to 0 for launch {launch_time}.\n")

							elif np.any(new_var[sfc_gap] > 1.0):
								new_var[sfc_gap] = 1.0
								print(f"Caution, '{key}' surface repair resulted in >1.0. " +
										f"Manually set the missing values at the ground to 1.00 for launch {launch_time}.\n")

						threshold = threshold_list[1][threshold_list[0].index(key)]
						si_unit = threshold_list[2][threshold_list[0].index(key)]
						if np.abs(new_var[0] - new_var[idx]) > threshold:
							print(f"Caution, '{key}' surface value deviates more than {threshold} {si_unit} from the " +
									f"lowest measurement (launch {launch_time}).\n")


						new_ipflag_dict[key][sfc_gap] = 1

				else:
					print(f"No measurements below {lim} m. Extrapolation of '{key}', launch {launch_time}" +
							" would eventually lead to wrong assumptions at the surface. Therefore aborted.\n")
					continue


			return new_DS, new_ipflag_dict


		def mark_outliers(DSDS_old, ill_keys): 

			"""
			Mark outliers: outliers defined when exceeding certain thresholds

			Parameters:
			-----------
			DSDS_old : xarray dataset
				Dropsonde dataset with height dimension.
			ill_keys : list of str
				List indicating the variables that will be checked for outliers.
			"""

			DSDS_new = DSDS_old

			# thresholds are defined by change of meteorol. variable with altitude: e.g. delta p / delta z
			thresholds = {'temp': 0.50, 	# K m-1
							'pres': 40, 	# Pa m-1 
							'rh': 2.5,  	# m-1
							'u': 1,  		# m s-1 m-1
							'v': 1}			# m s-1 m-1]
			dz = np.diff(DSDS_new.height.values) # delta z

			for key in ill_keys:

				if key not in ['lat', 'lon']:
					new_var = DSDS_new[key].values
					d_met = np.diff(new_var)		# change of meteorological variable 'key'

					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						exceed_idx = np.where(np.abs(d_met / dz) >= thresholds[key])[0]
					if len(exceed_idx) > 0: pdb.set_trace()
					new_var[exceed_idx] = np.nan

			return DSDS_new


		def plot_met_profile(sonde_dict, ill_keys, plot_path, plot_filename_base): # plots T profile and saves it in 'plot_path'

			units = ['K', 'Pa', '%']

			for key in ill_keys:	# plot each meteorological variable that has been modified:
				# Plotting after extrapolation:
				font_size = 14
				fig = plt.figure(figsize=(6,9))
				a1 = plt.axes()

				launch_date = datetime.datetime.utcfromtimestamp(sonde_dict['launch_time']).strftime("%Y%m%d_%H%M%S")
				a1.plot(sonde_dict[key], sonde_dict['Z'], linewidth=1.2, color=(0,0,0))

				titletext = r"Dropsonde " + key + " profile from EUREC4A campaign: " + launch_date
				plt.title(titletext, fontsize=font_size, wrap=True)
				a1.set_xlabel(key + " [" + units[ill_keys.index(key)] + "]", fontsize=font_size)
				a1.set_ylabel(r"Height [m]", fontsize=font_size)
				a1.grid(True, axis='x', which='both')
				a1.grid(True, axis='y', which='major')
				a1.set_ylim(bottom=0, top=sonde_dict['Z'][-1])

				if key == 'tdry':
					a1.set_xlim(left=240, right=305)
				elif key == 'pres':
					a1.set_xlim(left=10000, right=105000)
				elif key == 'rh':
					a1.set_xlim(left=0, right=100)

				plt.savefig(plot_path + plot_filename_base + "_" + key + ".png") #, dpi=250, bbox_inches='tight'
				plt.close()


		# check if repaired dropsonde files exist (then path_dropsonde_rep is not empty):
		path_dropsonde_rep_dir = os.path.dirname(self.path_dropsonde_rep)
		if os.listdir(path_dropsonde_rep_dir):
			print("Repaired (gap filled) dropsondes already seem to exist....\n")
			return


		# Import dropsonde data:
		self.dropsondes = dropsondes(self.path_dropsonde, self.sonde_dataset_type, version="", 
										height_grid=self.sonde_height_grid, return_DS=True)
		
		# load BAHAMAS data if available:
		if self.path_BAH:
			self.BAH = BAHAMAS(self.path_BAH, "all", version='unified', return_DS=True)
		else:
			print("Warning! BAHAMAS data path seems empty. Extrapolation between highest dropsonde " +
					"measurement and aircraft will be skipped.")

		tot_failure_warnings = 0			# counts the amount of times a critical variable has got no measurements
		tot_sonde_stuck = 0
		failed_sondes = []
		stuck_sondes = []

		print(f"Found {self.dropsondes.n_sondes} dropsonde(s).")

		# Start iteration over the sondes in the sonde_nc file:
		n_sondes = self.dropsondes.n_sondes
		for j in range(n_sondes):

			# limit datasets to the current sonde; BAH: average over +/- 5 sec of launch time:
			self.DSDS_j = self.dropsondes.DS.isel(launch_time=j) 		# j-th dropsonde dataset
			self.BAH_j = self.BAH.DS.sel(time=slice(self.DSDS_j.launch_time - np.timedelta64(5,"s"), 
													self.DSDS_j.launch_time + np.timedelta64(5,"s")))

			# check if BAH data is left: if true, average over time (and make sure to save the time in the dataset):
			if len(self.BAH_j.time) > 0:
				self.BAH_j['mean_time'] = self.BAH_j.time.mean()
				self.BAH_j = self.BAH_j.mean('time')
				self.BAH_j = self.BAH_j.rename({'mean_time': 'time'})


			launch_date = str(self.DSDS_j.launch_time.values.astype("datetime64[s]")).replace("T", " ")
			dropsonde_date = launch_date[:10].replace("-","")	# date displayed in the filename ... comfy way to find the right BAHAMAS data for std_extrapol
			launch_date_for_filename = str(self.DSDS_j.launch_time.dt.strftime("%Y%m%d_%H%M%SZ").values)

			print("########## Day: " + launch_date + " ##########\n")

			if launch_date_for_filename == '20200131_175736Z':
				print("Skip sonde on ", launch_date_for_filename, " as the sonde has a dry bias.")
				continue


			# add another condition that checks if e.g. nearly no measurements exist at all (for T, P and RH):
			self.obs_height = float(self.BAH_j.alt.values)
			n_musthave = np.count_nonzero(self.DSDS_j.height <= self.obs_height)		# these are the height levels that must be filled

			# limit dropsonde dataset to that height:
			self.DSDS_j = self.DSDS_j.sel(height=slice(0.0, self.obs_height))


			if np.any([np.count_nonzero(~np.isnan(self.DSDS_j.temp.values)) < 0.1*n_musthave,
				np.count_nonzero(~np.isnan(self.DSDS_j.pres.values)) < 0.1*n_musthave,
				np.count_nonzero(~np.isnan(self.DSDS_j.rh.values)) < 0.1*n_musthave,
				np.count_nonzero(~np.isnan(self.DSDS_j.u.values)) < 0.05*n_musthave,
				np.count_nonzero(~np.isnan(self.DSDS_j.v.values)) < 0.05*n_musthave]):
				tot_failure_warnings = tot_failure_warnings + 1
				failed_sondes.append(j)
				print(f"One PAMTRA-critical variable measurement failed. Skipping dropsonde {launch_date}Z. \n")
				continue

			# add yet another condition that checks if the sonde got stuck mid air:
			if (np.all(np.isnan(self.DSDS_j.temp.sel(height=slice(0.0, 1500.0)).values)) or
				np.all(np.isnan(self.DSDS_j.pres.sel(height=slice(0.0, 1500.0)).values)) or
				np.all(np.isnan(self.DSDS_j.rh.sel(height=slice(0.0, 1500.0)).values))):
				# then I assume that the whole launch was doomed
				print("Sonde got stuck in mid air. 'Z' doesn't seem to include any values < 1500 m.\n")
				stuck_sondes.append(j)
				tot_sonde_stuck = tot_sonde_stuck + 1
				continue


			# subsequent variables will be cured from holey nan value disease...:
			# pressure, temperature, relhum, wind (u & v & w), lat, lon.
			ill_keys = ['temp', 'pres', 'rh', 'lat', 'lon', 'u', 'v']
			sonde_ipflag = dict()		# will contain the interpolation flags for interpolated nan values in the middle of the drop
			for key in ill_keys:
				sonde_ipflag[key] = np.full_like(self.DSDS_j[key].values, 0)
				var_gap_filled, sonde_ipflag[key] = fill_gaps(self.DSDS_j[key].values)
				self.DSDS_j[key] = xr.DataArray(var_gap_filled, dims=['height'], attrs=self.DSDS_j[key].attrs)


			# now we still need to handle the nan values at the surface: perform surface repair for the atmospheric 
			# parameters. If there are no non-nan values in the lowest 5 % of the variable --> don't interpolate 
			# because the assumption would eventually lead to senseless surface values:
			self.DSDS_j, sonde_ipflag = repair_surface(self.DSDS_j, ill_keys, sonde_ipflag)


			# Extrapolating the ill_keys to the ceiling of the dropsondes (e.g. below aircraft altitude):
			# Need bahamas file for extrapolation limit:
			if self.path_BAH:
				self.DSDS_j, sonde_ipflag = std_extrapol_BAH(self.DSDS_j, ill_keys, self.BAH_j, sonde_ipflag)


			# find outliers and mark them (as nan): afterwards fill them again
			self.DSDS_j = mark_outliers(self.DSDS_j, ill_keys)
			for key in ill_keys:
				var_gap_filled, sonde_ipflag[key] = fill_gaps(self.DSDS_j[key].values)
				self.DSDS_j[key] = xr.DataArray(var_gap_filled, dims=['height'], attrs=self.DSDS_j[key].attrs)


			# save some additional BAHAMAS data into the dropsonde dataset:
			self.DSDS_j['reference_alt'] = xr.DataArray(self.obs_height, dims=[], 
													attrs={'long_name': "Aircraft altitude at dropsonde launch",
															'units': "m"})
			self.DSDS_j['reference_lat'] = xr.DataArray(self.BAH_j.lat.values, dims=[],
													attrs={'long_name': "Aircraft latitude at dropsonde launch",
															'units': "deg N"})
			self.DSDS_j['reference_lon'] = xr.DataArray(self.BAH_j.lon.values, dims=[],
													attrs={'long_name': "Aircraft longitude at dropsonde launch",
															'units': "deg E"})


			# Save the extrapolated sonde dictionary to a new nc file:
			path_dropsonde_rep_dir = os.path.dirname(self.path_dropsonde_rep)
			if not os.path.exists(path_dropsonde_rep_dir):
				os.makedirs(path_dropsonde_rep_dir)
			out_filename = f"HALO-AC3_HALO_Dropsondes_repaired_{launch_date_for_filename}_v01.nc"
			out_filename = os.path.join(self.path_dropsonde_rep, out_filename)

			self.save_repaired_dropsondes(out_filename)

		# clear memory:
		del self.dropsondes, self.BAH

		print(f"Dropsonde repairing service finds {tot_failure_warnings} failed sondes and {tot_sonde_stuck} stuck in mid-air.")


	def download_sst_data(self):

		"""
		Get SST data from CMC0.1deg-CMC-L4-GLOB-v3.0 via OPENDAP tool for each day between start and end
		date at 12 UTC. For this we need to select latitude and longitude boundaries as well as 
		the required dates.
		"""

		def better_ceil(input, digits):	# ceil to certain digit after decimal point
			return np.round(input + 0.49999999*10**(-digits), digits)

		def better_floor(input, digits):# floor to certain digit after decimal point
			return np.round(input - 0.49999999*10**(-digits), digits)

		def lat_bound_to_slice(lat_bound):	# converts latitude boundaries to OPENDAP slices of lat
			return [(lat_bound[0] + 90)*10, (lat_bound[1] + 90)*10]

		def lon_bound_to_slice(lon_bound):	# converts longitude boundaries to OPENDAP slices of lon
			return [(lon_bound[0] + 180)*10, (lon_bound[1] + 180)*10]


		# in python3, .urlretrieve() and .urlcleanup() are part of urllib.request,
		request = urllib.request


		# Check the existence of the path where the files shall be saved to:
		path_sst_dir = os.path.dirname(self.path_sst)
		if not os.path.exists(path_sst_dir):
			os.makedirs(path_sst_dir)

		# check if SST files exist (then path_sst_dir is not empty):
		if os.listdir(path_sst_dir):
			print("SST files already seem to exist....\n")
			return
		else:

			# need to convert the lat and lon boundaries to the slices of the OPENDAP tool:
			# but first get the to a precision of 1 after decimal point:
			self.lat_bound = [better_floor(self.sst_lat[0], 1), better_ceil(self.sst_lat[1], 1)]
			self.lon_bound = [better_floor(self.sst_lon[0], 1), better_ceil(self.sst_lon[1], 1)]

			lat_slice = lat_bound_to_slice(self.lat_bound)
			lon_slice = lon_bound_to_slice(self.lon_bound)


			for dr in self.daterange:

				print(dr)

				# try to load a file:
				request.urlcleanup()		# clear the cache of previous urlretrieve calls

				# define some shortcuts:
				daynumber = dr.dayofyear	# day number of the specified year
				thisyear = str(dr.year)
				date_formatted = dr.strftime("%Y%m%d")
				lat_slice_formatted = f"{int(lat_slice[0])}:1:{int(lat_slice[1])}"	# e.g. 450:1:900
				lon_slice_formatted = f"{int(lon_slice[0])}:1:{int(lon_slice[1])}" 	# e.g. 1140:1:1400
				daynumber = f"{daynumber:03}"

				outfile_name = date_formatted + "120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc.nc4"
				to_be_retrieved = ("https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/CMC/CMC0.1deg/v3/" +
									f"{thisyear}/{daynumber}/{date_formatted}120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc.nc4?" +
									f"time%5B0:1:0%5D,lat%5B{lat_slice_formatted}%5D,lon%5B{lon_slice_formatted}%5D,analysed_sst" +
									f"%5B0:1:0%5D%5B{lat_slice_formatted}%5D%5B{lon_slice_formatted}%5D,analysis_error" +
									f"%5B0:1:0%5D%5B{lat_slice_formatted}%5D%5B{lon_slice_formatted}%5D,sea_ice_fraction" +
									f"%5B0:1:0%5D%5B{lat_slice_formatted}%5D%5B{lon_slice_formatted}%5D,mask" +
									f"%5B0:1:0%5D%5B{lat_slice_formatted}%5D%5B{lon_slice_formatted}%5D")


				# Retrieve data:
				try:
					request.urlretrieve(to_be_retrieved, self.path_sst + outfile_name)


				except:	# if it couldn't be downloaded continue with next day
					print("Could not retrieve '" + to_be_retrieved + "' from server.")
					continue


	def fwd_sim_dropsondes_to_TB(self):

		"""
		Forward simulate dropsonde data (repaired with dropsonde_gap_filler and exported with 
		save_repaired_dropsondes) with PAMTRA to obtain brightness temperatures (TBs) similar to
		microwave radiometers onboard HALO.
		"""

		if 'PAMTRA_DATADIR' not in os.environ:
			os.environ['PAMTRA_DATADIR'] = "" # actual path is not required, but the variable has to be defined.
		import pyPamtra

		# Check if the PAMTRA output path exists:
		path_dropsonde_sim_dir = os.path.dirname(self.path_dropsonde_sim)
		if not os.path.exists(path_dropsonde_sim_dir):
			os.makedirs(path_dropsonde_sim_dir)

		# check if simulated TBs exist (then path_dropsonde_sim_dir is not empty):
		if os.listdir(path_dropsonde_sim_dir):
			print("Simulated dropsondes already seem to exist....\n")
			return


		# identify dropsonde and SST files:
		files_dropsondes = sorted(glob.glob(self.path_dropsonde_rep + "*v01.nc"))
		files_sst = sorted(glob.glob(self.path_sst + "*.nc.nc4"))


		# import dropsonde and SST data:
		self.DSDS = xr.open_mfdataset(files_dropsondes, combine='nested', concat_dim='launch_time')
		self.SST_DS = xr.open_mfdataset(files_sst, combine='nested', concat_dim='time')


		# loop over dropsondes, but use parallel pamtra on frequencies:
		n_sondes = len(self.DSDS.launch_time.values)
		for j in range(n_sondes):

			# select respective dropsonde and SST data:
			self.DSDS_j = self.DSDS.isel(launch_time=j) 		# j-th dropsonde dataset
			self.SST_DS_j = self.SST_DS.sel(time=self.DSDS_j.launch_time, method='nearest')

			# furthermore, limit dropsonde height to below flight altitude:
			self.DSDS_j = self.DSDS_j.sel(height=slice(0.0, self.DSDS_j.reference_alt.values))
			n_alt = len(self.DSDS_j.height.values)		# number of height levels

			print(f"Simulating sonde {str(self.DSDS_j.launch_time.values.astype('datetime64[s]'))}")


			# find the sonde launches that produced too many nan values so that cannot run: use the RH, T, P for that:
			# final check for nans:
			if not (np.all([~np.isnan(self.DSDS_j.temp.values), ~np.isnan(self.DSDS_j.pres.values), ~np.isnan(self.DSDS_j.rh.values)])):
				print('WARNING, nans detected in PAMTRA critical variable.')
				print(f"    nan-counts (temp, pres, rh): {np.isnan(self.DSDS_j.temp).sum()}, " +
						f"{np.isnan(self.DSDS_j.pres).sum()}, {np.isnan(self.DSDS_j.rh).sum()}")
				continue

			# check for surface wind data:
			u_10 = self.DSDS_j.u.sel(height=10.0, method='nearest')
			v_10 = self.DSDS_j.v.sel(height=10.0, method='nearest')
			if (not np.isnan(u_10.values + v_10.values)) & (np.abs(u_10.height.values - 10.0) < 10.0): 
				u_10 = float(u_10.values)
				v_10 = float(v_10.values)
			else:
				u_10 = 0.0
				v_10 = 0.0


			# HAMP FREQUENCIES:
			frq = [22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,31.4000,50.3000,51.7600,52.8000,
					53.7500,54.9400,56.6600,58.0000,90.0000,110.250,114.550,116.450,117.350,120.150,
					121.050,122.950,127.250,175.810,178.310,179.810,180.810,181.810,182.710,
					183.910,184.810,185.810,186.810,188.310,190.810]

			# create pamtra object; change settings:
			pam = pyPamtra.pyPamtra()

			pam.nmlSet['hydro_adaptive_grid'] = True
			pam.nmlSet['add_obs_height_to_layer'] = False		# adds observation layer height to simulation height vector
			pam.nmlSet['passive'] = True						# passive simulation
			pam.nmlSet['active'] = False						# False: no radar simulation

			pamData = dict()
			shape2d = [1, 1]

			# use highest non nan values of sonde for location information:
			if ~np.isnan(self.DSDS_j.reference_lon.values):
				reflon = self.DSDS_j.reference_lon.values
			else:
				reflon = self.DSDS_j.lon.values[~np.isnan(self.DSDS_j.lon.values)][-1]

			if ~np.isnan(self.DSDS_j.reference_lat.values):
				reflat = self.DSDS_j.reference_lat.values
			else:
				reflat = self.DSDS_j.lat.values[~np.isnan(self.DSDS_j.lat.values)][-1]

			obs_height = np.array([self.DSDS_j.reference_alt.values])

			# set time and location info:
			pamData['lon'] = np.broadcast_to(reflon, shape2d)
			pamData['lat'] = np.broadcast_to(reflat, shape2d)
			pamData['timestamp'] = np.broadcast_to(self.DSDS_j.launch_time.values.astype('datetime64[s]').astype('float64'), shape2d)
			pamData['obs_height'] = np.broadcast_to(obs_height, shape2d + [len(obs_height), ])

			# surface type & reflectivity:
			pamData['sfc_type'] = np.zeros(shape2d)			# 0: ocean, 1: land
			pamData['sfc_refl'] = np.chararray(shape2d)
			pamData['sfc_refl'][:] = 'F'
			pamData['sfc_refl'][pamData['sfc_type'] == 1] = 'S'


			# meteorolog. surface information:
			# to find the SST: use the designated lat,lon in pamData to find the closest entry in the GHRSST dataset:
			self.SST_DS_j = self.SST_DS_j.sel(lat=reflat, lon=reflon, method='nearest')
			sst = np.array([self.SST_DS_j.analysed_sst.values])


			# save data to pamData dict:
			pamData['groundtemp'] = np.broadcast_to(sst, shape2d)
			pamData['wind10u'] = np.broadcast_to(u_10, shape2d)
			pamData['wind10v'] = np.broadcast_to(v_10, shape2d)

			# 3d variables:
			shape3d = shape2d + [n_alt]
			pamData['hgt_lev'] = np.broadcast_to(self.DSDS_j.height.values, shape3d)
			pamData['temp_lev'] = np.broadcast_to(self.DSDS_j.temp.values, shape3d)		# in K
			pamData['press_lev'] = np.broadcast_to(self.DSDS_j.pres.values, shape3d)	# in Pa
			pamData['relhum_lev'] = np.broadcast_to(self.DSDS_j.rh.values, shape3d)		# in %

			# 4d variables: hydrometeors:
			shape4d = [1, 1, n_alt-1, 1]			# potentially 5 hydrometeor classes with this setting
			pamData['hydro_q'] = np.zeros(shape4d)
			pamData['hydro_q'][...,0] = 0# CLOUD

			# descriptorfile must be included. otherwise, pam.p.nhydro would be 0 which is not permitted. (OLD DESCRIPTOR FILE)
			descriptorFile = np.array([
				  #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
				   ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere', 'khvorostyanov01_drops', -99.0)],
				  dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
				  )
			for hyd in descriptorFile: pam.df.addHydrometeor(hyd)


			# Create pamtra profile and go:
			pam.createProfile(**pamData)

			n_cpus = int(multiprocessing.cpu_count()/2)		# half the number of available CPUs
			pam.runParallelPamtra(frq, pp_deltaX=0, pp_deltaY=0, pp_deltaF=1, pp_local_workers=n_cpus)

			# save output:
			launch_date_for_filename = str(self.DSDS_j.launch_time.dt.strftime("%Y%m%d_%H%M%SZ").values)
			filename_out = os.path.join(self.path_dropsonde_sim, f"HALO-AC3_HALO_Dropsondes_PAMTRA_simulated_{launch_date_for_filename}_v01.nc")
			pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)

			print(f"Saved PAMTRA simulations to {filename_out}.")


		# clear memory:
		del self.DSDS, self.SST_DS


	def TB_comparison(self):

		"""
		Compares observed and measured brightness temperatures (TBs) for clear sky scenes and computes offsets in 
		the radiometer data. Identifies clear sky scenes based on TB standard deviation thresholds (higher 
		noise in TBs in cloudy scenes). Radar data can be used to find additional cloudy scenes. 
		"""

		def post_process_pamtra(DS):

			"""
			Post process PAMTRA output by removing obsolete dimensions, selecting the 
			needed angle, average over polarizations, perform double side band averaging.

			Parameters:
			-----------
			DS : xarray dataset
				Dataset that will be post processed.
			"""

			# remove obsolete dimensions and select the right frequencies:
			DS = DS.isel(grid_y=0, angles=0, outlevel=0)	# angle index 0 == nadir (angle==180 deg); index -1 == zenith (angle==0 deg)
			DS['tb'] = DS.tb.mean(axis=-1)		# average over polarisation

			# double side band averaging:
			tb, freq_sb = Gband_double_side_band_average(DS.tb.values, DS.tb.frequency.values)
			tb, freq_sb = Fband_double_side_band_average(tb, freq_sb)
			DS = DS.sel(frequency=freq_sb)
			DS['tb'] = xr.DataArray(tb, dims=['grid_x', 'frequency'], coords=DS.tb.coords)

			return DS


		# Check if the sonde comparison output and plot path exist:
		out_path_dir = os.path.dirname(self.path_cssc_output)
		if not os.path.exists(out_path_dir):
			os.makedirs(out_path_dir)
		plot_path_dir = os.path.dirname(self.path_plot)
		if not os.path.exists(plot_path_dir):
			os.makedirs(plot_path_dir)


		# check if TB_comparison has already produced files:
		if os.listdir(out_path_dir):
			files_check = glob.glob(out_path_dir + "/" + "*TB_comparison*.nc")
			if len(files_check) > 0:
				print("TB_comparison seems to have been executed before. The following files were found....\n")
				for fi in files_check: print(fi)
				return


		# identify HALO-HAMP MWR data and simulated dropsonde files:
		files_mwr = sorted(glob.glob(self.path_mwr + "*.nc"))
		files_PAM_DS = sorted(glob.glob(self.path_dropsonde_sim + "*.nc"))
		self.PAM_DS = xr.open_mfdataset(files_PAM_DS, combine='nested', concat_dim='grid_x', preprocess=post_process_pamtra)


		# rename some dimensions and variables because I aligned different times along grid_x:
		self.PAM_DS = self.PAM_DS.rename_dims({'grid_x': 'time'})
		self.PAM_DS = self.PAM_DS.rename({'grid_x': 'time'})
		self.PAM_DS = self.PAM_DS.assign_coords({'time': self.PAM_DS.datatime})


		# Loop over MWR data files (one file for each research flight) and import 
		# HALO-HAMP mwr data, and slice the simulated dropsonde dataset:
		for file_mwr in files_mwr:

			# import MWR data and limit dropsonde data to that research flight time:
			self.MWR_DS_j = xr.open_dataset(file_mwr)
			self.MWR_DS_j = self.MWR_DS_j.sortby('freq')		# was eventually not sorted in ascending order
			self.PAM_DS_j = self.PAM_DS.sel(time=slice(self.MWR_DS_j.time[0], self.MWR_DS_j.time[-1]))
			self.n_freq = len(self.PAM_DS_j.frequency)
			n_sondes = len(self.PAM_DS_j.time)

			print(f"Comparing with {file_mwr}....")


			if len(self.PAM_DS_j.time) == 0:
				print(f"Could not find any dropsonde files between {self.MWR_DS_j.time[0].dt.strftime('%Y%m%d_%H%M%S').values} " +
						f"and {self.MWR_DS_j.time[-1].dt.strftime('%Y%m%d_%H%M%S').values}.")
				continue


			# load radar data if possible:
			if self.path_radar:
				launch_times_npdt = self.PAM_DS_j.time.values

				def cut_time(DS):
		
					# filter for time close to dropsonde launches:
					idx_temp = [np.where(np.abs(DS.time.values - lt) < np.timedelta64(10, "s"))[0] for lt in launch_times_npdt]
					idx = np.array([])
					for ii in idx_temp: idx = np.concatenate((idx, ii))
					idx = idx.astype('int64')

					DS = DS.isel(time=idx)

					# remove unnecessary variables:
					remove_vars = ["tpow", "npw1", "npw2", "cpw1", "cpw2", "grst", 
									"aziv", "LO_Frequency", "DetuneFine", "SNRgc", "VELgc", "RMSgc", 
									"LDRgc", "NPKgc", "SNRg", "VELg", "RMSg", "LDRg", 
									"NPKg", "SNRcx", "RHO", "DPS", "RHOwav", "LDRnormal", "HSDco", 
									"HSDcx", "ISDRco", "ISDRcx", "MRMco", "MRMcx", "RadarConst",
									"SNRCorFaCo", "SNRCorFaCx", "SKWg"]
					DS = DS.drop_vars(remove_vars)

					# compute radar refl in dBZ:
					DS['dBZ'] = 10*np.log10(DS.Zg)				# equivalent reflectivity factor in dBZ

					return DS

				# identify radar files:
				files_radar = sorted(glob.glob(self.path_radar + "*.nc"))
				RADAR_DS = xr.open_mfdataset(files_radar, combine='nested', concat_dim='time', preprocess=cut_time)


				# compute height to avoid ground clutter:
				RADAR_DS['alt'] = xr.DataArray(np.interp(RADAR_DS.time.values.astype("float64"), 
												launch_times_npdt.astype("float64"), self.PAM_DS_j.outlevels.values),
												dims=['time'])
				radar_height = np.full((len(RADAR_DS.time), len(RADAR_DS.range)), np.nan)
				for kk in range(len(RADAR_DS.time)):
					radar_height[kk,:] = RADAR_DS['alt'].values[kk] - RADAR_DS.range.values
				RADAR_DS['height'] = xr.DataArray(radar_height, dims=['time', 'range'])


				# count bins with radar reflectivity > -40 dBZ for a time step (watch for ground clutter):
				RADAR_DS['cloudy_flag'] = xr.where((RADAR_DS.dBZ > -40.0) & (RADAR_DS.height > 300.0), True, False)
				RADAR_DS['refl_bins_count'] = RADAR_DS['cloudy_flag'].sum("range")
				idx_radar = [np.where(np.abs(RADAR_DS.time.values - lt) < np.timedelta64(10, "s"))[0] for lt in self.PAM_DS_j.time.values]

				# time-averaged number of reflective bins for each dropsonde launch:
				avg_refl_bins_count = np.ones((n_sondes,))*9999
				for kk, i_r in enumerate(idx_radar): avg_refl_bins_count[kk] = int(RADAR_DS['refl_bins_count'][i_r].mean())
				avg_refl_bins_count = avg_refl_bins_count.astype("int64")
				radar_cloudy_flag = avg_refl_bins_count > 3

				# clear memory:
				del RADAR_DS, radar_height

			else:
				radar_cloudy_flag = np.full((n_sondes,), False)


			# mean and std dev of MWR TBs around dropsonde launches:
			idx = [np.where(np.abs(self.MWR_DS_j.time.values - lt) < np.timedelta64(10, "s"))[0] for lt in self.PAM_DS_j.time.values]
			tb_mean = np.zeros((n_sondes, self.n_freq))
			tb_std = np.zeros((n_sondes, self.n_freq))
			for k, ii in enumerate(idx):
				tb_mean[k,:] = np.asarray([np.nanmean(self.MWR_DS_j.TB.values[ii,:], axis=0)])
				tb_std[k,:] = np.asarray([np.nanstd(self.MWR_DS_j.TB.values[ii,:], axis=0, dtype=np.float64)])


			# max_tb_stddev will contain the maximum std deviation among all channels for each sonde launch
			max_tb_stddev = np.reshape(np.repeat(np.nanmax(tb_std, axis=1), self.n_freq), tb_std.shape)


			# proxy for cloudy scenes: find if std dev threshold is surpassed for any sonde launch:
			# For the G band entries we consider the std. of all (25) channels. In case it's greater than the
			# threshold, the G band entries of tb_used are set to False (==cloudy). 
			# The remaining channels (K,V,W,F) are assigned True if the std. in ALL (K,V,W,F) channels 
			# is less than the threshold. In case ANY non G band channel shows a greater std. dev.,
			# the remaining K,V,W,F channels are set to False in tb_used.
			stddev_threshold = 1.0		# in Kelvin
			tb_used = np.ones((n_sondes, self.n_freq), dtype=bool)	# will contain TBs of the MWR in cloudfree scenes

			# find indices of the different bands:
			idx_G = select_MWR_channels(tb_mean, self.PAM_DS_j.frequency.values, band='G', return_idx=2)
			idx_KVWF = select_MWR_channels(tb_mean, self.PAM_DS_j.frequency.values, band='K+V+W+F', return_idx=2)
			cloudy_G = [np.any(tb_std[k,:] > stddev_threshold) for k in range(n_sondes)]
			cloudy_rest = [np.any(tb_std[k,idx_KVWF] > stddev_threshold) for k in range(n_sondes)]

			# in tb_used: set cloudy launches to false:
			for k in range(n_sondes):
				if cloudy_G[k]: tb_used[k,idx_G] = False
				if cloudy_rest[k]: tb_used[k,idx_KVWF] = False

				# set tb_used to False, if radar sees anything within window
				if radar_cloudy_flag[k]: tb_used[k,:] = False

			# remove cases with extreme bias: considered cloudy (or sea ice) if the bias_threshold is exceeded
			bias_threshold = 30.0
			for k in range(self.n_freq): tb_used[np.abs(self.PAM_DS_j.tb.values[:,k] - tb_mean[:,k]) > bias_threshold, k] = False


			# Save stuff to dataset:
			TB_stat_DS = xr.Dataset(coords={'time': (['time'], self.PAM_DS_j.datatime.values,
														{'long_name': "Dropsonde launch time"}),
											'freq': (['freq'], self.PAM_DS_j.frequency.values,
														{'long_name': "Single band frequencies",
														'units': "GHz"})})


			# save data to dataset:
			TB_stat_DS['tb_mean'] = xr.DataArray(tb_mean, dims=['time', 'freq'])
			TB_stat_DS['tb_std'] = xr.DataArray(tb_std, dims=['time', 'freq'])
			TB_stat_DS['tb_sim'] = xr.DataArray(self.PAM_DS_j.tb.values, dims=['time', 'freq'])
			TB_stat_DS['tb_used'] = xr.DataArray(tb_used, dims=['time', 'freq'])
			TB_stat_DS['obsheight'] = xr.DataArray(self.PAM_DS_j.outlevels.values, dims=['time'])
			TB_stat_DS['lon'] = xr.DataArray(self.PAM_DS_j.longitude.values, dims=['time'])
			TB_stat_DS['lat'] = xr.DataArray(self.PAM_DS_j.latitude.values, dims=['time'])
			TB_stat_DS['stddev_threshold'] = stddev_threshold
			TB_stat_DS['bias_threshold'] = bias_threshold
			TB_stat_DS['date'] = str(self.MWR_DS_j.time[0].dt.strftime("%Y%m%d").values)

			# compute bias, rmse, correlation coeff, ... for clear sky scenes according to tb_used:
			TB_stat_DS['tb_mean_cs'] = xr.where(TB_stat_DS.tb_used, TB_stat_DS.tb_mean, np.nan)	# clear sky tb_mean
			TB_stat_DS['tb_std_cs'] = xr.where(TB_stat_DS.tb_used, TB_stat_DS.tb_std, np.nan)		# clear sky tb_std
			TB_stat_DS['tb_sim_cs'] = xr.where(TB_stat_DS.tb_used, TB_stat_DS.tb_sim, np.nan)		# clear sky tb_sim
			TB_stat_DS['bias'] = (TB_stat_DS.tb_mean_cs - TB_stat_DS.tb_sim_cs).mean('time')
			TB_stat_DS['rmse'] = np.sqrt(((TB_stat_DS.tb_mean_cs - TB_stat_DS.tb_sim_cs)**2).mean('time'))
			TB_stat_DS['R'] = xr.corr(TB_stat_DS.tb_sim_cs, TB_stat_DS.tb_mean_cs, dim='time')
			TB_stat_DS['N'] =  xr.DataArray(np.count_nonzero(~np.isnan(TB_stat_DS['tb_mean_cs']), axis=0), dims=['freq'])


			# Give attributes:
			vars_descr = {	'tb_mean': "TBs measured by microwave radiometer HAMP, averaged over 10 sec around dropsonde launches",
							'tb_std': "As tb_mean, but with standard deviation",
							'tb_sim': "TBs simulated with PAMTRA based on dropsonde data without hydrometeors and over ocean",
							'tb_used': "Flag indicating whether TB comparison should be applied or not. If True, clear sky and ready for comparison.",
							'obsheight': "Aircraft altitude (BAHAMAS) averaged over 5 sec around dropsonde launch; also used as obs_height in PAMTRA",
							'lon': "Aircraft longitude (BAHAMAS) averaged over 5 sec around dropsonde launch",
							'lat': "Aircraft latitude (BAHAMAS) averaged over 5 sec around dropsonde launch"}
			vars_units = {	'tb_mean': "K", 'tb_std': "K", 'tb_sim': "K", 'tb_used': "", 'obsheight': "m",
							'lon': 'deg E', 'lat': 'deg N'}
			vars_longname = {'tb_mean': "Mean measured TB",
							'tb_std': "Standard deviation of measured TB",
							'tb_sim': "Simulated TB",
							'tb_used': "Clear sky flag",
							'obsheight': "Observation height",
							'lon': "Longitude",
							'lat': "Latitude"}

			for key in vars_descr.keys():
				TB_stat_DS[key].attrs['long_name'] = vars_longname[key]
				TB_stat_DS[key].attrs['description'] = vars_descr[key]
				TB_stat_DS[key].attrs['units'] = vars_units[key]

			TB_stat_DS['tb_used'].attrs['comment'] = ("For G band frequencies, all channels " +
														"are checked whether they exceed stddev_threshold. " +
														"If any exceeds stddev_threshold, all G band channels are set False. " +
														"For the other channels, it suffices to be considered a cloudy case (tb_used=False) " +
														"if any non-G-band HAMP channel exceeds stddev_threshold. Furthermore, if " +
														"bias_threshold is exceeded for a certain frequency or dropsonde launch, tb_used " +
														"is also set to False.")
			TB_stat_DS['stddev_threshold'].attrs = {'long_name': "Standard deviation threshold",
													'description': "If standard deviation threshold is exceeded, the scene is potentially cloudy",
													'units': "K"}
			TB_stat_DS['bias_threshold'].attrs = {'long_name': "Bias threshold",
													'description': "If bias_threshold is exceeded, TBs are not be compared",
													'units': "K"}
			TB_stat_DS['date'].attrs = {'long_name': "Date of the research flight",
										'description': "First radiometer time step serves as date indicator"}
			TB_stat_DS['bias'].attrs = {'long_name': "Bias between measured and simulated TBs",
										'description': "Bias computed as tb_mean - tb_sim for clear sky cases (tb_used=True) averaged over time",
										'units': "K"}
			TB_stat_DS['rmse'].attrs = {'long_name': "RMSE between measured and simulated TBs",
										'description': "RMSE computed as sqrt(((tb_mean - tb_sim)**2).mean('time')) for clear sky cases (tb_used=True)",
										'units': "K"}
			TB_stat_DS['R'].attrs = {'long_name': "Pearson correlation coefficient for measured and simulated TBs",
										'description': "Pearson correlation coefficient computed over the time axis for clear sky cases (tb_used=True)",
										'units': ""}
			TB_stat_DS['N'].attrs = {'long_name': "Number of sondes applied for TB comparison for each frequency",
										'units': ""}
			for key in ['tb_mean_cs', 'tb_std_cs', 'tb_sim_cs']: TB_stat_DS[key].attrs['description'] = f"As {key[:-3]} but for tb_used==True"


			# Bring back coordinate attributes, which xarray deletes automatically:
			TB_stat_DS['freq'].attrs = {'long_name': "Single band frequencies",
										'units': "GHz"}


			# Global attributes:
			TB_stat_DS.attrs['title'] = ("HALO-(AC)3 HALO brightness temperature (TB) comparison between " +
											"PAMTRA-simulated dropsondes and HAMP microwave radiometer observations")
			TB_stat_DS.attrs['author'] = ("Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, " +
											"University of Cologne, Cologne, Germany")
			TB_stat_DS.attrs['python_version'] = f"python version: {sys.version}"
			TB_stat_DS.attrs['python_packages'] = (f"numpy: {np.__version__}, xarray: {xr.__version__}, " +
													f"pandas: {pd.__version__}, matplotlib: {mpl.__version__}")
			datetime_utc = dt.datetime.utcnow()
			TB_stat_DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")


			# time encoding:
			reftime = np.datetime64("2017-01-01T00:00:00").astype("datetime64[s]").astype("float64")
			TB_stat_DS['time'] = TB_stat_DS.time.values.astype("datetime64[s]").astype(np.float64) - reftime
			TB_stat_DS['time'].attrs['units'] = "seconds since 2017-01-01 00:00:00"
			TB_stat_DS['time'].attrs['long_name'] = "Dropsonde launch time"
			TB_stat_DS['time'].encoding['units'] = 'seconds since 2017-01-01 00:00:00'
			TB_stat_DS['time'].encoding['dtype'] = 'double'

			# export to file:
			launch_date_for_filename = str(self.MWR_DS_j.time[0].dt.strftime("%Y%m%d").values)
			filename = f"HALO-AC3_HALO_HAMP_dropsondes_TB_comparison_{launch_date_for_filename}_v01.nc"
			output_filename = self.path_cssc_output + filename
			TB_stat_DS.to_netcdf(output_filename, mode='w', format='NETCDF4')

			print(f"Saved TB comparison to '{output_filename}'.")


			# Visualizing results:
			self.visualise_TB_comparison(TB_stat_DS)

			TB_stat_DS.close()
			del TB_stat_DS


	def get_TB_offsets(self):

		"""
		Compute offsets (and slopes) of the measured radiometer TB data based on the output
		of TB_comparison for each research flight.
		"""

		# Import TB_comparison output:
		files = sorted(glob.glob(self.path_cssc_output + "*TB_comparison*.nc"))
		TB_stat_DS = xr.open_mfdataset(files, combine='nested', concat_dim='time')
		self.n_freq = len(TB_stat_DS.freq)


		# check if get_TB_offsets has already produced files:
		out_path_dir = os.path.dirname(self.path_cssc_output)
		if os.listdir(out_path_dir):
			files_check = glob.glob(out_path_dir + "/" + "*TB_offset_correction*.nc")
			if len(files_check) > 0:
				print("get_TB_offsets seems to have been executed before. The following files were found....\n")
				for fi in files_check: print(fi)
				return


		# group by date and compute biases, offsets, slopes:
		TB_stat_DS_grouped = TB_stat_DS.groupby('date')


		# take mean bias as backup value in case few clear sky dropsondes were detected:
		bias_mean = (TB_stat_DS.tb_mean - TB_stat_DS.tb_sim).where(TB_stat_DS.tb_used).mean('time')
		N_daily = TB_stat_DS.tb_used.groupby(TB_stat_DS.date).sum()			# number of used sondes for freqs on that day
		

		# initialise arrays and loop over the groups:
		dates = xr.DataArray(np.unique(TB_stat_DS.date), dims=['date'], attrs=TB_stat_DS.date.attrs)
		bias_daily = xr.DataArray(np.nan, coords=[dates, TB_stat_DS.freq])
		offsets_daily = xr.DataArray(np.nan, coords=[dates, TB_stat_DS.freq])
		slopes_daily = xr.DataArray(np.nan, coords=[dates, TB_stat_DS.freq])
		R_daily = xr.DataArray(np.nan, coords=[dates, TB_stat_DS.freq])

		n_f_bias = 2		# number of sondes required to accept daily bias; if not surpassed, campaign-mean bias is used
		for date, DS_date in TB_stat_DS_grouped:
			# use the daily bias if sufficient sondes were available, otherwise, use the mean bias)
			bias_daily.loc[{'date': date}] = xr.where(N_daily.loc[{'date': date}] > n_f_bias, DS_date.bias.isel(time=DS_date.date==date)[0], bias_mean)

			# for each frequency (and each date), compute offset, slope, R:
			for freq, DS_freq_date in DS_date.groupby('freq'):

				# only use values where tb_used is True:
				x_fit = DS_freq_date.tb_mean_cs.values
				y_fit = DS_freq_date.tb_sim_cs.values
				mask = np.isfinite(x_fit + y_fit)

				if mask.sum() < 2: # regression requires >= 2 points
					continue
				x_fit = x_fit[mask]
				y_fit = y_fit[mask]

				# linear fit of x and y: (yields the same as the least squares approach performed for the plot)
				slope, offset = np.polyfit(x_fit, y_fit, 1)

				# save slope, offset and corr coeff:
				offsets_daily.loc[{'date': date, 'freq': freq}] = offset
				slopes_daily.loc[{'date': date, 'freq': freq}] = slope
				R_daily.loc[{'date': date, 'freq': freq}] = np.corrcoef(x_fit, y_fit)[0,1]


			# save the data in a dataset:
			STAT_DS = xr.Dataset({'bias':		(['date', 'freq'], np.reshape(bias_daily.loc[{'date': date}].values, (1,self.n_freq)),
												{'long_name': "Merged from daily_bias and mean_bias",
												'units': "K"}),
									'slope':	(['date', 'freq'], np.reshape(xr.where(R_daily.loc[{'date': date}] > 0.9, slopes_daily.loc[{'date': date}], 1.0).values, (1, self.n_freq)),
												{'long_name': "Slope of linear fit between simulated and measured TBs for high correlation",
												'units': ""}),
									'offset':	(['date', 'freq'], np.reshape(xr.where(R_daily.loc[{'date': date}] > 0.9, offsets_daily.loc[{'date': date}], -bias_daily.loc[{'date': date}]).values, (1, self.n_freq)),
												{'long_name': "Offset of linear fit between simulated and measured TBs for high correlation",
												'units': "K"}),
									'mean_bias': (['freq'], bias_mean.values,
												{'long_name': "Mean bias for the entire campaign",
												'units': "K"}),
									'daily_bias':(['date', 'freq'], np.reshape(DS_date.bias.isel(time=DS_date.date==date)[0].values, (1, self.n_freq)),
												{'long_name': "Daily bias",
												'units': "K"}),
									'daily_slope':	(['date', 'freq'], np.reshape(slopes_daily.loc[{'date': date}].values, (1, self.n_freq)),
												{'long_name': "Unfiltered slope of linear fit between simulated and measured TBs",
												'units': ""}),
									'daily_offset':	(['date', 'freq'], np.reshape(offsets_daily.loc[{'date': date}].values, (1, self.n_freq)),
												{'long_name': "Unfiltered offset of linear fit between simulated and measured TBs",
												'units': "K"}),
									'daily_R':	(['date', 'freq'], np.reshape(R_daily.loc[{'date': date}].values, (1, self.n_freq)),
												{'long_name': "Pearson correlation coefficient between observed and simulated TBs"}),
									'N':		(['date', 'freq'], np.reshape(N_daily.loc[{'date': date}].values, (1, self.n_freq)),
												{'long_name': "Number of sondes available for bias computation",
												'units': ""}),
								}, 
								coords={'date': (['date'], np.array([date])),
										'freq': (['freq'], TB_stat_DS.freq.values, TB_stat_DS.freq.attrs)})


			# set attributes:
			bias_description = (f"Merged from daily_bias and mean_bias. mean_bias is used if less than {n_f_bias} clear sky " +
								"dropsondes were detected. Subtract this number from TB measurements to apply offset " +
								"correction without linear correction (corrected_tb = tb - bias). ")
			lin_fit_descript = ("Use slope and offset to apply a correction that is either linear or offset only (i.e., slope==1.): " +
								"corrected_tb = slope * tb + offset . True linear correction is provided if correlation " +
								"between observed and synthetic BT is high (R > 0.9).")
			STAT_DS['bias'].attrs['description'] = bias_description
			STAT_DS['slope'].attrs['description'] = lin_fit_descript
			STAT_DS['offset'].attrs['description'] = lin_fit_descript
			STAT_DS['date'].attrs = TB_stat_DS.date.attrs


			# Global attributes:
			STAT_DS.attrs['title'] = ("HALO-(AC)3 HALO brightness temperature (TB) correction derived from " +
										"comparisons between PAMTRA-simulated dropsondes and HAMP microwave radiometer observation")
			STAT_DS.attrs['author'] = ("Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, " +
											"University of Cologne, Cologne, Germany")
			STAT_DS.attrs['python_version'] = f"python version: {sys.version}"
			STAT_DS.attrs['python_packages'] = (f"numpy: {np.__version__}, xarray: {xr.__version__}, " +
													f"pandas: {pd.__version__}, matplotlib: {mpl.__version__}")
			datetime_utc = dt.datetime.utcnow()
			STAT_DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

			# export to file:
			filename = f"HALO-AC3_HALO_HAMP_TB_offset_correction_{date}.nc"
			output_filename = self.path_cssc_output + filename
			STAT_DS.to_netcdf(output_filename, mode='w', format='NETCDF4')
			print(f"Saved TB offset correction data to '{output_filename}'.")
			STAT_DS.close()


	def visualise_TB_comparison(self, TB_stat_DS):

		"""
		Creates a scatter plot from the TB comparison dataset to compare simualted (with PAMTRA) and 
		observed TBs and show bias, root mean squared error, correlation coefficient, and number of
		used dropsondes for comparison.

		Parameters:
		-----------
		TB_stat_DS : xarray dataset
			Dataset containing the output of self.TB_comparison(), which is also saved to file.
		"""

		# font sizes and colours
		fs = 10
		fs_small = fs - 2
		fs_dwarf = fs - 4
		fs_hobbit = fs - 6
		marker_size = 9

		dt_fmt = mdates.DateFormatter("%H:%M")


		# scatter plot:
		f1, a1 = plt.subplots(ncols=7, nrows=4, figsize=(15,9), constrained_layout=True)

		a1 = a1.flatten()
		for k in range(self.n_freq, len(a1)): a1[k].axis('off')	# these subplots are not needed


		# axis limits:
		abs_lims = np.array([0.0, 310.0])

		for k in range(self.n_freq):

			# plot data:
			# x_error for frequencies: 0.5 Kelvin for K & V and 1 Kelvin for the higher frq:
			if TB_stat_DS['freq'][k] < 90:
				xerror = 0.5
			else:
				xerror = 1.0

			nonnan_idx = np.where(~np.isnan(TB_stat_DS['tb_mean_cs'].values[:,k]) & 
									~np.isnan(TB_stat_DS['tb_sim_cs'].values[:,k]))[0]

			a1[k].errorbar(TB_stat_DS['tb_sim_cs'].values[nonnan_idx,k], TB_stat_DS['tb_mean_cs'].values[nonnan_idx,k],
							xerr=xerror, yerr=TB_stat_DS['tb_std_cs'].values[nonnan_idx,k], color=(0,0,0),
							elinewidth=0.75, linestyle='none', ecolor=(0,0,0))


			# generate a linear fit with least squares approach: notes, p.2:
			# filter nan values:
			y_fit = TB_stat_DS['tb_mean_cs'].values[:,k][nonnan_idx]
			x_fit = TB_stat_DS['tb_sim_cs'].values[:,k][nonnan_idx]

			# there must be at least 2 measurements to create a linear fit:
			if (len(y_fit) > 1) and (len(x_fit) > 1):
				G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T
				m_fit = np.linalg.inv(G_fit.T @ G_fit) @ G_fit.T @ y_fit	# least squares solution
				a = m_fit[0]
				b = m_fit[1]

				ds_fit = a1[k].plot(abs_lims, a*abs_lims + b, color=(0,0,0), linewidth=0.75, label="Best fit")

			# plot a line for orientation which would represent a perfect fit:
			a1[k].plot(abs_lims, abs_lims, color=(0,0,0), linestyle='dashed', linewidth=0.75, alpha=0.5, 
						label="Theoretical perfect fit")

			# add aux info (texts, annotations)
			a1[k].text(0.02, 0.98, 
						f"bias = {TB_stat_DS['bias'][k].values:.2f} K\n " +
						f"rmse = {TB_stat_DS['rmse'][k].values:.2f} K\n " +
						f"R = {TB_stat_DS['R'][k].values:.3f} \n " +
						f"N = {int(TB_stat_DS['N'][k].values)}", 
						fontsize=fs_dwarf, ha='left', va='top', transform=a1[k].transAxes)

		# add aux info for non plotted axis:
		le_string_1 = break_str_into_lines("Error bars are the std. of all TB measurements from 10 s before " +
											"to 10 s after the drop.", n_max=30)
		le_string_2 = break_str_into_lines("The bias is calculated from all drops for clear " +
											"sky scenes (see output netCDF file).", n_max=30)
		le_string_3 = break_str_into_lines(r"bias = TB$_{MWR}$ - TB$_{PAMTRA}$", n_max=35)
		a1[-1].text(0.02, 0.96, le_string_1 + "\n \n" + le_string_2 + "\n \n" + le_string_3, fontsize=fs_dwarf,
					ha='left', va='top', transform=a1[-1].transAxes) # wrap=True


		# extra plots for legend:
		a1[-2].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=0.75, label="Best fit")
		a1[-2].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linestyle='dashed', linewidth=0.75, alpha=0.5, 
						label="Theoretical perfect fit")

		# legend, colorbar
		lh, ll = a1[-2].get_legend_handles_labels()
		a1[-2].legend(lh, ll, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs_dwarf,
				framealpha=0.5, title="YYYY-MM-DD, #sondes")

		
		# set further properties:
		for k, ax in enumerate(a1):
			if k < self.n_freq:

				limits = np.asarray([np.nanmin(np.concatenate((TB_stat_DS['tb_sim_cs'].values[:,k], 
									TB_stat_DS['tb_mean_cs'].values[:,k]), axis=0))-2,
									np.nanmax(np.concatenate((TB_stat_DS['tb_sim_cs'].values[:,k], 
									TB_stat_DS['tb_mean_cs'].values[:,k]), axis=0))+2])
				if np.isnan(limits[0]):
					limits = np.array([0.,1.])

				# set axis limits:
				ax.set_xlim(limits[0], limits[1])
				ax.set_ylim(limits[0], limits[1])

				# set axis ticks, ticklabels and tick parameters:
				ax.minorticks_on()
				ax.tick_params(axis='both', labelsize=fs_dwarf)

				# aspect ratio:
				ax.set_aspect('equal')

				# set labels:
				ax.set_title(f"{TB_stat_DS.freq.values[k]:.2f} GHz", fontsize=fs)
				if k % 7 == 0:
					ax.set_ylabel("TB$_{\mathrm{MWR}}$ (K)", fontsize=fs)
				if k >= 21:
					ax.set_xlabel("TB$_{\mathrm{PAMTRA}}$ (K)", fontsize=fs)

		f1.subplots_adjust(hspace=0.3, wspace=0.3)

		plot_name = f"HALO-AC3_HALO_HAMP_dropsondes_TB_comparison_scatter_plot_{TB_stat_DS['date'].values}.png"
		plot_outname = self.path_plot + plot_name
		f1.savefig(plot_outname, dpi=300, bbox_inches='tight')
		print(f"Saved scatter plot of TB comparison to '{plot_outname}'.")

		# clear memory
		plt.close()
		gc.collect()


	def visualise_TB_offsets(self):

		"""
		Visualise TB comparison for each day and present offset corrected measured
		TBs.
		"""

		# import output of get_TB_offsets and TB_comparison:
		files_comp = sorted(glob.glob(self.path_cssc_output + "*TB_comparison*.nc"))
		files_offsets = sorted(glob.glob(self.path_cssc_output + "*TB_offset_correction*.nc"))
		TB_stat_DS = xr.open_mfdataset(files_comp, combine='nested', concat_dim='time')
		STAT_DS = xr.open_mfdataset(files_offsets, combine='nested', concat_dim='date')

		self.n_freq = len(TB_stat_DS.freq)


		# font sizes and colours
		fs = 10
		fs_small = fs - 2
		fs_dwarf = fs - 4
		fs_hobbit = fs - 6
		marker_size = 9

		dt_fmt = mdates.DateFormatter("%H:%M")


		# scatter plot:
		f1, a1 = plt.subplots(ncols=7, nrows=4, figsize=(15,9), constrained_layout=True)

		a1 = a1.flatten()
		for k in range(self.n_freq, len(a1)): a1[k].axis('off')	# these subplots are not needed


		# axis limits:
		abs_lims = np.array([0.0, 310.0])

		for ax, freq in zip(a1, TB_stat_DS.freq):

			# limit datasets to current frequency and compute offset corrected TBs
			STAT_DS_freq = STAT_DS.sel(freq=freq)
			TB_stat_DS_freq = TB_stat_DS.sel(freq=freq)

			# quick-upsample STAT_DS_freq to the time axis of TB_stat_DS_freq:
			STAT_DS_freq_date = STAT_DS_freq.sel(date=TB_stat_DS_freq.date)


			# axis limits:
			limits = xr.DataArray([TB_stat_DS_freq.tb_sim.min()-5, TB_stat_DS_freq.tb_sim.max()+5], dims=['time'])

			# # plot uncorrected data:
			# ax.plot(TB_stat_DS_freq.tb_sim_cs, TB_stat_DS_freq.tb_mean_cs,
					# 's', markerfacecolor='none', linewidth=0.5)

			# plot data:
			tbs = STAT_DS_freq.offset + STAT_DS_freq.slope*limits
			ax.plot(tbs.transpose('time', 'date'), limits, linewidth=0.5)

			# plot linear corrected TBs:
			ax.plot(TB_stat_DS_freq.tb_sim_cs, STAT_DS_freq_date.slope * TB_stat_DS_freq.tb_mean_cs + STAT_DS_freq_date.offset,
					'o', markerfacecolor='none', linewidth=0.5)

			# plot offset corrected TBs:
			ax.plot(TB_stat_DS_freq.tb_sim_cs, TB_stat_DS_freq.tb_mean_cs - STAT_DS_freq_date.bias, '*', linewidth=0.5)


			# set axis limits:
			ax.set_xlim(limits)
			ax.set_ylim(limits)



		# extra plots for legend:
		a1[-2].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=0.5, label="Linear fit of uncorrected data for each flight (colours)")
		a1[-2].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), marker='o', linestyle='none', markerfacecolor='none', linewidth=0.5, label="Linear correction")
		a1[-2].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), marker='*', linestyle='none', linewidth=0.5, label="Offset correction")

		# legend, colorbar
		lh, ll = a1[-2].get_legend_handles_labels()
		a1[-2].legend(lh, ll, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs_dwarf,
				framealpha=0.5)

		
		# set further properties:
		for k, ax in enumerate(a1):
			if k < self.n_freq:

				# set axis ticks, ticklabels and tick parameters:
				ax.minorticks_on()
				ax.tick_params(axis='both', labelsize=fs_dwarf)

				# aspect ratio:
				ax.set_aspect('equal')

				# set labels:
				ax.set_title(f"{TB_stat_DS.freq.values[k]:.2f} GHz", fontsize=fs)
				if k % 7 == 0:
					ax.set_ylabel("TB$_{\mathrm{MWR}}$ (K)", fontsize=fs)
				if k >= 21:
					ax.set_xlabel("TB$_{\mathrm{PAMTRA}}$ (K)", fontsize=fs)

		f1.subplots_adjust(hspace=0.3, wspace=0.3)

		plot_name = f"HALO-AC3_HALO_HAMP_dropsondes_TB_corrected_comparison_scatter_plot.png"
		plot_outname = self.path_plot + plot_name
		f1.savefig(plot_outname, dpi=300, bbox_inches='tight')
		print(f"Saved scatter plot of corrected TB comparison to '{plot_outname}'.")

		# clear memory
		plt.close()
		gc.collect()

		TB_stat_DS.close()
		STAT_DS.close()
