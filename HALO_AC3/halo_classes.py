import numpy as np
import datetime as dt
import xarray as xr
import pdb
import sys
import glob

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')	# so that functions and modules from MOSAiC can be used
from data_tools import *


# Dictionary translating Research Flight numbers and dates:
RF_dict = {
			'20220225': "RF00",
			"20220311": "RF01",
			"20220312": "RF02",
			"20220313": "RF03",
			"20220314": "RF04",
			"20220315": "RF05",
			"20220316": "RF06",
			"20220320": "RF07",
			"20220321": "RF08",
			"20220328": "RF09",
			"20220329": "RF10",
			"20220330": "RF11",
			"20220401": "RF12",
			"20220404": "RF13",
			"20220407": "RF14",
			"20220408": "RF15",
			"20220410": "RF16",
			"20220411": "RF17",
			"20220412": "RF18",
			}


class MWR:
	"""
		RAW microwave radiometer onboard HALO (part of HAMP). Time will be given in seconds since
		2017-01-01 00:00:00 UTC.

		For initialisation we need:
		path : str
			Path of raw HALO HAMP-MWR data. The path must simply link to HALO and then contain 
			the datefurther folders that then link to the HAMP MWR receivers (KV, 11990, 183): Example:
			path = "/data/obs/campaigns/eurec4a/HALO/" -> contains "./20020205/radiometer/" +
			["KV/", "11990/", "183/"].
		which_date : str
			Marks the flight day that shall be imported. To be specified in yyyymmdd (e.g. 20200213)!
		version : str
			Version of the HAMP MWR data. Options available: 'raw', 'halo_ac3_raw', 'synthetic_dropsonde'

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, which_date, version='halo_ac3_raw', **kwargs):

		assert len(which_date) == 8
		reftime = "2017-01-01 00:00:00"		# in UTC

		# init attributes:
		self.freq = dict()
		self.time = dict()
		self.time_npdt = dict()
		self.TB = dict()
		self.flag = dict()

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = dict()

		if version == 'raw':
			path_date = path + f"{which_date}/radiometer/"

			# import data: Identify receivers of the current day:
			self.avail = {'KV': False, '11990': False, '183': False}
			files = dict()
			
			for key in self.avail.keys():
				files[key] = sorted(glob.glob(path_date + key + f"/{which_date[2:]}[0-9][0-9].BRT.NC"))

				if len(files[key]) > 0:
					self.avail[key] = True

					# import data: cycle through receivers and import when possible:
					DS = xr.open_mfdataset(files[key][1:], concat_dim='time', combine='nested')

					# reduce unwanted dimensions:
					DS['frequencies'] = DS.frequencies[0,:]
					
					# Unify variable names by defining class attributes:
					self.freq[key] = DS.frequencies.values			# in GHz
					self.time_npdt[key] = DS.time.values			# in numpy datetime64
					self.time[key] = numpydatetime64_to_reftime(DS.time.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC
					self.TB[key] = DS.TBs.values					# in K, time x freq
					self.flag[key] = DS.rain_flag.values			# rain flag, no units

					if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
						self.DS[key] = DS

				else:
					print(f"No {key} data on {which_date} from HAMP MWR.\n")

		elif version == 'halo_ac3_raw':

			# import data: Identify receivers of the current day:
			self.avail = {'KV': False, '11990': False, '183': False}
			files = dict()
			
			for key in self.avail.keys():
				files[key] = sorted(glob.glob(path + f"hamp_{key.lower()}/HALO-AC3_HALO_hamp_*_{which_date}*.nc"))

				if len(files[key]) > 0:
					self.avail[key] = True

					# import data: cycle through receivers and import when possible:
					DS = xr.open_dataset(files[key][0])
					
					# Unify variable names by defining class attributes:
					self.freq[key] = DS.Freq.values					# in GHz
					self.time_npdt[key] = DS.time.values			# in numpy datetime64
					self.time[key] = numpydatetime64_to_reftime(DS.time.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC
					self.TB[key] = DS.TBs.values					# in K, time x freq
					self.flag[key] = DS.RF.values			# rain flag, no units

					if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
						self.DS[key] = DS

				else:
					print(f"No {key} data on {which_date} from HAMP MWR.\n")

		elif version == 'synthetic_dropsonde':

			files = dict()

			# import data:
			key = 'dropsonde'
			files[key] = sorted(glob.glob(path + f"HALO-AC3_HALO_Dropsondes_{which_date}_{RF_dict[which_date]}/" +
								"*.nc"))

			if len(files[key]) > 0:

				# import data: cycle through receivers and import when possible:
				DS = xr.open_mfdataset(files[key], concat_dim='grid_x', combine='nested')

				# Post process PAMTRA simulations: reduce undesired dimensions and			
				# unify variable names by defining class attributes:
				self.freq[key] = DS.frequency.values			# in GHz
				self.time_npdt[key] = DS.datatime.values.flatten()	# in numpy datetime64
				self.time[key] = numpydatetime64_to_reftime(DS.datatime.values.flatten(), reftime) # in seconds since 2017-01-01 00:00:00 UTC
				self.TB[key] = DS.tb.values[:,0,0,0,:,:].mean(axis=-1)		# in K, time x freq

				# apply double side band average:
				self.TB[key], self.freq[key] = Fband_double_side_band_average(self.TB[key], self.freq[key])
				self.TB[key], self.freq[key] = Gband_double_side_band_average(self.TB[key], self.freq[key])

				if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
					self.DS[key] = DS

			else:
				print(f"No {key} data on {which_date} from HAMP MWR.\n")


class radar:
	"""
		Cloud radar onboard HALO (part of HAMP).

		For initialisation we need:
		path : str
			Path of unified HALO HAMP radar data.
		which_date : str
			Marks the flight day that shall be imported. To be specified in 
			yyyymmdd (e.g. 20200213)!
		version : str
			Specifies the data version. Valid option depends on the instrument.

		**kwargs:
		return_DS : bool
			If True, the imported xarray dataset will also be set as a class attribute.
	"""

	def __init__(self, path, which_date, version='raw', **kwargs):

		# import data: Identify correct data (version, dates):
		filename = sorted(glob.glob(path + "hamp_mira/" + f"HALO-AC3_HALO_hamp_mira_{which_date}*.nc"))
		if len(filename) == 0:
			raise RuntimeError(f"Could not find and import {filename}.")
		else:
			filename = filename[0]

		if version == 'raw':
			data_DS = xr.open_dataset(filename)
		else:
			raise RuntimeError("Other versions than 'raw' radar data have not yet been implemented.")
		
		# Unify variable names by defining class attributes:
		self.time = numpydatetime64_to_reftime(data_DS.time.values, "2017-01-01 00:00:00")	# in sec since 2017-01-01 00:00:00 UTC
		self.time_npdt = data_DS.time.values		# in numpy datetime64
		self.range = data_DS.range.values			# in m from aircraft on
		self.Z = data_DS.Zg.values					# radar reflectivity im mm^6 m^-3
		self.dBZ = 10*np.log10(self.Z)				# equivalent reflectivity factor in dBZ

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = data_DS


class BAHAMAS:
	"""
		BAHAMAS data from HALO for time axis (and eventually other stuff later). Time will be converted
		to 2017-01-01 00:00:00 UTC. 

		For initialisation we need:
		path : str
			Path where HALO BAHAMAS data is located.
		which_date : str
			Marks the flight day that shall be imported. To be specified in yyyymmdd (e.g. 20200213)!
		version : str
			Version of the BAHAMAS data. Options available: 'nc_raw', 'halo_ac3_raw'

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
			reftime = "2017-01-01 00:00:00"			# in UTC
			self.time_npdt = DS.TIME.values			# np.datetime64 array
			self.time = numpydatetime64_to_reftime(DS.TIME.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC

		elif version == 'halo_ac3_raw':
			# Identify correct time: /data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/HALO-AC3_HALO_BAHAMAS_20220313_RF03
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
			reftime = "2017-01-01 00:00:00"			# in UTC
			self.time_npdt = DS.TIME.values			# np.datetime64 array
			self.time = numpydatetime64_to_reftime(DS.TIME.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC

		if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
			self.DS = DS