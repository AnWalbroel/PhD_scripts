import numpy as np
import datetime as dt
import pdb
import sys
import netCDF4 as nc
import xarray as xr

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')    # so that functions and modules from MOSAiC can be used
from data_tools import *


def import_HAMP_unified_product(
	filename,
	cut_low_altitude=True):

	"""
	Importing the unified HALO HAMP data (microwave radiometer, radar) (from EUREC4A campaign) created by
	Heike Konow (heike.konow<at>uni-hamburg.de).

	Parameters:
	-----------
	filename : str
		Path and filename of HAMP data.
	cut_low_altitude : bool
		If True, time steps with altitude below 6000 m are removed. If False,
		this won't be done. Default: True
	"""

	data_DS = xr.open_dataset(filename)

	if 'time' in data_DS:
		# convert time and remove nasty after decimal point values:
		data_DS['time_npdt'] = data_DS.time				# to also keep the numpy datetime64
		data_DS['time'] = np.rint(numpydatetime64_to_epochtime(data_DS.time.values)).astype(float)

		# make sure that time sampling is 1 second:
		assert np.all(np.diff(data_DS.time.values) == 1.0)

	# cut time steps of low altitudes:
	if cut_low_altitude and 'altitude' in data_DS:
		low_alt_mask = np.where(data_DS.altitude >= 6000)[0]

		data_DS = data_DS.isel(time=low_alt_mask)

	return data_DS
