from __future__ import print_function, division
import numpy as np
import urllib
import datetime as dt
import pandas as pd
import pdb
import os

# in python3, .urlretrieve() and .urlcleanup() are part of urllib.request,
# while in python2 they were directly under urllib.
try: # is it python3?
	request = urllib.request
	# however, documentation says:
	#   "The following functions and classes are ported from the Python 2 module urllib (as opposed to urllib2).
	#   They might become deprecated at some point in the future.""
except AttributeError:
	request = urllib # try python2 way



def download_sst_data(path_sst, lat_bound, lon_bound, start_date, end_date):

	############################################################################################
	# FUNCTIONS

	def better_ceil(input, digits):	# numpy's ceil and floor create nasty zeros or 9s after decimal point

		return np.round(input + 0.49999999*10**(-digits), digits)


	# this function is only used because I did not find anything useful to round OFF
	# to a random power of ten that doesn't create nasty after decimal point zeros or 9s
	def better_floor(input, digits):

		return np.round(input - 0.49999999*10**(-digits), digits)


	def lat_bound_to_slice(lat_bound):	# converts latitude boundaries to the OPENDAP slices for lat

		return [(lat_bound[0] + 90)*10, (lat_bound[1] + 90)*10]


	def lon_bound_to_slice(lon_bound):	# converts latitude boundaries to the OPENDAP slices for lat

		return [(lon_bound[0] + 180)*10, (lon_bound[1] + 180)*10]



	############################################################################################


	'''
	Get SST data from CMC0.1deg-CMC-L4-GLOB-v3.0 via OPENDAP tool. For this we need to select
	latitude and longitude boundaries as well as the required dates.
		path_sst = "/work/walbroel/data/sst_slice/"		# SST data will be saved here
		lat_bound = [20, 50.5]		# latitude boundaries in deg north (southern hemisphere with negative sign)
		lon_bound = [-40, 10]		# longitude boundaries in deg east (west of prime meridian: negative sign)
		start_date = "2020-03-06"	# start date in YYYY-MM-DD
		end_date = "2020-03-08"		# end date
	'''

	# Check the existence of the path where the files shall be saved to:
	path_sst_dir = os.path.dirname(path_sst)
	if not os.path.exists(path_sst_dir):
		os.makedirs(path_sst_dir)

	# convert dates to pandas datetime:
	start_date = pd.to_datetime(start_date, format="%Y-%m-%d")		# in YYYY-MM-DD
	end_date = pd.to_datetime(end_date, format="%Y-%m-%d")			# in YYYY-MM-DD

	daterange = pd.date_range(start_date, end_date, freq='D')		# daterange with days as increment


	# need to convert the lat and lon boundaries to the slices of the OPENDAP tool:
	# but first get the to a precision of 1 after decimal point:
	lat_bound = [better_floor(lat_bound[0], 1), better_ceil(lat_bound[1], 1)]
	lon_bound = [better_floor(lon_bound[0], 1), better_ceil(lon_bound[1], 1)]

	lat_slice = lat_bound_to_slice(lat_bound)
	lon_slice = lon_bound_to_slice(lon_bound)


	for dr in daterange:

		print(dr)

		# try to load a file:
		request.urlcleanup()		# clear the cache of previous urlretrieve calls

		# define some shortcuts:
		daynumber = dr.dayofyear	# day number of the specified year
		thisyear = str(dr.year)
		date_formatted = dr.strftime("%Y%m%d")
		lat_slice_formatted = "%d:1:%d"%(lat_slice[0], lat_slice[1])		# e.g. 450:1:900
		lon_slice_formatted = "%d:1:%d"%(lon_slice[0], lon_slice[1])		# e.g. 1140:1:1400
		daynumber = "%03d" % daynumber

		outfile_name = date_formatted + "120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc.nc4"
		to_be_retrieved = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/CMC/CMC0.1deg/v3/{!s}/{!s}/{!s}120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc.nc4?time%5B0:1:0%5D,lat%5B{!s}%5D,lon%5B{!s}%5D,analysed_sst%5B0:1:0%5D%5B{!s}%5D%5B{!s}%5D,analysis_error%5B0:1:0%5D%5B{!s}%5D%5B{!s}%5D,sea_ice_fraction%5B0:1:0%5D%5B{!s}%5D%5B{!s}%5D,mask%5B0:1:0%5D%5B{!s}%5D%5B{!s}%5D".format(thisyear, daynumber, date_formatted,
					lat_slice_formatted, lon_slice_formatted, lat_slice_formatted, lon_slice_formatted,
					lat_slice_formatted, lon_slice_formatted, lat_slice_formatted, lon_slice_formatted,
					lat_slice_formatted, lon_slice_formatted)


		try:
			request.urlretrieve(to_be_retrieved, path_sst + outfile_name)


		except:	# if it couldn't be downloaded continue with next day
			print("Could not retrieve '" + to_be_retrieved + "' from server.")
			continue
