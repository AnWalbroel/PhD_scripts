import numpy as np
import xarray as xr
import geopy
import geopy.distance
from scipy.spatial import KDTree

import os
import glob
import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *
from import_data import import_iasi_nc, import_PS_mastertrack

import pdb


"""
	Script to prepare IASI data for Polarstern track. Import IASI and Polarstern track data, cut
	the data and export IASI data to a new, smaller file. Call this script via 
	"python3 manage_iasi.py" or append a digit between 0 and 19 for the IASI subfolders, i.e.,
	"python3 manage_iasi.py 14".
	- import data (iasi, polarstern track)
	- find spatio-temporal overlap
	- export IASI data to file
"""


# paths:
path_data = {	'iasi': "/mnt/f/heavy_data/IASI/",		# subfolders may exist
				'ps_track': "/mnt/f/heavy_data/polarstern_track/"}
path_output = "/mnt/f/heavy_data/IASI_mosaic/"

# additional settings:
set_dict = {'max_dist': 50.0}			# max distance in km

path_output_dir = os.path.dirname(path_output)
if not os.path.exists(path_output_dir):
	os.makedirs(path_output_dir)


# import polarstern track:
files = sorted(glob.glob(path_data['ps_track'] + "PS122_3_link-to-mastertrack_V2.nc"))
PS_DS = import_PS_mastertrack(files, return_DS=True)

# import iasi data: consider each subfolder step by step:
# chose subfolder of IASI data if given in the system arguments:
subfolders = sorted(glob.glob(path_data['iasi'] + "*"))
if len(sys.argv) == 2:
	subfolders = subfolders[int(sys.argv[1]):int(sys.argv[1])+1]


for idx_folder, subfolder in enumerate(subfolders):

	# loop through subfolders
	files = sorted(glob.glob(subfolder + "/*.nc"))
	IASI_DS = import_iasi_nc(files)

	# adapt PS_DS time range according to IASI swath times:
	min_time_iasi = IASI_DS.record_start_time.values.min()
	max_time_iasi = IASI_DS.record_stop_time.values.max()
	PS_DS = PS_DS.sel(time=slice(min_time_iasi-np.timedelta64(7200,"s"), max_time_iasi+np.timedelta64(7200,"s")))


	# find overlap of IASI and Polarstern for each track data point: loop over it:
	iasi_lon_f = IASI_DS.lon.values.ravel()
	iasi_lat_f = IASI_DS.lat.values.ravel()
	iasi_r_start_f = np.repeat(np.reshape(IASI_DS.record_start_time.values, (len(IASI_DS.along_track),1)), len(IASI_DS.across_track), axis=1).ravel()
	iasi_r_stop_f = np.repeat(np.reshape(IASI_DS.record_stop_time.values, (len(IASI_DS.along_track),1)), len(IASI_DS.across_track), axis=1).ravel()
	len_iasi_f = len(iasi_lon_f)		# length of flattened IASI data

	# first, filter for latitudes close to those seen in the Polarstern track: this reduces the computation time of the loop below
	lat_mask = np.full((len_iasi_f,), False)
	lat_mask_idx = np.where((iasi_lat_f >= (np.floor(PS_DS.Latitude.values.min()) - 1.0)) & (iasi_lat_f <= (np.ceil(PS_DS.Latitude.values.max()) + 1.0)))[0]
	lat_mask[lat_mask_idx] = True		# where this array is True, latitudes are within the PS track expectations
	n_lat_mask_idx = len(lat_mask_idx)

	# reduce flattened IASI data:
	iasi_lon_f_masked = iasi_lon_f[lat_mask]
	iasi_lat_f_masked = iasi_lat_f[lat_mask]

	# create variables and arrays to save IASI data that fulfills the spatio-temporal overlap constraints
	list_idx_masked = list()
	n_time_ps_track = len(PS_DS.time.values)
	iasi_ps_keys = ['lat', 'lon', 'iwv', 'record_start_time', 'record_stop_time', 'degraded_ins_MDR',
					'degraded_proc_MDR', 'flag_amsubad', 'flag_avhrrbad', 'flag_fgcheck', 'flag_iasibad',
					'flag_itconv', 'flag_landsea', 'flag_mhsbad', 'flag_retcheck']

	# create empty arrays:
	iasi_ps_dict = dict()
	for key in iasi_ps_keys:

		# save mean (and std) of the following variables:
		if key in ['lat', 'lon', 'iwv']:
			iasi_ps_dict[key + '_mean'] = np.full((n_time_ps_track,), np.nan)
			if key == 'iwv': iasi_ps_dict[key + '_std'] = np.full((n_time_ps_track,), np.nan)

		elif key == 'record_start_time':
			iasi_ps_dict[key + '_min'] = np.zeros((n_time_ps_track,))
		elif key == 'record_stop_time':
			iasi_ps_dict[key + '_max'] = np.zeros((n_time_ps_track,))
		else:
			iasi_ps_dict[key] = np.full((n_time_ps_track,100), np.nan)

	# loop through PS track time:
	for k, ps_time in enumerate(PS_DS.time.values):

		if k%10 == 0: print(f"{k} of {n_time_ps_track}")
		if k == 100: print("Yes... I'm still going. Be patient.")
		if k == 200: print("just a little longer....")
		if k == 300: print("Memory and CPU usage still good?")

		# first, finde indices of IASI_DS that are within the spatial distance defined
		# in the settings:
		circ_centre = [PS_DS.Latitude.values[k], PS_DS.Longitude.values[k]]

		# loop through lat-masked IASI coordinates to find where distance to Polarstern is less than the threshold:
		iasi_ps_dist = np.ones((n_lat_mask_idx,))*(-1.0)
		for kk in range(n_lat_mask_idx):
			iasi_ps_dist[kk] = geopy.distance.distance((iasi_lat_f_masked[kk], iasi_lon_f_masked[kk]), (circ_centre[0], circ_centre[1])).km

		distance_mask = iasi_ps_dist <= set_dict['max_dist']	# True for data fulfilling the distance criterion
		iasi_idx_masked = lat_mask_idx[distance_mask]			# yields the indices where the flattened IASI data fulfills both masks

		# check for temporal overlap:
		iasi_r_start_f_masked = iasi_r_start_f[iasi_idx_masked]
		iasi_r_stop_f_masked = iasi_r_stop_f[iasi_idx_masked]
		iasi_mean_record_time_masked = iasi_r_start_f_masked + 0.5*(iasi_r_stop_f_masked - iasi_r_start_f_masked)

		iasi_time_space_mask = iasi_idx_masked[np.where(np.abs(iasi_mean_record_time_masked - ps_time) <= np.timedelta64(3000, "s"))[0]]
		n_iasi_left = len(iasi_time_space_mask)


		# save data to the dictionary id there is data to be saved:
		if n_iasi_left > 0:
			for key in iasi_ps_keys:
				if key in ['lat', 'lon', 'iwv']:
					iasi_ps_dict[key + "_mean"][k] = np.nanmean(IASI_DS[key].values.ravel()[iasi_time_space_mask])

					if key == 'iwv': iasi_ps_dict[key + "_std"][k] = np.nanstd(IASI_DS[key].values.ravel()[iasi_time_space_mask])

				elif key == 'record_start_time':
					iasi_ps_dict[key + '_min'][k] = np.nanmin(iasi_r_start_f[iasi_time_space_mask])
				elif key == 'record_stop_time':
					iasi_ps_dict[key + '_max'][k] = np.nanmax(iasi_r_stop_f[iasi_time_space_mask])

				elif key in ['degraded_ins_MDR', 'degraded_proc_MDR']:
					iasi_ps_dict[key][k,:n_iasi_left] = np.repeat(np.reshape(IASI_DS[key].values, (len(IASI_DS.along_track),1)), len(IASI_DS.across_track), axis=1).ravel()[iasi_time_space_mask]

				else:
					iasi_ps_dict[key][k,:n_iasi_left] = IASI_DS[key].values.ravel()[iasi_time_space_mask]


	# save data dict to xarray dataset, then to netCDF:
	IASI_PS_DS = xr.Dataset(coords={'time': (['time'], PS_DS.time.values)})

	for key in iasi_ps_keys:

		if key in ['lat', 'lon', 'iwv']:
			IASI_PS_DS[key + '_mean'] = xr.DataArray(iasi_ps_dict[key + '_mean'], dims=['time'], attrs=IASI_DS[key].attrs)
			IASI_PS_DS[key + '_mean'].attrs['long_name'] = "mean " + IASI_PS_DS[key + '_mean'].attrs['long_name']
			IASI_PS_DS[key + '_mean'].attrs['standard_name'] = "mean_" + IASI_PS_DS[key + '_mean'].attrs['standard_name']

			if key == 'iwv':
				IASI_PS_DS[key + '_std'] = xr.DataArray(iasi_ps_dict[key + '_std'], dims=['time'], attrs=IASI_DS[key].attrs)
				IASI_PS_DS[key + '_std'].attrs['long_name'] = "std " + IASI_PS_DS[key + '_std'].attrs['long_name']
				IASI_PS_DS[key + '_std'].attrs['standard_name'] = "std_" + IASI_PS_DS[key + '_std'].attrs['standard_name']

		elif key == 'record_start_time':
			IASI_PS_DS[key + '_min'] = xr.DataArray(iasi_ps_dict[key + '_min'].astype(np.float64), dims=['time'], attrs=IASI_DS[key].attrs)
		elif key == 'record_stop_time':
			IASI_PS_DS[key + '_max'] = xr.DataArray(iasi_ps_dict[key + '_max'].astype(np.float64), dims=['time'], attrs=IASI_DS[key].attrs)

		else:
			IASI_PS_DS[key] = xr.DataArray(iasi_ps_dict[key], dims=['time', 'n_points'], attrs=IASI_DS[key].attrs)

	# time encoding
	IASI_PS_DS['time'] = PS_DS.time.values.astype("datetime64[s]").astype(np.float64)
	IASI_PS_DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
	IASI_PS_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	IASI_PS_DS['time'].encoding['dtype'] = 'double'

	IASI_PS_DS.to_netcdf(path_output + f"MOSAiC_IASI_Polarstern_overlap_{int(sys.argv[1]):02}.nc", mode='w', format='NETCDF4')
	IASI_PS_DS.close()

	# clear memory:
	del IASI_PS_DS