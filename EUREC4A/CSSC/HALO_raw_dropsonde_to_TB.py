from __future__ import print_function, division
import numpy as np
import os
import datetime
import glob
from general_importer import readNCraw_V01, import_GHRSST, import_BAHAMAS_unified		# "General Importer reporting for duty"
import pdb
import matplotlib.pyplot as plt
import multiprocessing



def run_HALO_raw_dropsonde_to_TB(
	path_halo_dropsonde,
	path_sst_data,
	pam_out_path,
	obs_height='BAHAMAS',
	path_BAH_data=None,
):
	"""Run forward simulation of TB using interpolated dropsondes.

	Parameters
	----------
	path_halo_dropsonde : str
		Path of interpolated Dropsonde netCDF files.
	path_sst_data : str
		Path of SST data netCDF files.
	pam_out_path : str
		Output path
	obs_height : number or str
		If a number is given use this as assumed aircraft altitude.
		If 'BAHAMAS' is given get altitude from BAHAMAS files.
			BAHAMAS files have to be in unified netCDF format and `path_BAH_data' has to be set.
	path_BAH_data : str, optional
		Path of BAHAMAS data in unified netCDF files. Required if obs_height == 'BAHAMAS'.
	"""

	if 'PAMTRA_DATADIR' not in os.environ:
		os.environ['PAMTRA_DATADIR'] = "" # actual path is not required, but the variable has to be defined.
	import pyPamtra

	# Check if the PAMTRA output path exists:
	pam_out_path_dir = os.path.dirname(pam_out_path)
	if not os.path.exists(pam_out_path_dir):
		os.makedirs(pam_out_path_dir)

	HALO_sondes_NC = sorted(glob.glob(path_halo_dropsonde + "*v01.nc"))
	SST_files_NC = sorted(glob.glob(path_sst_data + "*.nc.nc4"))

	if isinstance(obs_height, str):
		if obs_height == 'BAHAMAS':
			if not isinstance(path_BAH_data, str):
				raise ValueError("path_BAH_data is required as string argument when obs_height == 'BAHAMAS'")
			BAH_files_NC = sorted(glob.glob(path_BAH_data + "bahamas*.nc"))
			if len(BAH_files_NC) == 0:
				raise RuntimeError("Could not find any BAHAMAS data in `%s'"  % (path_BAH_data + "bahamas*.nc"))
		else:
			raise ValueError("Unknown obs_height `%s'" % obs_height)
	else:
		BAH_files_NC = []
		obs_height_value = np.asarray(obs_height).flatten() # we need a 1D array, which is used for all dropsondes.

	for filename_in in HALO_sondes_NC:
		print('import', filename_in)
		# Import sonde data:
		# # filename_in = HALO_sondes_NC[0]
		sonde_dict = readNCraw_V01(filename_in)

		# Import GHRSST CMC data:
		dropsonde_date = datetime.datetime.utcfromtimestamp(int(sonde_dict['launch_time'])).strftime("%Y%m%d")
		sst_filename = [sst_file for idx, sst_file in enumerate(SST_files_NC) if dropsonde_date in sst_file]
		sst_keys = ['time', 'lat', 'lon', 'analysed_sst', 'analysis_error']
		sst_dict = import_GHRSST(sst_filename[0], sst_keys)

		if BAH_files_NC:
			# Import altitude and time data from BAHAMAS:
			bah_filename = [bah_file for bah_file in BAH_files_NC if dropsonde_date in bah_file]
			if len(bah_filename) == 0:
				raise RuntimeError("Could not find any BAHAMAS data for date `%s'" % dropsonde_date)
			bah_keys = ['time', 'altitude']
			bah_dict = import_BAHAMAS_unified(bah_filename[0], bah_keys)
			bah_dict['time'] = np.rint(bah_dict['time']).astype(float)		# must be done to avoid small fractions of seconds


		n_alt = len(sonde_dict['Z'])		# number of height levels

		# find the sonde launches that produced too many nan values so that cannot run: use the RH, T, P for that:
		if not (np.all([~np.isnan(sonde_dict['T']), ~np.isnan(sonde_dict['P']), ~np.isnan(sonde_dict['RH'])])):
			print('WARNING, NaN in T, P, or RH. Skip %s' % filename_in)
			print('    NaN-counts: %d, %d, %d' % (
				np.isnan(sonde_dict['T']).sum(),
				np.isnan(sonde_dict['P']).sum(),
				np.isnan(sonde_dict['RH']).sum()
			))
			continue

		if np.any(np.isnan(sonde_dict['Z'])): # sometimes, even Z can contain nan, when not using BAHAMAS
			print('WARNING, NaN in Z. Skip %s' % filename_in)
			print('    NaN-count: %d' % np.isnan(sonde_dict['Z']).sum())
			continue

		if np.isnan(sonde_dict['u_wind'][1] + sonde_dict['v_wind'][1]):
			print('WARNING, NaN in u_wind[1] or v_wind[1], Skip %s' % filename_in)
			print('    u_NaN, v_NaN?: %d, %d' % (
				np.isnan(sonde_dict['u_wind'][1]),
				np.isnan(sonde_dict['v_wind'][1]),
			))
			continue

		# assert np.all(~np.isnan(sonde_dict['RH']))


		# HAMP FREQUENCIES:
		frq = [22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,31.4000,50.3000,51.7600,52.8000,53.7500,54.9400,56.6600,58.0000,90.0000,110.250,114.550,116.450,117.350,120.150,121.050,122.950,127.250,170.810,175.810,178.310,179.810,180.810,181.810,182.710,183.910,184.810,185.810,186.810,188.310,190.810,195.810]

		# create pamtra object; change settings:
		pam = pyPamtra.pyPamtra()

		pam.nmlSet['hydro_adaptive_grid'] = True
		pam.nmlSet['add_obs_height_to_layer'] = False		# adds observation layer height to simulation height vector
		pam.nmlSet['passive'] = True						# passive simulation
		pam.nmlSet['active'] = False						# False: no radar simulation
		# pam.nmlSet['gas_mod'] = 'L93'						# default: 'R98'

		pamData = dict()
		shape2d = [1, 1]

		# use highest non nan values of sonde for location information:
		if ~np.isnan(sonde_dict['reference_lon']):
			reflon = sonde_dict['reference_lon']
		else:
			reflon = sonde_dict['lon'][~np.isnan(sonde_dict['lon'])][-1]

		if ~np.isnan(sonde_dict['reference_lat']):
			reflat = sonde_dict['reference_lat']
		else:
			reflat = sonde_dict['lat'][~np.isnan(sonde_dict['lat'])][-1]

		pamData['lon'] = np.broadcast_to(reflon, shape2d)
		pamData['lat'] = np.broadcast_to(reflat, shape2d)
		pamData['timestamp'] = np.broadcast_to(sonde_dict['launch_time'], shape2d)

		# to get the obs_height: average BAHAMAS altitude over +/- 10 seconds around launch_time:
		# find time index of the sonde launches:
		if isinstance(obs_height, str) and obs_height == 'BAHAMAS':
			bah_launch_idx = np.asarray([np.argwhere(bah_dict['time'] == pamData['timestamp'][0])]).flatten()		# had some dimensions too many -> flattened
			drop_alt = np.floor(np.asarray([np.mean(bah_dict['altitude'][i-10:i+10]) for i in bah_launch_idx])/100)*100
			obs_height_value = drop_alt

		print(dropsonde_date + "\n")

		# surface type & reflectivity:
		pamData['sfc_type'] = np.zeros(shape2d)			# 0: ocean, 1: land
		pamData['sfc_refl'] = np.chararray(shape2d)
		pamData['sfc_refl'][:] = 'F'
		pamData['sfc_refl'][pamData['sfc_type'] == 1] = 'S'

		pamData['obs_height'] = np.broadcast_to(obs_height_value, shape2d + [len(obs_height_value), ]) # must be changed to the actual top of soundings (or mwr altitude / bahamas altitude)

		# meteorolog. surface information:
		# to find the SST: use the designated lat,lon in pamData to find the closest entry in the GHRSST dataset:
		dlat = np.asarray([sst_dict['lat'] - pamData['lat'][0,0]])
		dlon = np.asarray([sst_dict['lon'] - pamData['lon'][0,0]])	# for each sonde, for each sst entry
		distance_lat_squared = (2*np.pi*6371000*dlat/360)**2
		distance_lon_squared = (2*np.pi*6371000*np.cos(2*np.pi*pamData['lat'][0,0]/360)*dlon/360)**2
		i_lat = np.argmin(distance_lat_squared)	# contains index of sst_dict['lat'] which had a min distance to pamData[lat] for each sonde
		i_lon = np.argmin(distance_lon_squared)	# contains index of sst_dict['lat'] which had a min distance to pamData[lat] for each sonde

		sst = sst_dict['SST'][0,i_lat,i_lon]		# [time, lat, lon]


		pamData['groundtemp'] = np.broadcast_to(sst, shape2d)
		pamData['wind10u'] = np.broadcast_to(sonde_dict['u_wind'][1], shape2d)		# = 1 because sfc would be 0 m; index 1 is 10 m
		pamData['wind10v'] = np.broadcast_to(sonde_dict['v_wind'][1], shape2d)		# = 1 because sfc would be 0 m; index 1 is 10 m

		# 3d variables:
		shape3d = shape2d + [n_alt]
		pamData['hgt_lev'] = np.broadcast_to(sonde_dict['Z'], shape3d)
		pamData['temp_lev'] = np.broadcast_to(sonde_dict['T'][:], shape3d)
		pamData['press_lev'] = np.broadcast_to(sonde_dict['P'][:], shape3d)
		pamData['relhum_lev'] = np.broadcast_to(sonde_dict['RH'][:], shape3d)

		# 4d variables: hydrometeors:
		shape4d = [1, 1, n_alt-1, 5]			# potentially 5 hydrometeor classes with this setting
		pamData['hydro_q'] = np.zeros(shape4d)
		pamData['hydro_q'][...,0] = 0# CLOUD
		pamData['hydro_q'][...,1] = 0# ICE
		pamData['hydro_q'][...,2] = 0# RAIN
		pamData['hydro_q'][...,3] = 0# SNOW
		pamData['hydro_q'][...,4] = 0# GRAUPEL


		# descriptorfile must be included. otherwise, pam.p.nhydro would be 0 which is not permitted. (OLD DESCRIPTOR FILE)
		descriptorFile = np.array([
			  #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
			   ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
			   ('iwc_q', -99.0, -1, -99.0, 130.0, 3.0, 0.684, 2.0, 3, 1, 'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, 'mie-sphere', 'heymsfield10_particles', -99.0),
			   ('rwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 50, 'exp', -99.0, -99.0, 8000000.0, -99.0, 0.00012, 0.006, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
			   ('swc_q', -99.0, -1, -99.0, 0.038, 2.0, 0.3971, 1.88, 3, 50, 'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, 5.1e-11, 0.02, 'mie-sphere', 'heymsfield10_particles', -99.0),
			   ('gwc_q', -99.0, -1, -99.0, 169.6, 3.1, -99.0, -99.0, 3, 50, 'exp', -99.0, -99.0, 4000000.0, -99.0, 1e-10, 0.01, 'mie-sphere', 'khvorostyanov01_spheres', -99.0)],
			  dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
			  )
		for hyd in descriptorFile: pam.df.addHydrometeor(hyd)


		# Create pamtra profile and go:
		pam.createProfile(**pamData)
		print("Starting PAMTRA on '" + filename_in + "':")
		# pam.runPamtra(frq)

		n_cpus = int(multiprocessing.cpu_count()/2)		# half the number of available CPUs
		pam.runParallelPamtra(frq, pp_deltaX=0, pp_deltaY=0, pp_deltaF=1, pp_local_workers=n_cpus)

		# save output:
		filename_out = os.path.join(pam_out_path, "pamtra_" + os.path.basename(filename_in))
		pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)


		# # # # # Save the dropsonde launch number:
		# # # # sonde_number_filename = "/work/walbroel/data/" + "sonde_number_" + filename_in[-15:-3]
		# # # # np.save(sonde_number_filename, whichsonde)