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

	"""
	Run forward simulation of TB using interpolated dropsondes.

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


	def detect_liq_cloud(
		z,
		temp,
		rh,
		rh_thres=95,
		temp_thres=253.15):

		"""
		Cloud detection following Karstens et al. 1994.

		Parameters:
		-----------
		z : array of floats
			height grid in m.
		temp : array of floats
			Temperature on height grid in K.
		rh : array of floats
			Relative humidity on height grid (in %).
		rh_thres : float
			Relative humidity threshold above which a cloud is assumed. Default: 0.95
		temp_thres : float
			Temperature threshold below which only ice clouds are assumed. Default: 253.15
		"""

		n_height = len(z)

		# determine cloud boundaries with rel hum thresholds (on LAYERS):
		cloud_bound_ind = np.zeros((n_height,))
		for k in range(n_height-1):
			if ((rh[k+1] + rh[k])/2 > rh_thres) and ((temp[k+1] + temp[k])/2 > temp_thres):

				if cloud_bound_ind[k] == 2:
					cloud_bound_ind[k] = 3		# within cloud
				else:
					cloud_bound_ind[k] = 1		# LEVEL of cloud base
				cloud_bound_ind[k+1] = 2		# LEVEL of cloud top

		i_cloud = np.where(cloud_bound_ind != 0)[0]

		# determine cloud base and top arrays:
		z_top = -99.
		z_base = -99.
		z_cloud = -99.

		if len(i_cloud) > 0:
			z_cloud = z[i_cloud]
			i_base = np.where(cloud_bound_ind == 1)[0]
			i_top = np.where(cloud_bound_ind == 2)[0]

			if len(i_base) != len(i_top):
				raise RuntimeError("Number of cloud bases does not equal the number of cloud tops." +
									"Error in HALO_raw_dropsonde_to_TB.py.")

			z_top = z[i_top]
			z_base = z[i_base]

		return z_top, z_base, z_cloud, i_cloud


	def adiab_lwc(
		k,
		temp,
		pres,
		z):

		"""
		Computation of adiabatic liquic water content (LWC). Translated from the IDL script
		'adiab.pro'. Based on Karstens et al. 1994.

		Parameters:
		-----------
		k : int
			Height layer (!) index.
		temp : array of floats
			Temperature on cloud height grid in K.
		pres : array of floats
			Pressure on cloud height grid in Pa.
		z : array of floats
			Cloud height levels in m.
		"""

		CP = 1005.0			# specific heat capacity of air at const pressure
		G = 9.80616			# gravitational acceleration
		e0 = 610.78
		Rv = 462.0
		RW = 461.5
		RL = 287.05
		T0 = 273.15

		# Set actual cloud base temperature to the measured one. Initialise Liquid
		# Water Content (LWC). Compute adiabatic LWC by integration from cloud base to
		# level k.
		temp_cloud = temp[0]
		LWC = 0.
		for kk in range(1, k+1):
			dz = z[kk] - z[kk-1]

			# compute actual cloud temperature:
			L = 2.501e+06 - 2372.0 * (temp[kk] - T0)
			e_sat = e0*np.exp((L / (Rv*273.15)) * (temp[kk] - T0) / temp[kk])
			temp_virt = temp[kk] / (1 - 0.379 * (e_sat / pres[kk]))
			rho = pres[kk] / (temp_virt * 287.04)				# density of air inside the cloud
			rho_v = (18.016 * e_sat * 1.0) / (8314.3 * temp[kk])	# abs humidity
			mix_rat_s = rho_v / (rho - rho_v)					# mixing ratio of water vapour
			Lvap = (2500.8-2.372*(temp[kk]-T0)) * 1000.0
			dtps = (G / CP) * (1 + (Lvap * mix_rat_s / RL / temp[kk])) / (1 + (mix_rat_s * Lvap**2 / CP / RW / temp[kk]**2))
			temp_cloud = temp_cloud - dtps * dz

			# compute adiabatic LWC:
			e_sat = e0*np.exp((L / (Rv*273.15)) * (temp_cloud - T0) / temp_cloud)
			temp_virt = temp_cloud / (1 - 0.379 * (e_sat / pres[kk]))
			rho = pres[kk] / (temp_virt * 287.04)
			rho_v = (18.016 * e_sat * 1.0) / (8314.3 * temp_cloud)
			mix_rat_s = rho_v / (rho - rho_v)
			Lvap = (2500.8-2.372*(temp_cloud-T0)) * 1000.0

			dtps_cloud = (G / CP) * (1 + (Lvap * mix_rat_s / RL / temp_cloud)) / (1 + (mix_rat_s * Lvap**2 / CP / RW / temp_cloud**2))
			LWC += (rho * CP/Lvap * ((G / CP) - dtps_cloud) * dz)			# Karstens et al. 1994, (1)


		return LWC


	def mod_ad_lwc(
		temp,
		pres,
		z):

		"""
		Computation of modified adiabatic liquic water content (LWC) based on
		Karstens et al. 1994. Translated from the IDL script 'mod_ad.pro'.

		Parameters:
		-----------
		temp : array of floats
			Temperature on cloud height grid in K.
		pres : array of floats
			Pressure on cloud height grid in Pa.
		z : array of floats
			Cloud height levels in m.
		"""

		n_levels = len(z)
		lwc = np.zeros((n_levels-1,))		# will be on layers, not levels!

		thick = 0.	# cloud thickness
		for k in range(n_levels-1):
			dz = z[k+1] - z[k]
			thick += dz
			lwc[k] = adiab_lwc(k+1, temp, pres, z)
			lwc[k] = lwc[k]*(-0.144779*np.log(thick) + 1.239387)

		return lwc



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


		# compute LWP if wanted:
		sonde_dict['Z_top'], sonde_dict['Z_base'], sonde_dict['Z_cloud'], i_cloud = detect_liq_cloud(sonde_dict['Z'], sonde_dict['T'],
																													sonde_dict['RH'])

		# loop through all cloud layers:
		sonde_dict['Z_layer'] = (sonde_dict['Z'][1:] + sonde_dict['Z'][:-1]) / 2
		if len(i_cloud) == 0:
			sonde_dict['Z_top'] = np.array([])
			sonde_dict['Z_base'] = np.array([])
			sonde_dict['Z_cloud'] = np.array([])
			
		n_clouds = len(sonde_dict['Z_top'])
		sonde_dict['lwc'] = np.zeros((n_alt-1,))
		for kkk in range(n_clouds):
			i_cloud_temp = i_cloud[np.where((sonde_dict['Z_cloud'] >= sonde_dict['Z_base'][kkk]) & 
									(sonde_dict['Z_cloud'] <= sonde_dict['Z_top'][kkk]))[0]]
			sonde_dict['T_cloud_temp'] = sonde_dict['T'][i_cloud_temp]
			sonde_dict['P_cloud_temp'] = sonde_dict['P'][i_cloud_temp]
			sonde_dict['Z_cloud_temp'] = sonde_dict['Z'][i_cloud_temp]
			lwc_temp = mod_ad_lwc(sonde_dict['T_cloud_temp'], sonde_dict['P_cloud_temp'], sonde_dict['Z_cloud_temp'])
			sonde_dict['lwc'][i_cloud_temp[:-1]] = lwc_temp

		sonde_dict['lwp'] = 0.
		for kkk in range(n_alt-1):
			sonde_dict['lwp'] += sonde_dict['lwc'][kkk] * (sonde_dict['Z'][kkk+1] - sonde_dict['Z'][kkk])

		print(f"LWP = {sonde_dict['lwp']:.2f}")


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
		# pamData['hydro_q'][...,0] = 0			# CLOUD
		pamData['hydro_q'][...,0] = np.reshape(sonde_dict['lwc'], (1,1,n_alt-1))	# CLOUD
		pamData['hydro_q'][...,1] = 0			# ICE
		pamData['hydro_q'][...,2] = 0			# RAIN
		pamData['hydro_q'][...,3] = 0			# SNOW
		pamData['hydro_q'][...,4] = 0			# GRAUPEL


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