import pyPamtra
import numpy as np
import pdb


def run_pamtra_run(sonde_dict, settings_dict):

	"""
	This function sets PAMTRA's settings and takes radiosonde data to simulate
	brightness temperatures out of radiosonde data with certain settings (i.e., 
	salinity, sea surface temperature, elevation angle, ...).

	Parameters:
	-----------
	sonde_dict : dict
		Dictionary of radiosonde data. 'temp' for temperature in K, 'pres' for pressure in Pa,
		'relhum' in %, wind data in U and V direction in m s-1.
	settings_dict : dict
		Dictionary of settings for the PAMTRA setup, for radiosonde setup and more.

	Output:
	-------
	pam : pam object
		Pam object in which simulated brightness temperatures can be found.
	"""


	# create pam object:
	pam = pyPamtra.pyPamtra()


	# general settings:
	pam.nmlSet['hydro_adaptive_grid'] = True
	pam.nmlSet['add_obs_height_to_layer'] = False		# adds observation layer height to simulation height vector
	pam.nmlSet['save_psd'] = False						# can save particle size distribution
	pam.nmlSet['passive'] = True						# passive simulation
	pam.nmlSet['active'] = False						# False: no radar
	pam.nmlSet['data_path'] = "/home/tenweg/pamtra/"
	pam.nmlSet['liq_mod'] = settings_dict['liq_mod']	# TKC (default) or Ell

	# define the pamtra profile: temp, relhum, pres, height, lat, lon, timestamp, lfrac, obs_height, ...
	pamData = dict()
	shape2d = (1,1)
	
	pamData['lon'] = np.broadcast_to(settings_dict['lon'], shape2d)
	pamData['lat'] = np.broadcast_to(settings_dict['lat'], shape2d)
	pamData['timestamp'] = sonde_dict['launch_time']
	pamData['sfc_type'] = np.ones(shape2d)

	# surface properties: either use lfrac or the other 4 lines
	# pamData['lfrac'] = lfrac
	pamData['sfc_model'] = np.ones(shape2d)		# 0 = sea, 1 = land --> and we ve got sea conditions only
	pamData['sfc_refl'] = np.chararray(shape2d)
	pamData['sfc_refl'][:] = settings_dict['sfc_refl']
	# pamData['sfc_salinity'] = np.broadcast_to(settings_dict['salinity'], shape2d)		# in PSU

	pamData['obs_height'] = np.broadcast_to(settings_dict['obs_height'], shape2d + (len(settings_dict['obs_height']), ))
	pamData['hgt_lev'] = sonde_dict['height']				# 0, 30, 60, 90, ..., 9000 m, level specification (instead of layer)
	

	# make sure relative humidity doesn't exceed sensible values:
	if np.any(sonde_dict['relhum'] > 100.0): pdb.set_trace()
	sonde_dict['relhum'][sonde_dict['relhum'] > 100.0] = 100.0
	sonde_dict['relhum'][sonde_dict['relhum'] < 0.0] = 0.0


	# put meteo data into pamData:
	shape3d = (1,1,len(sonde_dict['height']))
	pamData['relhum_lev'] = np.broadcast_to(sonde_dict['relhum'], shape3d)
	pamData['press_lev'] = np.broadcast_to(sonde_dict['pres'], shape3d)
	pamData['temp_lev'] = np.broadcast_to(sonde_dict['temp'], shape3d)

	# Surface winds
	pamData['wind10u'] = sonde_dict['u'][0]*0.0
	pamData['wind10v'] = sonde_dict['v'][0]*0.0
	pamData['groundtemp'] = settings_dict['sst']


	# 4d variables: hydrometeors:
	# with hydrometeors computed from cwp, iwp, rwp and swp of the testcase:
	# LWP will later be replaced by modified adiabatic computation in clouds
	# detected via 95 % rel. humidity (see /Notes/Miscallaneous/MiRAC-P_retrieval.txt).
	cwc = 0					# cloud water content
	iwc = 0					# ice water content
	rwc = 0					# rain water content
	swc = 0					# snow water content

	shape4D = [1, 1, len(pamData['hgt_lev'])-1, 1]
	shape3D = [1, 1, len(pamData['hgt_lev'])-1]
	pamData['hydro_q'] = np.zeros(shape4D)
	pamData["hydro_q"][:,:,:,0] = cwc

	descriptorFile = np.array([
		  #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
		   ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere', 'khvorostyanov01_drops', -99.0)], 
		  dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
		  )

	for hyd in descriptorFile: pam.df.addHydrometeor(hyd)


	# create pamtra profile from pamData and run pamtra at all specified frequencies:
	pam.createProfile(**pamData)
	pam.runPamtra(settings_dict['freqs'])


	return pam