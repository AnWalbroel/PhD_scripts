import numpy as np
import os
import datetime as dt
import glob
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing

import sys
sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import select_MWR_channels, Gband_double_side_band_average, break_str_into_lines
from met_tools import convert_rh_to_abshum, compute_IWV

os.environ['OPENBLAS_NUM_THREADS'] = "1" # workaround for pamtra and openblas, required if PAMTRA was compiled with multithreadding version of openBLAS

if 'PAMTRA_DATADIR' not in os.environ:
	os.environ['PAMTRA_DATADIR'] = "" # actual path is not required, but the variable has to be defined.
import pyPamtra


"""
	This script creates a synthetic atmospheric profile to investigate the influence
	of moisture layers in different altitudes. Temperature and pressure follow the ICAO
	standard atmosphere 
	( https://www.dwd.de/DE/service/lexikon/begriffe/S/Standardatmosphaere_pdf.pdf?__blob=publicationFile&v=3 ).
	The created profiles will be plotted.
"""


# paths:
path_pam_out = "/net/blanc/awalbroe/Data/EUREC4A/forward_sim_dropsondes/simple_profile_pamtra/"
path_plots = "/net/blanc/awalbroe/Plots/EUREC4A/simple_profile_pamtra/"

# Check if the PAMTRA output path exists:
pam_out_path_dir = os.path.dirname(path_pam_out)
if not os.path.exists(pam_out_path_dir):
	os.makedirs(pam_out_path_dir)


# Settings:
# specify humidity profile settings. options: 'default', 'elevated_inversion', 
# 'humid_at_aircraft'
hum_settings = ['default', 'elevated_inversion', 'elevated_inversion_balanced', 
				'humid_at_aircraft', 'humid_at_aircraft_balanced']
# hum_settings = ['default', 'combined_effects']
save_figures = True					# if True, figures will be saved
mwr_band = 'G'						# select MWR band: options: 'K', 'V', 'W', 'F', 'G' 
									# and combinations, i.e.: 'K+V+G*

n_settings = len(hum_settings)
tb_sim = np.zeros((n_settings,6))

for k, hum_set in enumerate(hum_settings):
	# set some geometrical variables:
	obs_height = np.array([10000.0])			# in m; as 1D array
	lat = 13.									# in deg N
	lon = -57.									# in deg E
	sst = 273.15 + 26.0
	time = (dt.datetime(2020,2,2,14,0) - dt.datetime(1970,1,1)).total_seconds()
	height = np.arange(0, obs_height[0] + 0.001, 50)
	height_icao = np.array([0., 1000., 2000., 3000., 5000., 7000., 9000., 10000.])

	# met. variables in SI units: rh must be converted to % for pamtra
	temp = np.array([273.15 + 25.0, 273.15 + 18.5, 273.15 + 12.0, 273.15 + 5.5, 
					273.15 - 7.5, 273.15 - 20.5, 273.15 - 33.5, 273.15 - 40.0])
	pres = np.array([101325., 89875., 79495., 70109., 54020., 41061., 30742., 26687.])

	rh_grid = np.arange(0., obs_height[0] + 0.001, 500)
	if hum_set == 'default':
		rh = np.array([0.70, 0.80, 0.90, 0.90, 0.80, 0.04, 0.12, 0.05, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04])			# 5500 - 10000 m

	elif hum_set == 'elevated_inversion':
		rh = np.array([0.70, 0.80, 0.90, 0.90, 0.80, 0.70, 0.04, 0.12, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04])			# 5500 - 10000 m

	elif hum_set == 'elevated_inversion_balanced':
		rh = np.array([0.70, 0.80, 0.90, 0.80, 0.70, 0.20, 0.04, 0.12, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04])			# 5500 - 10000 m

	elif hum_set == 'humid_at_aircraft':
		rh = np.array([0.70, 0.80, 0.90, 0.90, 0.80, 0.04, 0.12, 0.05, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.12, 0.12])			# 5500 - 10000 m

	elif hum_set == 'humid_at_aircraft_balanced':
		rh = np.array([0.70, 0.80, 0.90, 0.90, 0.80, 0.04, 0.12, 0.05, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.07, 0.07, 0.07, 0.07, 0.02, 0.02, 0.12, 0.12])			# 5500 - 10000 m

	elif hum_set == 'combined_effects':
		rh = np.array([0.70, 0.80, 0.90, 0.80, 0.70, 0.20, 0.04, 0.12, 0.05, 0.05, 0.05, 		# 0 - 5000 m
						0.05, 0.08, 0.07, 0.07, 0.07, 0.07, 0.02, 0.02, 0.12, 0.12])			# 5500 - 10000 m


	# interpolate data to finer height grid:
	temp = np.interp(height, height_icao, temp)
	pres = np.interp(height, height_icao, pres)
	rh = np.interp(height, rh_grid, rh)
	n_alt = len(height)

	# Compute IWV:
	rho_v = convert_rh_to_abshum(temp, rh)
	iwv = compute_IWV(rho_v, height)
	print(f"IWV = {iwv:.2f}")


	# Plot profiles:
	fig1 = plt.figure(figsize=(14,9))
	ax_T = plt.subplot2grid((1,3), (0,0))
	ax_P = plt.subplot2grid((1,3), (0,1))
	ax_RH = plt.subplot2grid((1,3), (0,2))

	ax_T.plot(temp, height, color=(0,0,0))
	ax_P.plot(pres, height, color=(0,0,0))
	ax_RH.plot(rh, height, color=(0,0,0))

	ax_T.minorticks_on()
	ax_P.minorticks_on()
	ax_RH.minorticks_on()


	ax_T.grid(which='both', axis='both')
	ax_P.grid(which='both', axis='both')
	ax_RH.grid(which='both', axis='both')

	ax_T.set_ylim(0.0, obs_height + 500.0)
	ax_P.set_ylim(0.0, obs_height + 500.0)
	ax_RH.set_ylim(0.0, obs_height + 500.0)

	ax_T.set_xlim(215, 295)
	ax_P.set_xlim(25000, 102500)
	ax_RH.set_xlim(0, 1)

	ax_T.set_ylabel("Height (m)")

	ax_T.set_xlabel("Temperature (K)")
	ax_P.set_xlabel("Pressure (Pa)")
	ax_RH.set_xlabel("Relative humidity ()")

	fig1.suptitle(f"Synthetic profile - {hum_set}")

	if save_figures:
		fig1.savefig(path_plots + f"Synthetic_profile_EUREC4A_{hum_set}.png", dpi=400)
	else:
		plt.show()
	plt.close()


	# HAMP FREQUENCIES:
	frq = [22.2400,23.0400,23.8400,25.4400,26.2400,27.8400,31.4000,50.3000,51.7600,52.8000,53.7500,
			54.9400,56.6600,58.0000,90.0000,110.250,114.550,116.450,117.350,120.150,121.050,122.950,
			127.250,170.810,175.810,178.310,179.810,180.810,181.810,182.710,183.910,184.810,185.810,
			186.810,188.310,190.810,195.810]

	# create pamtra object; change settings:
	pam = pyPamtra.pyPamtra()

	pam.nmlSet['hydro_adaptive_grid'] = True
	pam.nmlSet['add_obs_height_to_layer'] = False		# adds observation layer height to simulation height vector
	pam.nmlSet['passive'] = True						# passive simulation
	pam.nmlSet['active'] = False						# False: no radar simulation

	pamData = dict()
	shape2d = [1, 1]


	pamData['lon'] = np.broadcast_to(lon, shape2d)
	pamData['lat'] = np.broadcast_to(lat, shape2d)
	pamData['timestamp'] = np.broadcast_to(time, shape2d)


	# surface type & reflectivity:
	pamData['sfc_type'] = np.zeros(shape2d)			# 0: ocean, 1: land
	pamData['sfc_refl'] = np.chararray(shape2d)
	pamData['sfc_refl'][:] = 'F'
	pamData['sfc_refl'][pamData['sfc_type'] == 1] = 'S'

	pamData['obs_height'] = np.broadcast_to(obs_height, shape2d + [len(obs_height), ]) # observation altitude
	pamData['groundtemp'] = np.broadcast_to(sst, shape2d)
	pamData['wind10u'] = np.broadcast_to(0., shape2d)
	pamData['wind10v'] = np.broadcast_to(0., shape2d)

	# 3d variables:
	shape3d = shape2d + [n_alt]
	pamData['hgt_lev'] = np.broadcast_to(height, shape3d)
	pamData['temp_lev'] = np.broadcast_to(temp, shape3d)
	pamData['press_lev'] = np.broadcast_to(pres, shape3d)
	pamData['relhum_lev'] = np.broadcast_to(100*rh, shape3d)

	# 4d variables: hydrometeors:
	shape4d = [1, 1, n_alt-1, 5]			# potentially 5 hydrometeor classes with this setting
	pamData['hydro_q'] = np.zeros(shape4d)
	pamData['hydro_q'][...,0] = 0			# CLOUD
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
	print("Launching PAMTRA in 5....4....3....2....1....")
	# pam.runPamtra(frq)

	n_cpus = int(multiprocessing.cpu_count()/2)		# half the number of available CPUs
	pam.runParallelPamtra(frq, pp_deltaX=0, pp_deltaY=0, pp_deltaF=1, pp_local_workers=n_cpus)

	# # save output:
	# filename_out = os.path.join(path_pam_out, "pamtra_" + os.path.basename(filename_in))
	# pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)


	# Save TBs:
	tb_out = pam.r['tb'][:,0,0,0,:,:].mean(axis=-1)
	freq = np.asarray(pam.set['freqs'])
	tb_dsba, freq_dsba = Gband_double_side_band_average(tb_out, freq)
	tb_dsba, freq_dsba = select_MWR_channels(tb_dsba, freq_dsba, band=mwr_band, return_idx=0)
	tb_dsba = tb_dsba[:,:-1]			# remove last G band freq because this doesn't appear in MWR data
	freq_dsba = freq_dsba[:-1]
	tb_sim[k,:] = tb_dsba
	freq = freq_dsba


# plot created TBs
fs = 16
fig1 = plt.figure(figsize=(12,10))
ax1 = plt.axes()

n_freq = len(freq)
tb_cmap = mpl.cm.get_cmap('tab10', n_freq)
for k in range(n_freq):
	ax1.plot(np.arange(n_settings), tb_sim[:,k], color=tb_cmap(k), linewidth=1.4,
				marker='.', markersize=9, label=f"{freq[k]:.2f}")


lh, ll = ax1.get_legend_handles_labels()
ax1.legend(handles=lh, labels=ll, loc='upper right', fontsize=fs-2)

ax1.minorticks_on()
ax1.grid(which='both', axis='both')
ax1.set_xlim(-0.5, n_settings-0.5)

ax1.set_xticks(np.arange(n_settings))

hum_settings_labels = []
for hum_set in hum_settings:
	hum_settings_labels.append(break_str_into_lines(hum_set, 20, split_at='_'))
ax1.set_xticklabels(hum_settings_labels)

ax1.tick_params(labelsize=fs-4)

ax1.set_ylabel("TB (K)", fontsize=fs)
ax1.set_title("Simulated TBs", fontsize=fs)

if save_figures:
	fig1.savefig(path_plots + f"Sim_TBs_{mwr_band}band_rh_prof_variation.png", dpi=400)
else:
	plt.show()