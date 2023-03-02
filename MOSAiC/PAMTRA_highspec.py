import numpy as np
import datetime as dt
import glob
import xarray as xr
from import_data import *
from met_tools import *
from data_tools import *
from my_classes import cloudnet, radiosondes, radiometers
import pdb
import pyPamtra
import multiprocessing
import warnings
import matplotlib as mpl
mpl.rcParams.update({'font.family': "monospace"})
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
os.environ['OPENBLAS_NUM_THREADS'] = "1"


def run_pamtra(
	path_output,
	k,
	sonde,
	freqs,
	nmlSet_dir,
	sfc_choice_dir,
	obs_height=0,
	run_parallel=True,
	n_cpus=2):

	"""
	Running pamtra with certain settings.

	Parameters:
	path_output : dict of str
		Dictionary of strings containing the possible output paths.
	k : int
		Time index of the current radiosonde launch to be simulated.
	sonde : class
		Class sonde with lat, lon, launch_time, temp, height, rh, pres,
		and optionally u, v. All at a certain sonde launch time.
	freqs : array of floats
		Frequencies to be simulated (in GHz).
	nmlSet_dir : dict
		Dictionary of PAMTRA settings.
	sfc_choice_dir : dict
		Dictionary of surface settings.
	obs_height : int or float
		Observation height (height of simulated instrument) in m. Default: 0 m
	run_parallel : bool
		Specifies if PAMTRA is to be run parallelly (True) or not (False). Default: True
	n_cpus : int
		Number of CPUs used for the execution of parallel PAMTRA. Default: 2
	"""

	lt_dt = dt.datetime.utcfromtimestamp(sonde.launch_time[k])
	lt_string = dt.datetime.strftime(lt_dt, "%Y%m%d_%H%M%SZ")

	print(lt_string)

	# Configure PAMTRA:
	pam = pyPamtra.pyPamtra()

	# NML Settings:
	pam.nmlSet['active'] = False
	pam.nmlSet['creator'] = "awalbroe"
	pam.nmlSet['data_path'] = "/net/blanc/awalbroe/Codes/pamtra/"
	for key in nmlSet_dir.keys():
		pam.nmlSet[key] = nmlSet_dir[key]

	# # Testing settings:
	# pam.nmlSet['add_obs_height_to_layer'] = False 		############# try True; 'default': False
	# pam.nmlSet['hydro_includehydroinrhoair'] = True		############# try False, 'default': True
	# pam.nmlSet['lgas_extinction'] = True					############# try False, 'default': True
	# pam.nmlSet['liq_mod'] = "Ell"							############# try 'TKC' (usual default), 'default': "Ell"
	# pam.nmlSet['obs_height'] = obs_height					############# try 10 m or DIFFERENT VALUE THAN pamData['obs_height'], 'default': obs_height

	pam.nmlSet['file_desc'] = lt_string


	sfc_refl_chosen = sfc_choice_dir['sfc_refl_chosen']
	sfc_model_chosen = sfc_choice_dir['sfc_model_chosen']
	sfc_type_chosen = sfc_choice_dir['sfc_type_chosen']

	# provide data: frequencies used, T, relhum. p, hgt-grid
	pamData = dict()
	shape2d = (1,1)

	# fill pamData:
	if sonde.lat[k] > 87:		# emissivity data base doesn't quite cover the North Pole
		pamData['lat'] = np.broadcast_to(87, shape2d)
	else:
		pamData['lat'] = np.broadcast_to(sonde.lat[k], shape2d)
	pamData['lon'] = np.broadcast_to(sonde.lon[k], shape2d)
	pamData['timestamp'] = sonde.launch_time[k]				# in sec since 1970-01-01 00:00:00 UTC
	if sfc_type_chosen == 1:
		pamData['sfc_type'] = np.ones(shape2d)
	elif sfc_type_chosen == 0:
		pamData['sfc_type'] = np.zeros(shape2d)
	elif sfc_type_chosen == -9999:		# set emissivity manually
		pam.nmlSet['emissivity'] = 0.5

	# surface properties: either use lfrac or the other 4 lines
	if sfc_model_chosen == 1:
		pamData['sfc_model'] = np.ones(shape2d)
	elif sfc_model_chosen == 0:
		pamData['sfc_model'] = np.zeros(shape2d)
	pamData['sfc_refl'] = np.chararray(shape2d)
	pamData['sfc_refl'][:] = sfc_refl_chosen

	pamData['obs_height'] = np.broadcast_to([obs_height], shape2d + (len([obs_height]), ))
	pamData['groundtemp'] = sonde.temp[k,0]			# in K, of course
	pamData['hgt_lev'] = sonde.height[k,:]

	pamData['relhum_lev'] = sonde.rh[k,:]*100	# in %
	pamData['press_lev'] = sonde.pres[k,:]		# in Pa
	pamData['temp_lev'] = sonde.temp[k,:]		# in K

	# Surface winds:
	if radiosonde_wind and np.all((~np.isnan(sonde.u[k,2])) and (~np.isnan(sonde.v[k,2]))):
		pamData['wind10u'] = sonde.u[k,2]
		pamData['wind10v'] = sonde.v[k,2]
	else:
		pamData['wind10u'] = 0
		pamData['wind10v'] = 0


	# with hydrometeors computed from cwp, iwp, rwp and swp of the testcase:
	LWP = 0
	RWP = 0
	IWP = 0
	SWP = 0
	cwc = 0					# cloud water content
	iwc = 0					# ice water content
	rwc = 0					# rain water content
	swc = 0					# snow water content

	shape4D = [1, 1, len(pamData['hgt_lev'])-1, 4]
	shape3D = [1, 1, len(pamData['hgt_lev'])-1]
	pamData['hydro_q'] = np.zeros(shape4D)
	pamData["hydro_q"][:,:,:,0] = cwc
	pamData["hydro_q"][:,:,:,1] = iwc			######################################################################## try removing lines 1,2,3
	pamData["hydro_q"][:,:,:,2] = rwc
	pamData["hydro_q"][:,:,:,3] = swc

	descriptorFile = np.array([
		  #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as' 'beta_as' 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 'd_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
		   ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
		   ('iwc_q', 1.0, -1, 700.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 'mono', -99.0, -99.0, -99.0, -99.0, 6e-05, -99.0, 'mie-sphere', 'heymsfield10_particles', -99.0),
		   ('rwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 50, 'exp', 0.22, 2.2, -99.0, -99.0, 0.00012, 0.006, 'mie-sphere', 'khvorostyanov01_drops', -99.0),
		   ('swc_q', 1.0, -1, -99.0, 0.069, 2.0, -99.0, -99.0, 3, 50, 'exp', 2e-06, 0.0, -99.0, -99.0, 2e-04, 0.02, 'mie-sphere', 'heymsfield10_particles', -99.0)], 
		  dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), ('canting', '<f8')]
		  )

	for hyd in descriptorFile: pam.df.addHydrometeor(hyd)

	output_path = path_output
	# check if output folder exists. If it doesn't, create it.
	output_path_dir = os.path.dirname(output_path)
	if not os.path.exists(output_path_dir):
		os.makedirs(output_path_dir)


	# create pamtra profile:
	pam.createProfile(**pamData)
	# run pamtra:
	if run_parallel:
		pam.runParallelPamtra(freqs, pp_deltaX=0, pp_deltaY=0, pp_deltaF=1, pp_local_workers=n_cpus)
	else:
		pam.runPamtra(freqs)

	filename_out = output_path + "pam_out_RS_%s_%s_%s.nc"%(radiosonde_version, considered_period, lt_string)
	pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)


###############################################################

"""
	This program is used to forward simulate MOSAiC radiosondes to V band frequencies
	in a high spectral resolution (50 - 60 GHz, with 0.01 GHz spacing). This will be
	done for a clear sky winter and a clear sky summer radiosonde from the MOSAiC
	campaign.
	Winter case: 2020-03-05 04:54:32 UTC
	Summer case: 2020-08-05 22:54:39 UTC
"""


# controlling options (e.g. choose versions)	###########################################
radiosonde_version = "level_2"			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'
radiosonde_wind = True					# if True: wind from radiosondes will be included in importing routines
simulate = False						# option to either simulate (True) or not and directly go to plotting (False,
										# requires fwd. sim. TB files to be at hand!)
considered_period = 'winter'		# specify the selected case: 'winter' or 'summer'


# Paths:
path_radiosondes = {'level_2': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"}
path_output = "/net/blanc/awalbroe/Data/MOSAiC_PAMTRA_highspec/"		# path of PAMTRA output directory
path_plot = "/net/blanc/awalbroe/Plots/MOSAiC_PAMTRA_highspec/"


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'winter': ["2020-03-05", "2020-03-05"],
					'summer': ["2020-08-05", "2020-08-05"]}
date_start = daterange_options[considered_period][0]
date_end = daterange_options[considered_period][1]

# choose radiosonde path
path_radiosondes = path_radiosondes[radiosonde_version]
files = {'winter': "PS122_mosaic_radiosonde_level2_20200305_045432Z.nc",
			'summer': "PS122_mosaic_radiosonde_level2_20200805_225439Z.nc"}

if simulate:

	# Import radiosonde data:
	sonde = radiosondes(path_radiosondes + files[considered_period], s_version=radiosonde_version, single=True, with_wind=radiosonde_wind)

	if radiosonde_wind:
		# compute u and v direction and mask afterwards:
		sonde.u, sonde.v = wspeed_wdir_to_u_v(sonde.wspeed, sonde.wdir)


	# Handle some nasty nans (only fills small gaps occurring away from the top
	# and bottom of the launch column):
	sonde.fill_gaps_easy()

	freqs = np.arange(1.0, 400.0000000001, 0.1)
	print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f}")

	obs_height = 10									# in m; try 0 m or 10 m (or even 15 m)
	run_parallel = True								# run parallel PAMTRA
	if run_parallel: n_cpus = int(multiprocessing.cpu_count()/2)		# half the number of available CPUs
	# Loop through all sonde:
	nmlSet_dir = {'add_obs_height_to_layer': False,				# 'default': False
					'hydro_includehydroinrhoair': True,			# 'default': True
					'lgas_extinction': True,					# 'default': True
					'liq_mod': "Ell",							# 'default': "Ell"
					'obs_height': obs_height,
					'save_ssp': False}							# eventually requires True when I want to investigate the scattering matrix
	sfc_choice_dir = {'sfc_refl_chosen': 'S',					# try 'L': Specular; default: 'S'
					'sfc_model_chosen': 1,						# 0: water, 1: land; default: 1;
					'sfc_type_chosen': 1}						# 0: water, 1: land; default: 1;

	# Running PAMTRA:
	for k in range(len(sonde.launch_time)):
		run_pamtra(path_output, k, sonde, freqs, nmlSet_dir, sfc_choice_dir, obs_height, run_parallel=run_parallel, n_cpus=n_cpus)


else:

	# Import data:
	file_winter = glob.glob(path_output + "*_winter_*.nc")
	file_summer = glob.glob(path_output + "*_summer_*.nc")
	PAM_DS_winter = xr.open_dataset(file_winter[0])
	PAM_DS_summer = xr.open_dataset(file_summer[0])


	mwr_freqs = np.array([	22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000,
									175.810, 178.310, 179.810, 180.810, 181.810, 182.710,
									183.910, 184.810, 185.810, 186.810, 188.310, 190.810,
									243.000, 340.000])
	
	fs = 18

	fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(16,6))
	ylims = np.array([0, 295])


	x_plot = PAM_DS_winter.frequency.values
	y_plot = PAM_DS_winter.tb[0,0,0,-1,:,:].mean(axis=-1).values
	plot_winter = ax0.plot(x_plot, y_plot, color=(0,0,0), linewidth=1.25, linestyle='dashed',
								label=f"Winter, IWV = 0.89 mm")

	x_plot = PAM_DS_summer.frequency.values
	y_plot = PAM_DS_summer.tb[0,0,0,-1,:,:].mean(axis=-1).values
	plot_summer = ax0.plot(x_plot, y_plot, color=(0,0,0), linewidth=1.25,
								label=f"Summer, IWV = 16.09 mm")

	# fill between:
	ax0.fill_between(x_plot, y1=y_plot, y2=0.0, facecolor=(1.0,0.855,0.5,0.2))

	# add markers for HATPRO and MiRAC-P frequencies:
	for frq in mwr_freqs:
		if frq < 170.0:	# then HATPRO
			ax0.plot([frq, frq], ylims, color=(17.0/255.0,74.0/255.0,196.0/255.0), linewidth=0.75, zorder=-2)

		else:
			ax0.plot([frq, frq], ylims, color=(0.0,199.0/255.0,157.0/255.0), linewidth=0.75, zorder=-2)

	# dummy for legend:
	ax0.plot([np.nan, np.nan], [np.nan, np.nan], color=(17.0/255.0,74.0/255.0,196.0/255.0), linewidth=1.5, label="HATPRO frequencies")
	ax0.plot([np.nan, np.nan], [np.nan, np.nan], color=(0.0,199.0/255.0,157.0/255.0), linewidth=1.5, label="MiRAC-P frequencies")

	# add band identifiers (text labels):
	band_labels = ["K", "V", "G", "243", "340"]
	band_bounds = {'K': [20, 35], 'V': [50, 60], 'G': [170, 200], "243": [230, 250], '340': [335, 345]}
	for band in band_labels:
		frq_band = mwr_freqs[(mwr_freqs >= band_bounds[band][0]) & (mwr_freqs <= band_bounds[band][1])]
		avg_freq_band = np.mean(frq_band)
		ax0.text(avg_freq_band, 1.00*np.diff(ylims)+ylims[0], 
					f"{band}", 
					ha='center', va='bottom', color=(0,0,0), fontsize=fs-2, transform=ax0.transData)

	# and another one indicating the meaning of K, V, ...:
	ax0.text(0.00, 1.00, "Freq. bands", 
			ha='right', va='bottom', color=(0,0,0), fontsize=fs-2, transform=ax0.transAxes)

	ax0.set_xlim(left=0.0, right=x_plot[-1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	han, solo = ax0.get_legend_handles_labels()
	le_leg = ax0.legend(handles=han, labels=solo, loc='lower right', fontsize=fs-2, framealpha=1.0)

	ax0.set_title("Simulated TBs - Microwave spectrum", fontsize=fs, pad=24)
	ax0.set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=0.75)
	ax0.set_ylabel("TB (K)", fontsize=fs, labelpad=0.75)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	ax0.tick_params(axis='both', labelsize=fs-2)


	fig0.savefig(path_plot + "MOSAiC_radiosonde_PAMTRA_1-500GHz_highres.png", dpi=400, bbox_inches='tight')


print("Done....")