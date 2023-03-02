import numpy as np
import datetime as dt
import glob
from import_data import *
from met_tools import *
from data_tools import *
from my_classes import cloudnet, radiosondes, radiometers
import pdb
import pyPamtra
# import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = "1"


def run_pamtra(
	path_output,
	k,
	sonde,
	freqs,
	freq_label,
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
	freqs : dict of float arrays
		Dictionary containing two sets of frequencies. One for 'hatpro', one
		for 'mirac-p'. Must be used in combination with freq_label.
	freq_label : str
		Frequency label to choose the instrument, whose measurements shall be
		simulated. Valid options: 'hatpro', 'mirac-p'.
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
	# pam.nmlSet['save_ssp'] = False						############# eventually requires True when I want to investigate the scattering matrix

	pam.nmlSet['file_desc'] = "%s"%(lt_string)


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
	# LWP will later be replaced by modified adiabatic computation in clouds
	# detected via 95 % rel. humidity (see /Notes/Miscallaneous/MiRAC-P_retrieval.txt).
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
	pamData["hydro_q"][:,:,:,1] = iwc
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

	tests_string = "test_%i%i%i_%s_%im_%s_%i%i"%(int(pam.nmlSet['add_obs_height_to_layer']), 
												int(pam.nmlSet['hydro_includehydroinrhoair']), 
												int(pam.nmlSet['lgas_extinction']),
												pam.nmlSet['liq_mod'],
												int(obs_height),
												sfc_refl_chosen,
												sfc_type_chosen,
												sfc_model_chosen)
		# e.g.: "test_011_Ell_0m_0": structured like this: test_<0 or 1 (depending 
		# on False or True >, then: _<liq_mod>_<obs_height (in m)>...
	output_path = path_output[freq_label] + tests_string + "/"
	# check if output folder exists. If it doesn't, create it.
	output_path_dir = os.path.dirname(output_path)
	if not os.path.exists(output_path_dir):
		os.makedirs(output_path_dir)


	# create pamtra profile:
	pam.createProfile(**pamData)
	# run pamtra:
	pam.runParallelPamtra(freqs[freq_label], pp_deltaX=0, pp_deltaY=0, pp_deltaF=1, pp_local_workers=n_cpus)
	# pam.runPamtra(freqs[freq_label])

	filename_out = output_path + "pam_out_RS_%s_%s_%s.nc"%(radiosonde_version, considered_period, lt_string)
	pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)


###############################################################

"""
	This program is used to forward simulate MOSAiC radiosondes to HATPRO and MiRAC-P
	frequencies so that differences between simulated and observed TBs can be analysed.
"""


# controlling options (e.g. choose versions)	###########################################
radiosonde_version = "level_2"			# MOSAiC radiosonde version: options: 'level_2' (default), 'mossonde', 'psYYMMDDwHH'
radiosonde_wind = True					# if True: wind from radiosondes will be included in importing routines
mirac_version = 'RPG'					# MiRAC-P version: options: 'RPG' (default (2021-04-30)), 'i00', 'i01', 'i02', 'i03'
hatpro_version = 'v01'					# HATPRO version: options: 'i01', 'v00', 'v01' (latter is default (2021-04-30))
freq_label = 'mirac-p'					# either 'hatpro' or 'mirac-p'... it specifies which instrument is considered
simulate = False						# option to either simulate (True) or not and directly go to plotting (False,
										# requires fwd. sim. TB files to be at hand!)
color_code_iwv = True				# if true, the scatter and time series plots will be color-coded by IWV
save_figures = False
save_offset_slope = True			# save computed TB offset and slope to NETCDF file
compare_IDL_stp_sims = False		# option not to compare PAMTRA sims with MiRAC-P obs, but with sims via IDL STP tool
plot_iwv_x_TB_diff = False			# option to plot (IWV) x (TB_obs - TB_sim)
plot_for_calibration_times = True	# if true, a TB comparison for each frequency for each calibration period will be created;
									# IT IS REQUIRED TO UNCOMMENT A FOR LOOP AND INDENT THE LINES BELOW IF TRUE
									# if false, entire MOSAiC campaign will be considered at once
considered_period = 'mosaic'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# 'user': user defined


# Paths:
path_radiosondes = {'level_2': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/",
					'mossonde': "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/",
					'psYYMMDDwHH': "/data/testbed/datasets/MOSAiC/rs41/"}
path_hatpro = "/data/obs/campaigns/mosaic/hatpro/l1/"
path_mirac = {'RPG': "/data/obs/campaigns/mosaic/mirac-p/l1/",
				'level_1': "/data/obs/campaigns/mosaic/mirac-p/l2/"}
path_cloudnet = "/data/obs/campaigns/mosaic/cloudnet/products/classification/"
# path of the PAMTRA output for HATPRO and MiRAC-P frequencies:
path_output = {'hatpro': "/net/blanc/awalbroe/Data/MOSAiC_radiosondes/fwd_sim/hatpro_freq/",
				'mirac-p': "/net/blanc/awalbroe/Data/MOSAiC_radiosondes/fwd_sim/lhumpro_freq/"}
path_mwr_offset_output = "/net/blanc/awalbroe/Data/MOSAiC_radiometers_offsets/"		# here the offsets and slope of mwr offsets will be saved to
path_plot = {'hatpro': "/net/blanc/awalbroe/Plots/TB_comparison_mwr_pamtrasonde/hatpro_freq/",
				'mirac-p': "/net/blanc/awalbroe/Plots/TB_comparison_mwr_pamtrasonde/lhumpro_freq/"}

if compare_IDL_stp_sims:
	path_idl_tbs = "/net/blanc/awalbroe/Data/MOSAiC_radiosondes/fwd_sim/"
	if plot_for_calibration_times:
		raise ValueError("It doesn't make sense to plot IDL STP TBs versus PAMTRA TBs for different calibration periods. " +
							" Just use the entire time period!")

if plot_iwv_x_TB_diff:
	if save_offset_slope:
		raise ValueError("You probably want to save sim. vs. obs. TB statistics (bias, rmse, R, ...). But setting both " +
							"'plot_iwv_x_TB_diff' and 'save_offset_slope' True, no such statistics will be computed.")
	if color_code_iwv:
		warnings.warn("'color_code_iwv' set to False because it doesn't make sense to color code IWV when it is explicitly" +
						" plotted against.")
		color_code_iwv = False		# I


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-02-25", "2020-04-01"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"
if not date_start and not date_end: raise ValueError("Please specify a date range in yyyy-mm-dd format.")

# choose radiosonde path
path_radiosondes = path_radiosondes[radiosonde_version]
if mirac_version in ['i00', 'i01', 'i02', 'i03', 'v00', 'v01']:
	path_mirac = path_mirac['level_1']
else:
	path_mirac = path_mirac[mirac_version]


# Import radiosonde data:
sonde = radiosondes(path_radiosondes, s_version=radiosonde_version, with_wind=radiosonde_wind, date_start=date_start, date_end=date_end)

# Import cloudnet data to filter for clear sky scenes.
# After filtereing out cloudy scenes, time overlaps with
# radiosondes will be searched.
cloudnet_data = cloudnet(path_cloudnet, date_start, date_end)
cloudnet_data = cloudnet_data.clear_sky_only(truncate=False, ignore_x_lowest=1)			# default: ignore_x_lowest=3

# Unfortunately, it does not suffice to check time overlaps of sonde launches with 
# clear sky time stamps of cloudnet (with data_tools.filter_time function). That would
# only check if one launch time exists around clear sky time stamps. But this does not
# ensure, that e.g., all cloudnet time stamps from launch time until launch time + 30
# minutes are clear sky.
t_window = 1800		# time window (in seconds) from sonde launch time until launch time + t_window
ths = 1.00			# fraction of cloudnet time stamps that must be clear sky from launch time : launch_time + t_window
s_mask = filter_clear_sky_sondes_cloudnet(sonde.launch_time, cloudnet_data.time, cloudnet_data.is_clear_sky, ths, 
											window=t_window)

# Truncate sondes:
sonde.launch_time = sonde.launch_time[s_mask]
sonde.lat = sonde.lat[s_mask]
sonde.lon = sonde.lon[s_mask]
sonde.pres = sonde.pres[s_mask,:]			# in Pa
sonde.temp = sonde.temp[s_mask,:]			# in K
sonde.rh = sonde.rh[s_mask,:]				# between 0 and 1
sonde.height = sonde.height[s_mask,:]		# in m
sonde.rho_v = sonde.rho_v[s_mask,:]			# in kg m^-3
sonde.q = sonde.q[s_mask,:]					# in kg kg^-1
sonde.iwv = sonde.iwv[s_mask]				# in kg m^-2
if radiosonde_wind:
	# compute u and v direction and mask afterwards:
	sonde.u, sonde.v = wspeed_wdir_to_u_v(sonde.wspeed, sonde.wdir)

	sonde.u = sonde.u[s_mask,:]	# in m s^-1
	sonde.v = sonde.v[s_mask,:]		# in deg

# Handle some nasty nans (only fills small gaps occurring away from the top
# and bottom of the launch column):
sonde.fill_gaps_easy()

freqs = {'hatpro': [22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4,
				51.26, 52.28, 53.86, 54.94, 56.66, 57.3, 58],
		'mirac-p': np.sort([183.31-0.6, 183.31+0.6, 183.31-1.5, 183.31+1.5,
					183.31-2.5, 183.31+2.5, 183.31-3.5, 183.31+3.5,
					183.31-5.0, 183.31+5.0, 183.31-7.5, 183.31+7.5, 243.0, 340.0])}

# # for freq_label in ['hatpro', 'mirac-p']:
obs_height = 10									# in m; try 0 m or 10 m (or even 15 m)
run_parallel = False								# run parallel PAMTRA
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
if simulate:
	for k in range(len(sonde.launch_time)):
		run_pamtra(path_output, k, sonde, freqs, freq_label, nmlSet_dir, sfc_choice_dir, obs_height, run_parallel=run_parallel, n_cpus=n_cpus)

del cloudnet_data

# Now the PAMTRA stuff has been simulated, we can clear up the workspace a bit and
# start the comparison with real MWR observations:


# Import mwr data:
if freq_label == 'mirac-p':
	mwr = radiometers(path_mirac, 'mirac-p', version=mirac_version, date_start=date_start, date_end=date_end, truncate_flagged=True)
	n_freq = len(mwr.freq)

	if plot_for_calibration_times:
		mwr.get_calibration_times('mirac-p', to_epochtime=True)
		n_calib = len(mwr.calibration_times)

		# calibration periods: start of MOSAiC - first calibration; between calibrations; last calib. to end of MOSAiC
		mwr.calibration_periods = [[datetime_to_epochtime(dt.datetime.strptime(daterange_options['mosaic'][0], "%Y-%m-%d")), 
									mwr.calibration_times[0]]]
		for ct_idx, ct in enumerate(mwr.calibration_times):
			if ct_idx < n_calib - 1:
				mwr.calibration_periods.append([ct, mwr.calibration_times[ct_idx+1]])
		mwr.calibration_periods.append([mwr.calibration_times[-1], 
										datetime_to_epochtime(dt.datetime.strptime(daterange_options['mosaic'][1], "%Y-%m-%d"))])

elif freq_label == 'hatpro':
	mwr = radiometers(path_hatpro, 'hatpro', version=hatpro_version, date_start=date_start, date_end=date_end, truncate_flagged=True)
	# With mwr.__dict__.keys() you can inquire the attributes of the class / instance.
	n_freq = len(mwr.freq)

	if plot_for_calibration_times:
		mwr.get_calibration_times('hatpro', to_epochtime=True)
		n_calib = len(mwr.calibration_times)

		# calibration periods: start of MOSAiC - first calibration; between calibrations; last calib. to end of MOSAiC
		mwr.calibration_periods = [[datetime_to_epochtime(dt.datetime.strptime(daterange_options['mosaic'][0], "%Y-%m-%d")), 
									mwr.calibration_times[0]]]
		for ct_idx, ct in enumerate(mwr.calibration_times):
			if ct_idx < n_calib - 1:
				mwr.calibration_periods.append([ct, mwr.calibration_times[ct_idx+1]])
		mwr.calibration_periods.append([mwr.calibration_times[-1], 
										datetime_to_epochtime(dt.datetime.strptime(daterange_options['mosaic'][1], "%Y-%m-%d"))])

# Import and concatenate all simulated sondes:
path_output_chosen = path_output[freq_label]
all_out_folders = sorted(glob.glob(path_output_chosen + "*"))

path_plot_chosen = path_plot[freq_label]

# Cycle through all test scenarios (all out_folders):
if save_offset_slope:	# initialise arrays where the values can be saved to
	n_calib_periods = len(mwr.calibration_periods)
	slope_save = np.zeros((n_calib_periods, n_freq))	# slope values will be saved here
	offset_save = np.zeros((n_calib_periods, n_freq))	# offset values will be saved here
	bias_save = np.zeros((n_calib_periods, n_freq))		# bias values will be saved here
	n_samp = np.zeros((n_calib_periods,))				# number of samples for computation of slope, ... saved here

for all_out_folder in all_out_folders:

	if "test_011_Ell_10m_S_11" in all_out_folder:
		print(all_out_folder)
		all_out_folder = all_out_folder + "/"

		all_out_files = sorted(glob.glob(all_out_folder + "*.nc"))
		
		# Concatenate all output files along dimension x of PAMTRA output files:
		FWD_SIM_DS = xr.open_mfdataset(all_out_files, concat_dim='grid_x', combine='nested',
										preprocess=pam_out_drop_useless_dims)

		if freq_label == 'mirac-p':
			# Apply double side band average on FWD_SIM_DS TBs:
			FWD_SIM_DS['tb_dsba'] = xr.DataArray(Gband_double_side_band_average(FWD_SIM_DS.tb.values,
																				FWD_SIM_DS.frequency.values)[0],
												coords={'grid_x': FWD_SIM_DS.grid_x,
														'frequency_dsba': np.array([183.91, 184.81, 185.81, 186.81,
																					188.31, 190.81, 243., 340.])},
												dims=['grid_x', 'frequency_dsba'])

			# Renaming variables:
			FWD_SIM_DS = FWD_SIM_DS.rename({'tb': 'tb_original', 'frequency': 'frequency_original'})
			FWD_SIM_DS = FWD_SIM_DS.rename({'tb_dsba': 'tb', 'frequency_dsba': 'frequency'})
			
							
		# Average MWR data over radiosonde launch : launch + 5 minutes:
		# For this, create arrays ranging from launch through launch + 300 seconds and interpolate
		# MWR data on this time axis.
		n_sondes = len(FWD_SIM_DS.time)
		new_time_axis = np.zeros((n_sondes, 301))
		mwr_dict = {'time': np.full((n_sondes,), np.nan),
					'TB': 	np.full((n_sondes, n_freq), np.nan)}

		for k in range(n_sondes):
			print(k)
			new_time_axis[k,:] = np.arange(FWD_SIM_DS.time[k], FWD_SIM_DS.time[k]+301)

			if not compare_IDL_stp_sims:
				mwr_dict['time'][k] = FWD_SIM_DS.time.values[k]
				for jj in range(n_freq):
					if (new_time_axis[k,-1] >= mwr.time[0]) and (new_time_axis[k,0] <= mwr.time[-1]):
						mwr_dict['TB'][k,jj] = np.nanmean(np.interp(new_time_axis[k,:], mwr.time,
															mwr.TB[:,jj], left=np.nan, right=np.nan), axis=0)

		# Create plots:
		fs = 19		# fontsize


		if compare_IDL_stp_sims:	# then import IDL STP simulated MOSAiC radiosondes
			idl_sim_file = glob.glob(path_idl_tbs + "TB_calc_*level2.txt")
			idl_tb_dict = import_IDL_TBs_txt(idl_sim_file[0])

			# perform double side band average:
			idl_tb_dict['tb'], idl_tb_dict['freq'] = Gband_double_side_band_average(idl_tb_dict['tb'], idl_tb_dict['freq'])

			# select frequencies:
			if freq_label == 'mirac-p':
				freq_idx = np.where((idl_tb_dict['freq'] > 180) & (idl_tb_dict['freq'] < 350))[0]
			elif freq_label == 'hatpro':
				freq_idx = np.where((idl_tb_dict['freq'] > 15) & (idl_tb_dict['freq'] < 80))[0]

			idl_tb_dict['tb'] = idl_tb_dict['tb'][:,freq_idx]
			idl_tb_dict['freq'] = idl_tb_dict['freq'][freq_idx]

			# select correct radiosondes: Cycle through my simulated sondes and find matches with idl sim. sondes:
			idl_time_idx = np.asarray([np.where(np.abs(idl_tb_dict['time'] - rs_time) < 3600)[0] for rs_time in FWD_SIM_DS.time.values])
			idl_tb_sel = np.full((n_sondes, len(idl_tb_dict['freq'])), np.nan)
			idl_time_sel = np.full((n_sondes,), np.nan)
			idl_lwp_sel = np.full((n_sondes,), np.nan)
			for kk, idl_ti in enumerate(idl_time_idx):
				if idl_ti.size > 0:
					idl_time_idx[kk] = idl_ti[0]
					idl_tb_sel[kk,:] = idl_tb_dict['tb'][idl_ti[0],:]
					idl_time_sel[kk] = idl_tb_dict['time'][idl_ti[0]]
					idl_lwp_sel[kk] = idl_tb_dict['lwp'][idl_ti[0]]
				else:
					idl_time_idx[kk] = 999999
					idl_lwp_sel[kk] = -99

			idl_time_idx = idl_time_idx.astype(np.int32)

			idl_tb_dict['time'] = idl_time_sel
			idl_tb_dict['tb'] = idl_tb_sel
			idl_tb_dict['lwp'] = idl_lwp_sel
			

			# # # # # COMMENT (UNCOMMENT) THIS FOR LOOP IF plot_for_calibration_times is True (False) # # # # #
			# # # # # and don't forget to indent the block below the for loop accordingly			  # # # # #
			# for ct_idx, ct in enumerate(mwr.calibration_periods):
			
				# # limit sonde launches to current to next calib. period:						 # # # # #
				# sc_idx = np.where((FWD_SIM_DS.time >= ct[0]) & (FWD_SIM_DS.time < ct[1]))[0]	 # # # # #
				# if len(sc_idx) == 0: continue													 # # # # #
				# # # # # # # # # # # # # sc_idl_idx = np.where((idl_tb_dict['time'] >= ct[0]) & (idl_tb_dict['time'] < ct[1]))[0] # # # # #
				
				# following time dependent values exist: FWD_SIM_DS.time, FWD_SIM_DS.tb, FWD_SIM_DS.tb_original
				# mwr_dict['TB'][:,jj], time_dt (COUPLED TO FWD_SIM_DS.time)

			if plot_for_calibration_times:
				plot_path = (path_plot_chosen + all_out_folder[all_out_folder.find("test_"):] + "idl_vs_pamtra/" +
							"cal_period_%02i/"%ct_idx)
			else:
				plot_path = path_plot_chosen + all_out_folder[all_out_folder.find("test_"):] + "idl_vs_pamtra/"
			# check if plot_path folder exists. If it doesn't, create it.
			plot_path_dir = os.path.dirname(plot_path)
			if not os.path.exists(plot_path_dir):
				os.makedirs(plot_path_dir)

			# Use a slice of the FWD_SIM_DS and idl_tb_dict:
			if plot_for_calibration_times:
				FWD_SIM_DS_slice = FWD_SIM_DS.isel(grid_x=sc_idx)	# time is saved on dimension grid_x
				
				idl_tb_dict_slice = copy.deepcopy(idl_tb_dict)
				idl_tb_dict_slice['time'] = idl_tb_dict_slice['time'][sc_idx]
				idl_tb_dict_slice['lwp'] = idl_tb_dict_slice['lwp'][sc_idx]
				idl_tb_dict_slice['tb'] = idl_tb_dict_slice['tb'][sc_idx,:]

				sonde.iwv_slice = sonde.iwv[sc_idx]

				# check if sonde.launch_time is equal to FWD_SIM_DS time:
				assert np.nanmax(np.abs(sonde.launch_time[sc_idx] - FWD_SIM_DS_slice.time.values)) == 0

			else:	# just use the entire datasets in this case
				FWD_SIM_DS_slice = FWD_SIM_DS
				idl_tb_dict_slice = copy.deepcopy(idl_tb_dict)

				sonde.iwv_slice = sonde.iwv

				# check if sonde.launch_time is equal to FWD_SIM_DS time:
				assert np.nanmax(np.abs(sonde.launch_time - FWD_SIM_DS_slice.time.values)) == 0


			# Scatter plot and TB difference time series (for each frequency):
			time_dt = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in FWD_SIM_DS_slice.time.values])
			for jj, frq in enumerate(FWD_SIM_DS_slice.frequency.values):
				print(frq)
				fig1 = plt.figure(figsize=(22,10))
				ax0 = plt.subplot2grid((1,3), (0,0), colspan=1)
				ax1 = plt.subplot2grid((1,3), (0,1), colspan=2)


				# reduce cases to LWP = 0:
				where_no_lwp = np.where(idl_tb_dict_slice['lwp'] == 0)[0]


				# Compute statistics for scatterplot:
				sc_N = np.count_nonzero(~np.isnan(FWD_SIM_DS_slice.tb[where_no_lwp,jj]) &
									~np.isnan(idl_tb_dict_slice['tb'][where_no_lwp,jj]))
				sc_bias = np.nanmean(idl_tb_dict_slice['tb'][where_no_lwp,jj] - FWD_SIM_DS_slice.tb[where_no_lwp,jj])
				sc_rmse = np.sqrt(np.nanmean((np.abs(FWD_SIM_DS_slice.tb[where_no_lwp,jj] - idl_tb_dict_slice['tb'][where_no_lwp,jj]))**2))
				where_nonnan = np.argwhere(~np.isnan(idl_tb_dict_slice['tb'][where_no_lwp,jj]) & ~np.isnan(FWD_SIM_DS_slice.tb.values[where_no_lwp,jj])).flatten()
					# -> must be used to ignore nans in corrcoef
				sc_R = np.corrcoef(FWD_SIM_DS_slice.tb[where_no_lwp,jj][where_nonnan], idl_tb_dict_slice['tb'][where_no_lwp,jj][where_nonnan])[0,1]

					
				# ------------------------------------- #


				# Colorbar with discrete levels:
				if color_code_iwv:
					cmap = mpl.cm.plasma
					bounds = np.arange(0, 31, 2.5).tolist()
					normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)
			
					ax0_scatter = ax0.scatter(FWD_SIM_DS_slice.tb[where_no_lwp,jj], idl_tb_dict_slice['tb'][where_no_lwp,jj], c=sonde.iwv_slice[where_no_lwp],
												s=25.0, cmap=cmap, norm=normalize, alpha=0.65)

				else:
					ax0.plot(FWD_SIM_DS_slice.tb[where_no_lwp,jj], idl_tb_dict_slice['tb'][where_no_lwp,jj], linestyle='none', marker='+',
								color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


				# diagonal line:
				xlims = np.asarray([np.nanmin(np.array([idl_tb_dict_slice['tb'][:,jj],FWD_SIM_DS_slice.tb.values[:,jj]]))-3,
							np.nanmax(np.array([idl_tb_dict_slice['tb'][:,jj],FWD_SIM_DS_slice.tb.values[:,jj]]))+3])
				ylims = xlims
				ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


				# generate a linear fit with least squares approach: notes, p.2:
				# filter nan values:
				x_fit = FWD_SIM_DS_slice.tb.values[where_no_lwp,jj]
				y_fit = idl_tb_dict_slice['tb'][where_no_lwp,jj]

				mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

				y_fit = y_fit[mask]
				x_fit = x_fit[mask]

				# there must be at least 2 measurements to create a linear fit:
				if (len(y_fit) > 1) and (len(x_fit) > 1):
					# G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
					# m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
					# slope = m_fit[0]
					# offset = m_fit[1]

					# You can use that from above or simply np.polyfit with deg=1:
					slope, offset = np.polyfit(x_fit, y_fit, 1)
					ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))


				# Corrected TBs plotted if following two lines are uncommented
				# ax0.plot(FWD_SIM_DS_slice.tb[:,jj], (idl_tb_dict_slice['tb'][:,jj]-offset)/slope, linestyle='none', marker='+',
								# color=(1,0,0), markeredgecolor=(1,0,0), markersize=5.0, alpha=0.65)


				ax0.set_xlim(left=xlims[0], right=xlims[1])
				ax0.set_ylim(bottom=ylims[0], top=ylims[1])

						# add statistics:
				ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
						np.nanmean(np.concatenate((idl_tb_dict_slice['tb'][where_no_lwp,jj], FWD_SIM_DS_slice.tb.values[where_no_lwp,jj]), axis=0)),
						sc_bias, sc_rmse, sc_R),
						horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

				leg_handles, leg_labels = ax0.get_legend_handles_labels()
				ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

				ax0.set_aspect('equal', 'box')

				ax0.set_title("Simulated TBs (PAMTRA vs. IDL STP),\n%.2f GHz"%frq, fontsize=fs, pad=0)
				ax0.set_xlabel("TB$_{\mathrm{PAMTRA}}$ (K)", fontsize=fs, labelpad=0.5)
				ax0.set_ylabel("TB$_{\mathrm{IDL-STP}}$ (K)", fontsize=fs, labelpad=1.0)

				ax0.minorticks_on()
				ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

				ax0.tick_params(axis='both', labelsize=fs-4)


				if color_code_iwv:
					ax1_scatter = ax1.scatter(time_dt[where_no_lwp], idl_tb_dict_slice['tb'][where_no_lwp,jj] - FWD_SIM_DS_slice.tb[where_no_lwp,jj], c=sonde.iwv_slice[where_no_lwp],
												s=36.0, cmap=cmap, norm=normalize, alpha=1.0)

					cb1 = fig1.colorbar(mappable=ax1_scatter, ax=ax1,
										boundaries=bounds, # to get the triangles at the colorbar boundaries
										spacing='proportional',	# try 'proportional'
										ticks=bounds[0::2],			# on is better
										extend='max', fraction=0.075, orientation='vertical', pad=0)
					cb1.set_label(label="IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs-4)	# change fontsize of colorbar label
					cb1.ax.tick_params(labelsize=fs-4)		# change size of colorbar ticks

				else:
					ax1.plot(time_dt[where_no_lwp], idl_tb_dict_slice['tb'][where_no_lwp,jj] - FWD_SIM_DS_slice.tb[where_no_lwp,jj], linestyle='none',
								color=(0,0,0), marker='o', markeredgecolor=(0,0,0), markersize=6.0, alpha=0.65)

				# 0 line:
				if not plot_for_calibration_times:
					ct = [datetime_to_epochtime(dt.datetime.strptime(date_start, "%Y-%m-%d")), 
							datetime_to_epochtime(dt.datetime.strptime(date_end, "%Y-%m-%d"))]
				ct_dt_0 = dt.datetime.utcfromtimestamp(ct[0])
				ct_dt_1 = dt.datetime.utcfromtimestamp(ct[-1])
				ax1.plot([ct_dt_0, ct_dt_1], [0,0], linewidth=1.25, color=(0,0,0))

				ax1.set_xlabel("2019/2020", fontsize=fs, labelpad=0.5)
				ax1.set_ylabel("TB$_{\mathrm{IDL-STP}}$ - TB$_{\mathrm{PAMTRA}}$ (K)", fontsize=fs, labelpad=0)
				ax1.set_title("%s - %s"%(dt.datetime.strftime(ct_dt_0, "%Y-%m-%d"), 
								dt.datetime.strftime(ct_dt_1, "%Y-%m-%d")), fontsize=fs, pad=0)
				ax1.set_ylim(bottom=-10, top=10)
				ax1.set_xlim(left=ct_dt_0, right=ct_dt_1)

				ax1.tick_params(axis='both', labelsize=fs-4)

				fig1.autofmt_xdate()
				ax1.minorticks_on()
				ax1.grid(which='both', axis='y', color=(0.5,0.5,0.5), alpha=0.5)


				# # LWP:
				# ax2 = ax1.twinx()
				# ax2.plot(time_dt[where_no_lwp], idl_tb_dict_slice['lwp'][where_no_lwp], color=(0,0,0), marker='o',
							# markeredgecolor=(0,0,0), markersize=6.0)
				# ax2.set_xlabel("LWP (kg$\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=6)
				# ax2.set_ylim(bottom=0, top=0.08)


				if save_figures:
					name_base = "TB_comparison_sim_obs_clear_sky_%05i_cGHz"%(frq*100)
					fig1.savefig(plot_path + name_base + ".png", dpi=400)
				else:
					plt.show()
				# plt.close()
				1/0




		# # # # # COMMENT (UNCOMMENT) THIS FOR LOOP IF plot_for_calibration_times is True (False) # # # # #
		# # # # # and don't forget to indent the block below the for loop accordingly			  # # # # #
		for ct_idx, ct in enumerate(mwr.calibration_periods):
		
			# limit sonde launches to current to next calib. period:						 # # # # #
			sc_idx = np.where((FWD_SIM_DS.time >= ct[0]) & (FWD_SIM_DS.time < ct[1]))[0]	 # # # # #
			if len(sc_idx) == 0: continue													 # # # # #
			# following time dependent values exist: FWD_SIM_DS.time, FWD_SIM_DS.tb, FWD_SIM_DS.tb_original
			# mwr_dict['TB'][:,jj], time_dt (COUPLED TO FWD_SIM_DS.time)

			if plot_for_calibration_times:
				plot_path = path_plot_chosen + all_out_folder[all_out_folder.find("test_"):] + "cal_period_%02i/"%ct_idx
			else:
				plot_path = path_plot_chosen + all_out_folder[all_out_folder.find("test_"):]
			# check if plot_path folder exists. If it doesn't, create it.
			plot_path_dir = os.path.dirname(plot_path)
			if not os.path.exists(plot_path_dir):
				os.makedirs(plot_path_dir)


			# Use a slice of the FWD_SIM_DS and mwr_dict:
			if plot_for_calibration_times:
				FWD_SIM_DS_slice = FWD_SIM_DS.isel(grid_x=sc_idx)	# time is saved on dimension grid_x
				
				mwr_dict_slice = copy.deepcopy(mwr_dict)
				mwr_dict_slice['time'] = mwr_dict_slice['time'][sc_idx]
				mwr_dict_slice['TB'] = mwr_dict_slice['TB'][sc_idx,:]

				sonde.iwv_slice = sonde.iwv[sc_idx]

				# check if sonde.launch_time is equal to FWD_SIM_DS time:
				assert np.nanmax(np.abs(sonde.launch_time[sc_idx] - FWD_SIM_DS_slice.time.values)) == 0

			else:	# just use the entire datasets in this case
				FWD_SIM_DS_slice = FWD_SIM_DS
				mwr_dict_slice = copy.deepcopy(mwr_dict)

				sonde.iwv_slice = sonde.iwv

				# check if sonde.launch_time is equal to FWD_SIM_DS time:
				assert np.nanmax(np.abs(sonde.launch_time - FWD_SIM_DS_slice.time.values)) == 0


			# Scatter plot and TB difference time series (for each frequency):
			time_dt = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in FWD_SIM_DS_slice.time.values])
			for jj, frq in enumerate(FWD_SIM_DS_slice.frequency.values):
				print(frq)
				fig1 = plt.figure(figsize=(22,10))
				ax0 = plt.subplot2grid((1,3), (0,0), colspan=1)
				ax1 = plt.subplot2grid((1,3), (0,1), colspan=2)


				if plot_iwv_x_TB_diff:	# no statistics computed... directly jump to plotting

					ax0.plot(sonde.iwv_slice, mwr_dict_slice['TB'][:,jj] - FWD_SIM_DS_slice.tb[:,jj], linestyle='none', marker='o',
								color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


					# diagonal line:
					xlims = np.asarray([0, np.nanmax(sonde.iwv_slice)])
					ylims = np.asarray([-10, 10])
					ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


					# generate a linear fit with least squares approach: notes, p.2:
					# filter nan values:
					x_fit = sonde.iwv_slice
					y_fit = mwr_dict_slice['TB'][:,jj] - FWD_SIM_DS_slice.tb.values[:,jj]

					mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

					y_fit = y_fit[mask]
					x_fit = x_fit[mask]

					# there must be at least 2 measurements to create a linear fit:
					if (len(y_fit) > 1) and (len(x_fit) > 1):
						# G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
						# m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
						# slope = m_fit[0]
						# offset = m_fit[1]

						# You can use that from above or simply np.polyfit with deg=1:
						slope, offset = np.polyfit(x_fit, y_fit, 1)

						if save_offset_slope:
							slope_save[ct_idx,jj] = slope
							offset_save[ct_idx,jj] = offset

						ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, 
											label="Best fit: y = %.2fx + %.2f"%(slope,offset))


					ax0.set_xlim(left=xlims[0], right=xlims[1])
					ax0.set_ylim(bottom=ylims[0], top=ylims[1])

					leg_handles, leg_labels = ax0.get_legend_handles_labels()
					ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

					ax0.set_aspect(1.0/ax0.get_data_ratio(), 'box')

					ax0.set_title("IWV against observed (obs) - simulated (sim) TBs,\n%.2f GHz"%frq, fontsize=fs, pad=0)
					ax0.set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
					ax0.set_ylabel("TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", fontsize=fs, labelpad=1.0)

					ax0.minorticks_on()
					ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

					ax0.tick_params(axis='both', labelsize=fs-4)


				else:
					# Compute statistics for scatterplot:
					sc_N = np.count_nonzero(~np.isnan(FWD_SIM_DS_slice.tb[:,jj]) &
										~np.isnan(mwr_dict_slice['TB'][:,jj]))
					sc_bias = np.nanmean(mwr_dict_slice['TB'][:,jj] - FWD_SIM_DS_slice.tb[:,jj])
					sc_rmse = np.sqrt(np.nanmean((np.abs(FWD_SIM_DS_slice.tb[:,jj] - mwr_dict_slice['TB'][:,jj]))**2))
					where_nonnan = np.argwhere(~np.isnan(mwr_dict_slice['TB'][:,jj]) & ~np.isnan(FWD_SIM_DS_slice.tb.values[:,jj])).flatten()
						# -> must be used to ignore nans in corrcoef
					sc_R = np.corrcoef(FWD_SIM_DS_slice.tb[:,jj][where_nonnan], mwr_dict_slice['TB'][:,jj][where_nonnan])[0,1]

					# save bias values and number of samples:
					if save_offset_slope:
						bias_save[ct_idx,jj] = sc_bias
						n_samp[ct_idx] = sc_N
						
					# ------------------------------------- #


					# Colorbar with discrete levels:
					if color_code_iwv:
						cmap = mpl.cm.plasma
						bounds = np.arange(0, 31, 2.5).tolist()
						normalize = mpl.colors.BoundaryNorm(bounds, cmap.N)
				
						ax0_scatter = ax0.scatter(FWD_SIM_DS_slice.tb[:,jj], mwr_dict_slice['TB'][:,jj], c=sonde.iwv_slice,
													s=25.0, cmap=cmap, norm=normalize, alpha=0.65)

					else:
						ax0.plot(FWD_SIM_DS_slice.tb[:,jj], mwr_dict_slice['TB'][:,jj], linestyle='none', marker='+',
									color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


					# diagonal line:
					xlims = np.asarray([np.nanmin(np.array([mwr_dict_slice['TB'][:,jj],FWD_SIM_DS_slice.tb.values[:,jj]]))-3,
								np.nanmax(np.array([mwr_dict_slice['TB'][:,jj],FWD_SIM_DS_slice.tb.values[:,jj]]))+3])
					ylims = xlims
					ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


					# generate a linear fit with least squares approach: notes, p.2:
					# filter nan values:
					x_fit = FWD_SIM_DS_slice.tb.values[:,jj]
					y_fit = mwr_dict_slice['TB'][:,jj]

					mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

					y_fit = y_fit[mask]
					x_fit = x_fit[mask]

					# there must be at least 2 measurements to create a linear fit:
					if (len(y_fit) > 1) and (len(x_fit) > 1):
						# G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
						# m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
						# slope = m_fit[0]
						# offset = m_fit[1]

						# You can use that from above or simply np.polyfit with deg=1:
						slope, offset = np.polyfit(x_fit, y_fit, 1)

						if save_offset_slope:
							slope_save[ct_idx,jj] = slope
							offset_save[ct_idx,jj] = offset

						ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))


					# Corrected TBs plotted if following two lines are uncommented
					# ax0.plot(FWD_SIM_DS_slice.tb[:,jj], (mwr_dict_slice['TB'][:,jj]-offset)/slope, linestyle='none', marker='+',
									# color=(1,0,0), markeredgecolor=(1,0,0), markersize=5.0, alpha=0.65)


					ax0.set_xlim(left=xlims[0], right=xlims[1])
					ax0.set_ylim(bottom=ylims[0], top=ylims[1])

							# add statistics:
					ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
							np.nanmean(np.concatenate((mwr_dict_slice['TB'][:,jj], FWD_SIM_DS_slice.tb.values[:,jj]), axis=0)),
							sc_bias, sc_rmse, sc_R),
							horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

					leg_handles, leg_labels = ax0.get_legend_handles_labels()
					ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

					ax0.set_aspect('equal', 'box')

					ax0.set_title("Simulated (sim) vs. observed (obs) TBs,\n%.2f GHz"%frq, fontsize=fs, pad=0)
					ax0.set_xlabel("TB$_{\mathrm{sim}}$ (K)", fontsize=fs, labelpad=0.5)
					ax0.set_ylabel("TB$_{\mathrm{obs}}$ (K)", fontsize=fs, labelpad=1.0)

					ax0.minorticks_on()
					ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

					ax0.tick_params(axis='both', labelsize=fs-4)



				if color_code_iwv:
					ax1_scatter = ax1.scatter(time_dt, mwr_dict_slice['TB'][:,jj] - FWD_SIM_DS_slice.tb[:,jj], c=sonde.iwv_slice,
												s=36.0, cmap=cmap, norm=normalize, alpha=1.0)

					cb1 = fig1.colorbar(mappable=ax1_scatter, ax=ax1,
										boundaries=bounds, # to get the triangles at the colorbar boundaries
										spacing='proportional',	# try 'proportional'
										ticks=bounds[0::2],			# on is better
										extend='max', fraction=0.075, orientation='vertical', pad=0)
					cb1.set_label(label="IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs-4)	# change fontsize of colorbar label
					cb1.ax.tick_params(labelsize=fs-4)		# change size of colorbar ticks

				else:
					ax1.plot(time_dt, mwr_dict_slice['TB'][:,jj] - FWD_SIM_DS_slice.tb[:,jj], linestyle='none',
								color=(0,0,0), marker='o', markeredgecolor=(0,0,0), markersize=6.0, alpha=0.65)

				# 0 line:
				ct_dt_0 = dt.datetime.utcfromtimestamp(ct[0])
				ct_dt_1 = dt.datetime.utcfromtimestamp(ct[-1])
				ax1.plot([ct_dt_0, ct_dt_1], [0,0], linewidth=1.25, color=(0,0,0))

				ax1.set_xlabel("2019/2020", fontsize=fs, labelpad=0.5)
				ax1.set_ylabel("TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", fontsize=fs, labelpad=0)
				ax1.set_title("%s - %s"%(dt.datetime.strftime(ct_dt_0, "%Y-%m-%d"), 
								dt.datetime.strftime(ct_dt_1, "%Y-%m-%d")), fontsize=fs, pad=0)
				ax1.set_ylim(bottom=-10, top=10)
				ax1.set_xlim(left=ct_dt_0, right=ct_dt_1)

				ax1.tick_params(axis='both', labelsize=fs-4)

				fig1.autofmt_xdate()
				ax1.minorticks_on()
				ax1.grid(which='both', axis='y', color=(0.5,0.5,0.5), alpha=0.5)

				if save_figures:
					name_base = "TB_comparison_sim_obs_clear_sky_%05i_cGHz"%(frq*100)
					fig1.savefig(plot_path + name_base + ".png", dpi=400)
				else:
					plt.show()
				# plt.close()


		# Save offset and slope of the observed TB for the current instrument,
		# current calibration period and all frequencies
		if save_offset_slope and plot_for_calibration_times:
			OFFSET_SLOPE_DS = xr.Dataset({
						'slope': 	(['time', 'frequency'], slope_save,
									{'description': "Slope of the linear fit: TB_obs = slope*TB_sim + offset",
									'units': ""}),
						'offset': 	(['time', 'frequency'], offset_save,
									{'description': "Offset of the linear fit: TB_obs = slope*TB_sim + offset",
									'units': "K"}),
						'bias':		(['time', 'frequency'], bias_save,
									{'description': ("Bias of observed TBs with respect to simulated TBs. " +
													"Bias = mean(TB_obs - TB_sim, dim='time')"),
									'units': "K"}),
						'n_samp':	(['time'], n_samp,
									{'description': ("Number of samples that were used to compute bias, slope " +
													"and offset for this calibration period."),
									'units': ""})
									},
						coords=		{'frequency': (['frequency'], FWD_SIM_DS.frequency,
										{'description': "Microwave radiometer channel frequency",
										'units': "GHz"}),
									'calibration_period_start': (['time'], np.asarray(mwr.calibration_periods)[:,0],
										{'description': "Start of calibration period or start of MOSAiC campaign.",
										'units': "seconds since 1970-01-01 00:00:00 UTC"}),
									'calibration_period_end': (['time'], np.asarray(mwr.calibration_periods)[:,1],
										{'description': "End of calibration period or end of MOSAiC campaign.",
										'units': "seconds since 1970-01-01 00:00:00 UTC"})
									})

			# add some attributes:
			OFFSET_SLOPE_DS.attrs['description'] = ("Offset and slope of a linear fit between observed (obs) and simulated (sim)" +
													"brightness temperatures (TBs) for each frequency and calibration " +
													"period of the instrument. Linear fit: TB_obs = slope*TB_sim + offset")
			OFFSET_SLOPE_DS.attrs['HOW_TO_APPLY'] = ("To correct the measured TBs of your chosen radiometer, compute the following: " +
													"TB_corrected = (TB_obs - offset)/slope")
			OFFSET_SLOPE_DS.attrs['HOW_TO_APPLY_BIAS'] = ("In case you just want to correct a bias independent of TBs, use: " +
															"TB_corrected = (TB_obs - bias)")
			OFFSET_SLOPE_DS.attrs['instrument'] = freq_label
			OFFSET_SLOPE_DS.attrs['used_forward_model'] = "PAMTRA, DOI: 10.5194/gmd-13-4229-2020"
			OFFSET_SLOPE_DS.attrs['add_info'] = ("Simulations are based on MOSAiC radiosonde, version '%s'. Zenith "%radiosonde_version +
												"looking geometry. Assumed height (in m) of instrument above MSL: 10")
			OFFSET_SLOPE_DS.attrs['radiosonde_info'] = ("MOSAiC Leg 1: https://doi.pangaea.de/10.1594/PANGAEA.928651, " +
														"MOSAiC Leg 2: https://doi.pangaea.de/10.1594/PANGAEA.928659, " +
														"MOSAiC Leg 3: https://doi.pangaea.de/10.1594/PANGAEA.928669, " +
														"MOSAiC Leg 4: https://doi.pangaea.de/10.1594/PANGAEA.928715, " +
														"MOSAiC Leg 5: https://doi.pangaea.de/10.1594/PANGAEA.928735")
			OFFSET_SLOPE_DS.attrs['author'] = "Andreas Walbröl, a.walbroel@uni-koeln.de"
			OFFSET_SLOPE_DS.attrs['history'] = "Created on: %s UTC"%(dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))


			file_save = path_mwr_offset_output + "MOSAiC_%s_radiometer_clear_sky_offset_correction.nc"%(freq_label)
			OFFSET_SLOPE_DS.to_netcdf(file_save, mode='w', format="NETCDF4")
			OFFSET_SLOPE_DS.close()

			# mention how to apply offset and slope on measured MWR TB to get the corrected values

			# save with instrument name

print("Done")