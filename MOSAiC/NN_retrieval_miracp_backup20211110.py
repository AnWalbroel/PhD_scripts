import pdb
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import PercentFormatter
from import_data import (import_mirac_BRT_RPG_daterange, import_radiosonde_daterange, import_mirac_IWV_LWP_RPG_daterange, import_mirac_MET_RPG_daterange,
							import_PS_mastertrack)
from my_classes import radiosondes, radiometers
from data_tools import compute_retrieval_statistics, compute_DOY, select_MWR_channels, outliers_per_eye

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow

from sklearn.model_selection import KFold

ssstart = dt.datetime.utcnow()


def load_geoinfo_MOSAiC_polarstern():

	"""
	Load Polarstern track information (lat, lon).
	"""

	# Data paths:
	path_hatpro_level1 = "/data/obs/campaigns/mosaic/hatpro/l1/"
	path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"
	path_ps_track = "/data/obs/campaigns/mosaic/polarstern_track/"

	# Specify date range:
	date_start = "2019-09-01"
	date_end = "2020-10-12"			# default: "2020-10-12"

	# Import and concatenate polarstern track data: Cycle through all PS track files
	# and concatenate them
	ps_track_files = sorted(glob.glob(path_ps_track + "*.nc"))
	ps_track_dict = {'lat': np.array([]), 'lon': np.array([]), 'time': np.array([])}
	ps_keys = ['Latitude', 'Longitude', 'time']
	for pf_file in ps_track_files:
		ps_track_dict_temp = import_PS_mastertrack(pf_file, ps_keys)
		
		# concatenate ps_track_dict_temp data:
		ps_track_dict['lat'] = np.concatenate((ps_track_dict['lat'], ps_track_dict_temp['Latitude']), axis=0)
		ps_track_dict['lon'] = np.concatenate((ps_track_dict['lon'], ps_track_dict_temp['Longitude']), axis=0)
		ps_track_dict['time'] = np.concatenate((ps_track_dict['time'], ps_track_dict_temp['time']), axis=0)

	# sort by time just to make sure we have ascending time:
	ps_sort_idx = np.argsort(ps_track_dict['time'])
	for key in ps_track_dict.keys(): ps_track_dict[key] = ps_track_dict[key][ps_sort_idx]

	return ps_track_dict


def save_obs_predictions(
	prediction,
	mwr_dict,
	error_dict_syn,
	aux_info):

	"""
	Save the Neural Network prediction to a netCDF file. Variables to be included:
	time, flag, output variable (prediction), standard error (std. dev. (bias corrected!),
	lat, lon, zsl (altitude above mean sea level),

	Parameters:
	-----------
	prediction : array of floats
		Array that contains the predicted output and is to be saved.
	mwr_dict : dict
		Dictionary containing data of the MiRAC-P.
	error_dict : dict
		Dictionary containing information about the errors (RMSE, bias).
	aux_info : dict
		Dictionary containing additional information about the NN.
	"""

	# Set the flag values:
	mwr_dict['RF'] = outliers_per_eye(mwr_dict['RF'], mwr_dict['time'], instrument='mirac',
										filename="/net/blanc/awalbroe/Codes/MOSAiC/MiRAC-P_outliers.txt")
	mwr_dict['RF'][mwr_dict['RF'] == 99.0] = 1.0


	# Add geoinfo data: Load it, interpolate it on the MWR time axis, set the right attribute (source of
	# information), 
	ps_track_dict = load_geoinfo_MOSAiC_polarstern()

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(aux_info['date_start'], "%Y-%m-%d")
	date_end = dt.datetime.strptime(aux_info['date_end'], "%Y-%m-%d")

	# MOSAiC Legs to identify the correct Polarstern Track file:
	MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}

	# Add source of Polarstern track information as global attribute:
	source_PS_track = {'leg1': "Rex, Markus (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/1, Tromsø - Arctic Ocean, 2019-09-20 - 2019-12-13 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924668",
						'leg2': "Haas, Christian (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/2, Arctic Ocean - Arctic Ocean, 2019-12-13 - 2020-02-24 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924674",
						'leg3': "Kanzow, Torsten (2020): Links to master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/3, Arctic Ocean - Longyearbyen, 2020-02-24 - 2020-06-04 " +
								"(Version 2). Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924681",
						'leg4': "Rex, Markus (2021): Master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/4, Longyearbyen - Arctic Ocean, 2020-06-04 - 2020-08-12. " +
								"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.926829",
						'leg5': "Rex, Markus (2021): Master tracks in different resolutions of " +
								"POLARSTERN cruise PS122/5, Arctic Ocean - Bremerhaven, 2020-08-12 - 2020-10-12. " +
								"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, " +
								"Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.926910"}

	# set the global attribute:
	if (date_start >= MOSAiC_legs['leg1'][0]) and (date_end < MOSAiC_legs['leg1'][1]):
		globat = source_PS_track['leg1']
	elif (date_start >= MOSAiC_legs['leg2'][0]) and (date_end < MOSAiC_legs['leg2'][1]):
		globat = source_PS_track['leg2']
	elif (date_start >= MOSAiC_legs['leg3'][0]) and (date_end < MOSAiC_legs['leg3'][1]):
		globat = source_PS_track['leg3']
	elif (date_start >= MOSAiC_legs['leg4'][0]) and (date_end < MOSAiC_legs['leg4'][1]):
		globat = source_PS_track['leg4']
	elif (date_start >= MOSAiC_legs['leg5'][0]) and (date_end <= MOSAiC_legs['leg5'][1]):
		globat = source_PS_track['leg5']

	# interpolate Polarstern track data on mwr data time axis:
	for ps_key in ['lat', 'lon', 'time']:
		ps_track_dict[ps_key + "_ip"] = np.interp(np.rint(mwr_dict['time']), ps_track_dict['time'], ps_track_dict[ps_key])
	

	# Save predictions to xarray dataset, then to netcdf:
	nc_output_name = f"MOSAiC_mirac-p_l2_iwv_{aux_info['date_start'].replace('-', '')}-{aux_info['date_end'].replace('-', '')}"


	# create Dataset:
	if aux_info['predictand'] == 'iwv':
		DS = xr.Dataset({'prw':			(['time'], prediction.flatten(),
										{'description': "Retrieved integrated water vapour (iwv) or precipitable water (prw)",
										'standard_name': "atmosphere_mass_content_of_water_vapor",
										'units': "kg m-2"}),
						'flag':			(['time'], mwr_dict['RF'],
										{'description': "Quality flags of the MiRAC-P data. 0: data is okay, 1: flagged manually (visual inspection)",
										'standard_name': "quality_flag"}),
						'prw_err':		([], error_dict_syn['stddev'],
										{'description': "Standard deviation of retrieved prw (iwv)",
										'comments': "Values can be large due to misfit for atmosphere mass content of water vapour (prw) > 10 kg m-2",
										'units': "kg m-2"}),
						'latitude':		(['time'], ps_track_dict['lat_ip'],
										{'description': "Latitude of the instrument's location (onboard Polarstern)",
										'standard_name': "latitude",
										'comments': f"Source: {globat}",
										'units': "degree_north"}),
						'longitude':	(['time'], ps_track_dict['lon_ip'],
										{'description': "Longitude of the instrument's location (onboard Polarstern)",
										'standard_name': "longitude",
										'comments': f"Source: {globat}",
										'units': "degree_east"}),
						'zh':			(['time'], np.full_like(mwr_dict['time'], 15.0),
										{'description': "Height above mean sea level",
										'standard_name': "height",
										'units': "m"})},
						coords=			{'time': (['time'], mwr_dict['time'].astype(np.int64),
													{'description': "Time in seconds since 1970-01-01 00:00:00 UTC",
													'standard_name': "time",
													'units': "seconds since 1970-01-01 00:00:00 UTC"})})

		DS.attrs['retrieval_type'] = "Neural Networks (Tensorflow V2.5.0, Keras V2.5.0), Python Version: 3.8.10"
		DS.attrs['retrieval_batch_size'] = f"{str(aux_info['batch_size'])}"
		DS.attrs['retrieval_epochs'] = f"{str(aux_info['epochs'])}"
		DS.attrs['retrieval_learning_rate'] = f"{str(aux_info['learning_rate'])}"
		DS.attrs['retrieval_activation_function'] = f"{aux_info['activation']} (from input to hidden layer)"
		DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_info['feature_range'])}"
		DS.attrs['retrieval_hidden_layers_nodes'] = f"1: 32 (kernel_initializer={aux_info['kernel_init']})"
		DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
		DS.attrs['retrieval_callbacks'] = "EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)"

	if site == 'pol':
		DS.attrs['training_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
		if aux_info['nya_test_data']:
			DS.attrs['test_data'] = "Ny Alesund radiosondes 2006-2017"
		else:
			DS.attrs['test_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
	elif site == 'nya':
		DS.attrs['training_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
		DS.attrs['test_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
	DS.attrs['n_training'] = aux_info['n_training']
	DS.attrs['n_test'] = aux_info['n_test']

	DS.attrs['input_vector'] = ("(TB_183.31+/-0.6, TB_183.31+/-1.5, TB_183.31+/-2.5, TB_183.31+/-3.5, TB_183.31+/-5.0, " +
											"TB_183.31+/-7.5, TB_243.00, TB_340.00")
	if 'pres_sfc' in aux_info['predictors']:
		DS.attrs['input_vector'] = DS.input_vector + ", pres_sfc"

	if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
		DS.attrs['input_vector'] = DS.input_vector + ", cos(DayOfYear), sin(DayOfYear)"
	DS.attrs['input_vector'] = DS.input_vector + ")"
	DS.attrs['training_test_TB_noise_std_dev'] = ("TB_183.31+/-0.6: 0.75, TB_183.31+/-1.5: 0.75, TB_183.31+/-2.5: 0.75, " +
											"TB_183.31+/-3.5: 0.75, TB_183.31+/-5.0: 0.75, " +
											"TB_183.31+/-7.5: 0.75, TB_243.00: 4.2, TB_340.00: 4.5")
	DS.attrs['output_vector'] = "(IWV)"

	DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
	datetime_utc = dt.datetime.utcnow()
	DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

	# encode time:
	encoding = {'time': dict()}
	encoding['time']['dtype'] = 'int64'
	encoding['time']['units'] = 'seconds since 1970-01-01 00:00:00'

	pdb.set_trace()

	DS.to_netcdf(path_output + nc_output_name + ".nc", mode='w', format="NETCDF4", encoding=encoding)
	DS.close()


def simple_quality_control(predictand_training, predictand_test, aux_info):

	"""
	Quality control of the data: See RPG Software Manual 
	(RPG_MWR_STD_Software_Manual_G5_2021.pdf)

	Parameters:
	predictand_training : radiometer class
		Contains information about the training data predictand.
	predictand_test : radiometer class
		Contains information about the test data predictand.
	aux_info : dict
		Contains additional information.
	"""

	height_dif_training = np.diff(predictand_training.height, axis=1)
	height_dif_test = np.diff(predictand_test.height, axis=1)
	pres_dif_training = np.diff(predictand_training.pres, axis=1)
	pres_dif_test = np.diff(predictand_test.pres, axis=1)

	# check if height increases in subsequent levels, pressure decreases with height,
	# temp in [190, 330], pres_sfc > 50000, pres in [1, 107000], height in [-200, 70000],
	# temp and pres information at least up to 10 km; hum. information up to -30 deg C,
	# n_levels >= 10
	# (assert is split into many parts to more easily identify broken variables)
	# Broken temp, pres, height, or humidity values might cause the computed IWV to be
	# erroneour
	assert ((np.all(height_dif_training > 0)) and (np.all(height_dif_test > 0)) and 
			(np.all(pres_dif_training < 0)) and (np.all(pres_dif_test < 0)))
	assert ((np.all(predictand_training.temp <= 330)) and (np.all(predictand_training.temp >= 190)) 
			and (np.all(predictand_test.temp <= 330)) and (np.all(predictand_test.temp >= 190)))
	assert ((np.all(predictand_training.pres[:,0] > 50000)) and (np.all(predictand_test.pres[:,0] > 50000)) 
			and (np.all(predictand_training.pres > 1)) and (np.all(predictand_training.pres < 107000)) 
			and (np.all(predictand_test.pres > 1)) and (np.all(predictand_test.pres < 107000)))
	assert ((np.all(predictand_training.height[:,0] > -200)) and (np.all(predictand_training.height[:,-1] < 70000)) 
			and (np.all(predictand_test.height[:,0] > -200)) and (np.all(predictand_test.height[:,-1] < 70000)))
	assert (predictand_training.height.shape[1] >= 10) and (predictand_test.height.shape[1] >= 10)

	# on a regular grid, it's simple to check if temp and pres information exist up to 10 km height:
	idx_10km_train = np.where(predictand_training.height[0,:] >= 10000)[0]
	idx_10km_test = np.where(predictand_test.height[0,:] >= 10000)[0]

	for k in range(aux_info['n_training']): 
		assert ((np.any(~np.isnan(predictand_training.temp[k,idx_10km_train]))) and 
				(np.any(~np.isnan(predictand_training.pres[k,idx_10km_train]))))

		# check if hum. information available up to -30 deg C:
		idx_243K = np.where(predictand_training.temp[k,:] <= 243.15)[0]
		assert np.any(~np.isnan(predictand_training.rh[k,idx_243K]))

	for k in range(aux_info['n_test']): 
		assert ((np.any(~np.isnan(predictand_test.temp[k,idx_10km_test]))) and 
				(np.any(~np.isnan(predictand_test.pres[k,idx_10km_test]))))

		# check if hum. information available up to -30 deg C:
		idx_243K = np.where(predictand_test.temp[k,:] <= 243.15)[0]
		assert np.any(~np.isnan(predictand_test.rh[k,idx_243K]))


def quick_plot_prediction(prediction, target):
	
	plt.plot(prediction, label='prediction')
	plt.plot(predictand_test.output, color=(0,0,0), label='target')
	plt.legend()
	plt.show()


def plot_DOY_predictand(predictor_training, predictand_training, aux_info, save_figures=True):

	x_stuff_total = np.concatenate((predictor_training.DOY_1, predictor_training.DOY_2), axis=1)
	
	if aux_info['predictand'] == 'iwv':
		y_stuff = predictand_training.iwv
	elif aux_info['predictand'] == 'lwp':
		y_stuff = predictand_training.lwp

	fs = 19

	fig1, ax0 = plt.subplots(1,2)
	fig1.set_size_inches(20,10)
	ax0 = ax0.flatten()

	if aux_info['predictand'] == 'iwv':
		ylims = np.asarray([0, 35])
	elif aux_info['predictand'] == 'lwp':
		ylims = np.asarray([0, 750])
	for k in range(2):

		x_stuff = x_stuff_total[:,k]

			
		# ------------------------------------- #

		ax0[k].plot(x_stuff, y_stuff, linestyle='none', marker='.',
					color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


		# diagonal line:
		xlims = np.asarray([-1, 1])
		ax0[k].plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")

		ax0[k].set_xlim(left=xlims[0], right=xlims[1])
		ax0[k].set_ylim(bottom=ylims[0], top=ylims[1])

		leg_handles, leg_labels = ax0[k].get_legend_handles_labels()
		ax0[k].legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

		ax0[k].set_aspect('equal', 'box')
		ax0[k].set_aspect(1.0/ax0[k].get_data_ratio(), 'box')

		ax0[k].set_title(f"Day of the Year (DOY) vs. {aux_info['predictand']}", fontsize=fs, pad=0.1)
		if k == 0:
			ax0[k].set_xlabel("cos(DOY)", fontsize=fs, labelpad=0.5)
		elif k == 1:
			ax0[k].set_xlabel("sin(DOY)", fontsize=fs, labelpad=0.5)

		if aux_info['predictand'] == 'iwv':
			ax0[k].set_ylabel(f"{aux_info['predictand']} ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)
		elif aux_info['predictand'] == 'lwp':
			ax0[k].set_ylabel(f"{aux_info['predictand']} ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax0[k].minorticks_on()
		ax0[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax0[k].tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		fig1.savefig(path + f"Scatter_DOY_vs_{aux_info['predictand']}_training.png", dpi=400)
	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_pres_sfc_predictand(predictor_training, predictand_training, aux_info, save_figures=True):

	x_stuff = predictor_training.pres_sfc
	if aux_info['predictand'] == 'iwv':
		y_stuff = predictand_training.iwv
	elif aux_info['predictand'] == 'lwp':
		y_stuff = predictand_training.lwp

	fs = 19

	fig1 = plt.figure(figsize=(14.6,6.6))
	ax0 = plt.axes()

		
	# ------------------------------------- #

	ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.',
				color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


	# diagonal line:
	xlims = np.asarray([95000, 107000])
	if aux_info['predictand'] == 'iwv':
		ylims = np.asarray([0, 35])
	elif aux_info['predictand'] == 'lwp':
		ylims = np.asarray([0, 750])

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	ax0.set_aspect('equal', 'box')
	ax0.set_aspect(1.0/ax0.get_data_ratio(), 'box')

	ax0.set_title("Surface pressure (pres$_{\mathrm{sfc}}$) vs. %s"%aux_info['predictand'], fontsize=fs, pad=0.1)
	ax0.set_xlabel("pres$_{\mathrm{sfc}}$ (Pa)", fontsize=fs, labelpad=0.5)
	if aux_info['predictand'] == 'iwv':
		ax0.set_ylabel("%s ($\mathrm{kg}\,\mathrm{m}^{-2}$)"%aux_info['predictand'], fontsize=fs, labelpad=1.0)
	elif aux_info['predictand'] == 'lwp':
		ax0.set_ylabel("%s ($\mathrm{g}\,\mathrm{m}^{-2}$)"%aux_info['predictand'], fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		fig1.savefig(path + f"Scatter_pres_sfc_vs_{aux_info['predictand']}_training.png", dpi=400)
	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_time_series_comparison(prediction_obs, mwr_dict, mirac_dict, sonde_dict, 
						aux_info, save_figures=True):

	fs = 19

	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(16,10)

	if aux_info['predictand'] == 'iwv':
		ax1.plot(mirac_dict['time_dt'][mirac_dict['RF'] == 0], mirac_dict['IWV'][mirac_dict['RF'] == 0], 
					linewidth=0.8, color=(0.2,0.4,0.9))
	elif aux_info['predictand'] == 'lwp':
		ax1.plot(mirac_dict['time_dt'][mirac_dict['RF'] == 0], mirac_dict['LWP'][mirac_dict['RF'] == 0], 
					linewidth=0.8, color=(0.2,0.4,0.9))
	# dummy plot:
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=(0.2,0.4,0.9), label='RPG')

	ax1.plot(mwr_dict['time_dt'], prediction_obs, linewidth=0.9, color=(0,0,0))
	# dummy plot:
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=(0,0,0), label='keras')

	if aux_info['predictand'] == 'iwv':
		ax1.plot(sonde_dict['launch_time_dt'], sonde_dict['iwv'], linestyle='none', marker='.', color=(1,0,0), markersize=6.0, label='sonde')

	lh0, ll0 = ax1.get_legend_handles_labels()
	ax1.legend(handles=lh0, labels=ll0, loc="upper right", fontsize=fs-6)

	# diagonal line and axis limits:
	if aux_info['predictand'] == 'iwv':
		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			ylims_dict = {	'leg1': np.asarray([0,10]),
							'leg2': np.asarray([0,8]),
							'leg3': np.asarray([0,20]),
							'leg4': np.asarray([0,35]),
							'leg5': np.asarray([0,25])}
			ylims = ylims_dict[aux_info['considered_period']]

		elif aux_info['considered_period'] == 'mosaic':
			ylims = np.asarray([0.0, 35.0])

		else:
			ylims = np.asarray([-0.5, 15])
	else:
		ylims = np.asarray([0, 750])
	
	ax1.set_ylim(bottom=ylims[0], top=ylims[1])


	ax1.set_title("Batch_size: %i, Epochs: %i"%(aux_info['batch_size'], aux_info['epochs']), fontsize=fs, pad=0)
	if aux_info['predictand'] == 'iwv':
		ax1.set_ylabel("iwv (mm)", fontsize=fs, labelpad=1.0)
	elif aux_info['predictand'] == 'lwp':
		ax1.set_ylabel("lwp ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)
	ax1.tick_params(axis='both', labelsize=fs-5)
	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Timeseries_%s_BS%i_EPOCHS%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i"%(aux_info['mosaic_leg'])
		elif aux_info['considered_period'] == 'mosaic':
			plotname = plotname + "_MOSAiC"

		if aux_info['include_pres_sfc']:
			plotname = plotname + "_psfc"

		plotname = plotname + "_seed%02i"%aux_info['seed']

		fig1.savefig(path + plotname + ".png", dpi=400)
	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_bias_histogram(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mwr_dict['time'] >= lt) & 
						(mwr_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(prediction_obs[hat])


	### this block is for MiRAC-P retrieved IWV done by RPG ###
	# # find temporal overlaps sonde with MiRAC-P obs:
	# # no absolute value because we want the closest mwr time after radiosonde launch!
	# launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	# hatson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) & 
						# (mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	# n_sondes = len(sonde_dict['launch_time'])

	# y_stuff = np.full((n_sondes,), np.nan)
	# for kk, hat in enumerate(hatson_idx):
		# y_stuff[kk] = np.nanmean(mirac_dict['IWV'][hat])
	###_____________________________________________________###


	# eventually separate into three IWV categories: IWV in [0,5), [5,10), [10,inf)
	mask_bot = ((sonde_dict['iwv'] >= 0) & (sonde_dict['iwv'] < 5))
	mask_mid = ((sonde_dict['iwv'] >= 5) & (sonde_dict['iwv'] < 10))
	mask_top = ((sonde_dict['iwv'] >= 10) & (sonde_dict['iwv'] < 100))
	bias_bot = y_stuff[mask_bot] - sonde_dict['iwv'][mask_bot]
	bias_mid = y_stuff[mask_mid] - sonde_dict['iwv'][mask_mid]
	bias_top = y_stuff[mask_top] - sonde_dict['iwv'][mask_top]
	bias_categorized = [bias_bot, bias_mid, bias_top]
	bias = y_stuff - sonde_dict['iwv']

	n_bias = float(len(bias))
	# weights_bias = np.ones_like(bias) / n_bias
	weights_bias_categorized = [np.ones_like(bias_bot) / n_bias, 
								np.ones_like(bias_mid) / n_bias, 
								np.ones_like(bias_top) / n_bias]


	fs = 19
	c_M = (0,0.729,0.675)

	x_lim = [-6.25, 6.25]			# bias in mm or kg m^-2

	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(16,8)

	# ax1.hist(bias, bins=np.arange(-6.25, 6.26, 0.5), weights=weights_bias, color=c_M, ec=(0,0,0))
	# ax1.hist(bias_categorized, bins=np.arange(-6.25, 6.26, 0.5), density=True, 
				# color=['blue', 'white', 'red'], ec=(0,0,0), stacked=True)
	if aux_info['shall_be_stacked']:
		ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
					color=['blue', 'white', 'red'], ec=(0,0,0), stacked=True,
					label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])
	else:
		ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
					color=['blue', 'white', 'red'], ec=(0,0,0),
					label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])

	ax1.text(0.98, 0.96, "Min = %.2f\n Max = %.2f\n Mean = %.2f\n Median = %.2f"%(np.nanmin(bias),
				np.nanmax(bias), np.nanmean(bias), np.nanmedian(bias)), ha='right', va='top', 
				transform=ax1.transAxes, fontsize=fs-4, bbox=dict(boxstyle='round', ec=(0,0,0),
				fc=(1,1,1), alpha=0.5))

	# legend:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	ax1.set_ylim(bottom=0.0, top=0.75)
	ax1.set_xlim(left=x_lim[0], right=x_lim[1])

	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1.set_xlabel("IWV difference ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0)
	ax1.set_ylabel("Frequency occurrence", fontsize = fs)
	ax1.tick_params(axis='both', labelsize=fs-2)

	ax1.set_title("IWV difference (retrieved - radiosonde)", fontsize=fs, pad=0.4)


	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Bias_hist_%s_BS%i_EPOCHS%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i"%(aux_info['mosaic_leg'])
		elif aux_info['considered_period'] == 'mosaic':
			plotname = plotname + "_MOSAiC"

		if not aux_info['shall_be_stacked']:
			plotname = plotname + "_notstacked"

		if 'pres_sfc' in aux_info['predictors']:
			plotname = plotname + "_psfc"

		if 'fold_no' in aux_info.keys():
			plotname += "%i"%(aux_info['fold_no'])

		plotname = plotname + "_seed%02i"%aux_info['seed']

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()


def scatterplot_comparison(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mwr_dict['time'] >= lt) & 
						(mwr_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	y_std = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(prediction_obs[hat])
		y_std[kk] = np.nanstd(prediction_obs[hat])


	x_stuff = sonde_dict['iwv']


	fs = 19
	c_M = (0,0.729,0.675)

	fig1 = plt.figure(figsize=(14.6,7.0))
	ax0 = plt.axes()

	# Compute statistics for scatterplot:
	stats_dict = compute_retrieval_statistics(x_stuff, y_stuff, compute_stddev=True)

	sc_N = stats_dict['N']
	sc_bias = stats_dict['bias']
	sc_rmse = stats_dict['rmse']
	sc_R = stats_dict['R']

	# also compute rmse and bias for specific ranges only:
	# 'bias': np.nanmean(y_stuff - x_stuff),
	# 'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
	if aux_info['predictand'] == 'iwv':	# in mm
		range_bot = [0,5]
		range_mid = [5,10]
		range_top = [10,100]
	elif aux_info['predictand'] == 'lwp': # in kg m^-2
		range_bot = [0,0.025]
		range_mid = [0.025,0.100]
		range_top = [0.100, 1e+06]
		
	mask_bot = ((x_stuff >= range_bot[0]) & (x_stuff < range_bot[1]))
	mask_mid = ((x_stuff >= range_mid[0]) & (x_stuff < range_mid[1]))
	mask_top = ((x_stuff >= range_top[0]) & (x_stuff < range_top[1]))
	error_dict = {	'rmse_tot': sc_rmse,
					'rmse_bot': np.sqrt(np.nanmean((x_stuff[mask_bot] - y_stuff[mask_bot])**2)),
					'rmse_mid': np.sqrt(np.nanmean((x_stuff[mask_mid] - y_stuff[mask_mid])**2)),
					'rmse_top': np.sqrt(np.nanmean((x_stuff[mask_top] - y_stuff[mask_top])**2)),
					'stddev': 	stats_dict['stddev'],
					'bias_tot': sc_bias,
					'bias_bot': np.nanmean(y_stuff[mask_bot] - x_stuff[mask_bot]),
					'bias_mid': np.nanmean(y_stuff[mask_mid] - x_stuff[mask_mid]),
					'bias_top': np.nanmean(y_stuff[mask_top] - x_stuff[mask_top])}


	# ------------------------------------- #

	ax0.errorbar(x_stuff, y_stuff, yerr=y_std, ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P')


	# diagonal line and axis limits:
	if aux_info['predictand'] == 'iwv':
		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			xlims_dict = {	'leg1': np.asarray([0,10]),
							'leg2': np.asarray([0,8]),
							'leg3': np.asarray([0,20]),
							'leg4': np.asarray([0,35]),
							'leg5': np.asarray([0,25])}
			xlims = xlims_dict[aux_info['considered_period']]

		elif aux_info['considered_period'] == 'mosaic':
			xlims = np.asarray([0.0, 30.0])

		else:
			xlims = np.asarray([0, 30])

	elif aux_info['predictand'] == 'lwp':
		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			xlims_dict = {	'leg1': np.asarray([0,0.75]),
							'leg2': np.asarray([0,0.5]),
							'leg3': np.asarray([0,0.75]),
							'leg4': np.asarray([0,1.0]),
							'leg5': np.asarray([0,1.0])}
			xlims = xlims_dict[aux_info['considered_period']]

		elif aux_info['considered_period'] == 'mosaic':
			xlims = np.asarray([0.0, 0.75])

		else:
			xlims = np.asarray([0, 0.75])

	ylims = xlims
	ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	mask = np.isfinite(x_stuff + y_stuff)		# check for nans and inf.

	y_fit = y_stuff[mask]
	x_fit = x_stuff[mask]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		slope, offset = np.polyfit(x_fit, y_fit, 1)
		ds_fit = ax0.plot(xlims, slope*xlims + offset, color=c_M, linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	# add statistics:
	ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
			np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R),
			horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

	leg_handles, leg_labels = ax0.get_legend_handles_labels()
	ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

	ax0.set_aspect('equal', 'box')

	if aux_info['predictand'] == 'iwv':
		ax0.set_title("Retrieved (pred) vs. observed (radiosonde, obs) IWV", fontsize=fs, pad=15.0)
		ax0.set_xlabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)
	elif aux_info['predictand'] == 'lwp':
		ax0.set_title("Retrieved (pred) vs. observed (radiosonde, obs) LWP", fontsize=fs, pad=15.0)
		ax0.set_xlabel("LWP$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("LWP$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Scatter_%s_BS%i_EPOCHS%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i"%(aux_info['mosaic_leg'])

		elif aux_info['considered_period'] == 'mosaic':
			plotname = plotname + "_MOSAiC"

		if 'pres_sfc' in aux_info['predictors']:
			plotname = plotname + "_psfc"

		if 'fold_no' in aux_info.keys():
			plotname += "%i"%(aux_info['fold_no'])

		plotname = plotname + "_seed%02i"%aux_info['seed']

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()

	return error_dict


def scatterplot_comparison_syn(prediction, predictand_test, aux_info, save_figures=True):

	"""
	Same as scatterplot_comparison but for test data evaluation (prediction of synthetic data).
	"""

	x_stuff = predictand_test.output
	y_stuff = prediction.flatten()

	fs = 19
	c_M = (0,0.729,0.675)

	fig1 = plt.figure(figsize=(14.6,7.0))
	ax0 = plt.axes()

	# Compute statistics for scatterplot:
	stats_dict = compute_retrieval_statistics(x_stuff, y_stuff, compute_stddev=True)

	sc_N = stats_dict['N']
	sc_bias = stats_dict['bias']
	sc_rmse = stats_dict['rmse']
	sc_R = stats_dict['R']

	# also compute rmse and bias for specific IWV ranges only:
	# 'bias': np.nanmean(y_stuff - x_stuff),
	# 'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
	if aux_info['predictand'] == 'iwv':	# in mm
		range_bot = [0,5]
		range_mid = [5,10]
		range_top = [10,100]
	elif aux_info['predictand'] == 'lwp': # in kg m^-2
		range_bot = [0,0.025]
		range_mid = [0.025,0.100]
		range_top = [0.100, 1e+06]
		
	mask_bot = ((x_stuff >= range_bot[0]) & (x_stuff < range_bot[1]))
	mask_mid = ((x_stuff >= range_mid[0]) & (x_stuff < range_mid[1]))
	mask_top = ((x_stuff >= range_top[0]) & (x_stuff < range_top[1]))
	error_dict = {	'rmse_tot': sc_rmse,
					'rmse_bot': np.sqrt(np.nanmean((x_stuff[mask_bot] - y_stuff[mask_bot])**2)),
					'rmse_mid': np.sqrt(np.nanmean((x_stuff[mask_mid] - y_stuff[mask_mid])**2)),
					'rmse_top': np.sqrt(np.nanmean((x_stuff[mask_top] - y_stuff[mask_top])**2)),
					'stddev': 	stats_dict['stddev'],
					'bias_tot': sc_bias,
					'bias_bot': np.nanmean(y_stuff[mask_bot] - x_stuff[mask_bot]),
					'bias_mid': np.nanmean(y_stuff[mask_mid] - x_stuff[mask_mid]),
					'bias_top': np.nanmean(y_stuff[mask_top] - x_stuff[mask_top])}


	# ------------------------------------- #

	ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.', markerfacecolor=c_M, markeredgecolor=(0,0,0),
						markersize=5, label='Test data')


	# diagonal line and axis limits:
	if aux_info['predictand'] == 'iwv':
		xlims = np.asarray([0, 30])
	elif aux_info['predictand'] == 'lwp':
		xlims = np.asarray([0, 0.75])
	ylims = xlims
	ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	mask = np.isfinite(x_stuff + y_stuff)		# check for nans and inf.

	y_fit = y_stuff[mask]
	x_fit = x_stuff[mask]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		slope, offset = np.polyfit(x_fit, y_fit, 1)
		ds_fit = ax0.plot(xlims, slope*xlims + offset, color=c_M, linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	# add statistics:
	ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
			np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R),
			horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

	leg_handles, leg_labels = ax0.get_legend_handles_labels()
	ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

	ax0.set_aspect('equal', 'box')

	if aux_info['predictand'] == 'iwv':
		ax0.set_title("Retrieved (pred) vs. target (tar) IWV", fontsize=fs, pad=15.0)
		ax0.set_xlabel("IWV$_{\mathrm{tar}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)
	elif aux_info['predictand'] == 'lwp':
		ax0.set_title("Retrieved (pred) vs. target (tar) LWP", fontsize=fs, pad=15.0)
		ax0.set_xlabel("LWP$_{\mathrm{tar}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("LWP$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)


	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Scatter_TESTDATA_%s_BS%i_EPOCHS%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if 'pres_sfc' in aux_info['predictors']:
			plotname = plotname + "_psfc"

		if 'fold_no' in aux_info.keys():
			plotname += "%i"%(aux_info['fold_no'])

		plotname = plotname + "_seed%02i"%aux_info['seed']

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()

	return error_dict


def scatterplot_reference(mirac_dict, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) & 
						(mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	y_std = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(mirac_dict['IWV'][hat])
		y_std[kk] = np.nanstd(mirac_dict['IWV'][hat])


	x_stuff = sonde_dict['iwv']


	fs = 19
	c_M = (0,0.729,0.675)

	fig1 = plt.figure(figsize=(14.6,7.0))
	ax0 = plt.axes()

	# Compute statistics for scatterplot:
	stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

	sc_N = stats_dict['N']
	sc_bias = stats_dict['bias']
	sc_rmse = stats_dict['rmse']
	sc_R = stats_dict['R']


	# ------------------------------------- #

	ax0.errorbar(x_stuff, y_stuff, yerr=y_std, ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P RPG')


	# diagonal line and axis limits:
	if aux_info['predictand'] == 'iwv':
		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			xlims_dict = {	'leg1': np.asarray([0,10]),
							'leg2': np.asarray([0,8]),
							'leg3': np.asarray([0,20]),
							'leg4': np.asarray([0,35]),
							'leg5': np.asarray([0,25])}
			xlims = xlims_dict[aux_info['considered_period']]

		elif aux_info['considered_period'] == 'mosaic':
			xlims = np.asarray([0.0, 30.0])

		else:
			xlims = np.asarray([0, 30])

	elif aux_info['predictand'] == 'lwp':
		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			xlims_dict = {	'leg1': np.asarray([0,0.75]),
							'leg2': np.asarray([0,0.5]),
							'leg3': np.asarray([0,0.75]),
							'leg4': np.asarray([0,1.0]),
							'leg5': np.asarray([0,1.0])}
			xlims = xlims_dict[aux_info['considered_period']]

		elif aux_info['considered_period'] == 'mosaic':
			xlims = np.asarray([0.0, 0.75])

		else:
			xlims = np.asarray([0, 0.75])
	ylims = xlims
	ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	mask = np.isfinite(x_stuff + y_stuff)		# check for nans and inf.

	y_fit = y_stuff[mask]
	x_fit = x_stuff[mask]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		slope, offset = np.polyfit(x_fit, y_fit, 1)
		ds_fit = ax0.plot(xlims, slope*xlims + offset, color=c_M, linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	# add statistics:
	ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
			np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R),
			horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

	leg_handles, leg_labels = ax0.get_legend_handles_labels()
	ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

	ax0.set_aspect('equal', 'box')
	if aux_info['predictand'] == 'iwv':
		ax0.set_title("Retrieved (RPG) vs. observed (radiosonde, obs) IWV", fontsize=fs, pad=15.0)
		ax0.set_xlabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{RPG}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)
	if aux_info['predictand'] == 'lwp':
		ax0.set_title("Retrieved (RPG) vs. observed (radiosonde, obs) LWP", fontsize=fs, pad=15.0)
		ax0.set_xlabel("LWP$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("LWP$_{\mathrm{RPG}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Scatter_MiRAC-P_RPG"

		plotname = plotname + "_%s-%s"%(aux_info['date_start'].replace('-',''), 
										aux_info['date_end'].replace('-',''))

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()
	1/0


def NN_retrieval(predictor_training, predictand_training, predictor_test,
					predictand_test, mwr_dict, sonde_dict, aux_info,
					save_figures=True, predictor_evaluation=None,
					predictand_evaluation=None):

	# specify output:
	if aux_info['predictand'] == 'iwv':
		predictand_training.output = predictand_training.iwv
		predictand_test.output = predictand_test.iwv
		if aux_info['nya_eval_data']: predictand_eval.output = predictand_eval.iwv
	elif aux_info['predictand'] == 'lwp':
		predictand_training.output = predictand_training.lwp
		predictand_test.output = predictand_test.lwp
		if aux_info['nya_eval_data']: predictand_eval.output = predictand_eval.lwp
	elif aux_info['predictand'] == 'q_profile':
		predictand_training.output = predictand_training.q
		predictand_test.output = predictand_test.q
		if aux_info['nya_eval_data']: predictand_eval.output = predictand_eval.q


	tensorflow.random.set_seed(aux_info['seed'])

	input_shape = predictor_training.input.shape
	model = Sequential()
	model.add(Dense(32, input_dim=input_shape[1], activation=aux_info['activation'], kernel_initializer=aux_info['kernel_init']))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=aux_info['learning_rate']), 
					# metrics=[tensorflow.keras.metrics.RootMeanSquaredError()],
					)

	# train the NN:
	history = model.fit(predictor_training.input_scaled, predictand_training.output, batch_size=aux_info['batch_size'],
				epochs=aux_info['epochs'], verbose=1,
				validation_data=(predictor_test.input_scaled, predictand_test.output),
				callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
				)

	test_loss = history.history['val_loss'][-1]
	print("n_epochs executed: ", len(history.history['loss']))
	print("Test loss: ", test_loss)

	# Quick-Plot validation and training loss:
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.ylim((0,2))
	# plt.show()
	plt.savefig("/net/blanc/awalbroe/Plots/NN_test/keras_test/Loss_val_loss_%s_E%i_%02i.png"%(
					aux_info['file_descr'], aux_info['epochs'],aux_info['seed']),
					dpi=200)
	plt.close()

	# Predict non-training data:
	if aux_info['nya_eval_data']:
		prediction_syn = model.predict(predictor_eval.input_scaled)
	else:
		prediction_syn = model.predict(predictor_test.input_scaled)

	prediction_obs = model.predict(mwr_dict['input_scaled'])

	# some plots for evaluation: (scatter plot and bias histogram):
	error_dict = scatterplot_comparison(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True)
	if aux_info['nya_eval_data']:
		error_dict_syn = scatterplot_comparison_syn(prediction_syn, predictand_eval, aux_info, save_figures=True)
	else:
		error_dict_syn = scatterplot_comparison_syn(prediction_syn, predictand_test, aux_info, save_figures=True)

	plot_bias_histogram(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True)

	error_dict['test_loss'] = test_loss

	if aux_info['save_obs_predictions']:
		save_obs_predictions(prediction_obs, mwr_dict, error_dict_syn, aux_info)
	if aux_info['save_test_predictions']:
		save_test_predictions(prediction_syn, error_dict_syn, aux_info)

	return error_dict, error_dict_syn



###################################################################################################
###################################################################################################


"""
	In this script, Tensorflow.Keras will be used to retrieve of LWP, IWV (and humidity profiles) 
	from ground-based microwave radiometer (MWR) TB measurements of the MiRAC-P. The following
	steps are executed:
	- Importing training and test data (i.e., ERA-Interim provided by E. Orlandi);
		split into training and test data sets
	- quality control of the data (see RPG_MWR_STD_Software_Manual G5_2021.pdf p. 128);
	- rescale input vector (predictors)
	- define and build Neural Network model (try different settings)
	- compile model: choose loss function and optimiser
	- fit model (training): try various subsets of the entire data as training; try
		different batch sizes and learning rates; validate with test data
	- evaluate model (with test data)
	- predict unknown output from new data (application on MiRAC-P obs during MOSAiC)
"""


aux_info = dict()	# dictionary that collects additional information
site = 'pol'		# options of training and test data: 'nya' for Ny Alesudn radiosondes
					# 'pol': ERA-Interim grid points north of 84.5 deg N
rs_version = 'mwr_pro'			# radiosonde type: 'mwr_pro' means that the structure is built
								# so that mwr_pro retrieval can read it
test_purpose = "Testing final setup" # specify the intention of a test (used for the retrieval statistics output .nc file)
aux_info['file_descr'] = "final_setup"				# file name addition (of some plots and netCDF output
aux_info['predictors'] = ["TBs", "DOY_1", "DOY_2"]	# specify input vector (predictors): options: TBs, DOY_1, DOY_2, pres_sfc
													# TBs: all MiRAC-P channels
													# DOY_1: cos(day_of_year)
													# DOY_2: sin(day_of_year)
													# pres_sfc: surface pressure
aux_info['predictand'] = "iwv"						# output variable / predictand: options: "iwv", "lwp", ("q_profile")


yrs = {'pol': ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
				"2012", "2013", "2014", "2015", "2016", "2017"],
		'nya': ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
				"2015", "2016", "2017"]}		# available years of data
yrs = yrs[site]
n_yrs = len(yrs)
n_training = round(0.7*n_yrs)			# number of training years; default: 0.94
n_test = n_yrs - n_training

path_output = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
path_data = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
			'pol': "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/combined/"}		# path of training/test data
path_data = path_data[site]
path_data_obs = "/data/obs/campaigns/mosaic/mirac-p/l1/"
path_radiosondes = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"
path_mirac_level1 = "/data/obs/campaigns/mosaic/mirac-p/l1/"

aux_info['shall_be_stacked'] = False		# specifies if bias histogram shall be stacked or 
											# plotted side by side per bin
aux_info['downsample_obs'] = True			# if True, only every i.e., 3rd second MiRAC-P obs is used
aux_info['add_TB_noise'] = True				# if True, random noise will be added to training and test data. 
											# Remember to define a noise dictionary if True
aux_info['TB_G_corr_err'] = False			# option to add another 'correlated error' to G band frequencies
aux_info['nya_test_data'] = False			# If True, Ny Alesund data will be used for validation (test data)
aux_info['nya_eval_data'] = True			# If True, Ny Alesund data will be used for retrieval evaluation 
											# (only to be used when retrieval is finished)
aux_info['save_obs_predictions'] = False		# if True, predictions made from MiRAC-P observations will be saved
											# to a netCDF file
aux_info['save_test_predictions'] = False	# if True, predictions made from test data will be saved to a 
											# netCDF file
aux_info['mosaic_leg'] = 5
aux_info['considered_period'] = "user"	# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# ("leg%i"%(aux_info['mosaic_leg']))
									# 'user': user defined
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-12"],
					'leg2': ["2019-12-13", "2020-02-23"],
					'leg3': ["2020-02-24", "2020-06-03"],
					'leg4': ["2020-06-04", "2020-08-11"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-01-01", "2020-07-31"]}
aux_info['date_start'] = daterange_options[aux_info['considered_period']][0]	# def: "2019-09-30"
aux_info['date_end'] = daterange_options[aux_info['considered_period']][1]		# def: "2020-10-02"


# Load observed MiRAC-P data:
mwr_dict = import_mirac_BRT_RPG_daterange(path_data_obs, aux_info['date_start'], aux_info['date_end'],
											verbose=1)
aux_info['n_obs'] = len(mwr_dict['time'])

# eventually resample mwr data to speed up testing procedures:
if aux_info['downsample_obs']:
	mwr_dict['time'] = mwr_dict['time'][::3]
	mwr_dict['RF'] = mwr_dict['RF'][::3]
	mwr_dict['TBs'] = mwr_dict['TBs'][::3,:]
mwr_dict['input'] = mwr_dict['TBs']			# start building input vector


# eventually load observed surface pressure data and continue building input vector:
include_pres_sfc = False
if 'pres_sfc' in aux_info['predictors']:
	include_pres_sfc = True
	mwr_dict_add = import_mirac_MET_RPG_daterange(path_data_obs, aux_info['date_start'], aux_info['date_end'], verbose=1)

	# unfortunately, .MET and .BRT files are not on the same time axis. Therefore, resampling is required:
	mwr_dict['pres'] = np.full((aux_info['n_obs'],1), -9999.0)
	sii = 0
	for iii, mwrt in enumerate(mwr_dict['time']): 
		idi = np.where(mwr_dict_add['time'][sii:sii+1500] == mwrt)[0]
		if idi.size > 0:
			sii = idi[0] + sii
			mwr_dict['pres'][iii,0] = mwr_dict_add['pres'][sii]

	# repair missing values:
	pres_fail_idx = np.where(mwr_dict['pres'] < 0)[0]
	assert np.all(np.diff(pres_fail_idx) > 1)		# then lin. interpolation with width=1 can be used
	for iii in pres_fail_idx:
		mwr_dict['pres'][iii,0] = 0.5*(mwr_dict['pres'][iii-1,0] + mwr_dict['pres'][iii+1,0])

	# into input vector:
	mwr_dict['input'] = np.concatenate((mwr_dict['input'], mwr_dict['pres']), axis=1)


# Compute DOY_1 and DOY_2 if needed and further build input vector:
if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
	mwr_dict['DOY_1'], mwr_dict['DOY_2'] = compute_DOY(mwr_dict['time'], return_dt=False, reshape=True)

	mwr_dict['input'] = np.concatenate((mwr_dict['input'], mwr_dict['DOY_1'], mwr_dict['DOY_2']),
										axis=1)


# Load radiosonde data for comparison (if predictand is iwv or q_profile):
if aux_info['predictand'] in ['iwv', 'q_profile']:
	sonde_dict = import_radiosonde_daterange(path_radiosondes, aux_info['date_start'], 
												aux_info['date_end'], s_version='level_2')
	# sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])


# # # # Load std RPG MiRAC for comparison:
# # # mirac_dict = import_mirac_IWV_LWP_RPG_daterange(path_mirac_level1, aux_info['date_start'], 
					# # # aux_info['date_end'], which_retrieval=aux_info['predictand'], minute_avg=False, verbose=1)
# # # # mirac_dict['time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in mirac_dict['time']])


# NN settings:
aux_info['activation'] = "exponential"			# default or best estimate: exponential
aux_info['feature_range'] = (-3.0,1.0)				# best est. with exponential (-3.0, 1.0)
aux_info['batch_size'] = 64
aux_info['epochs'] = 100
aux_info['learning_rate'] = 0.001			# default: 0.001
aux_info['kernel_init'] = 'glorot_uniform'			# default: 'glorot_uniform'
# aux_info['K'] = 10							# number of folds for cross validation

# dict which will save information about each test
retrieval_stats = {	'test_loss': list(),
					'rmse_tot': list(),
					'rmse_bot': list(),
					'rmse_mid': list(),
					'rmse_top': list(),
					'bias_tot': list(),
					'bias_bot': list(),
					'bias_mid': list(),
					'bias_top': list(),
					'batch_size': list(),
					'epochs': list(),
					'activation': list(),
					'seed': list(),
					'learning_rate': list(),
					'feature_range': list()}
retrieval_stats_syn = {'rmse_tot': list(),
					'rmse_bot': list(),
					'rmse_mid': list(),
					'rmse_top': list(),
					'bias_tot': list(),
					'bias_bot': list(),
					'bias_mid': list(),
					'bias_top': list()}
aux_info_stats = ['batch_size', 'epochs', 'activation', 
					'seed', 'learning_rate', 'feature_range']


for aux_info['seed'] in [3,7,9,10,12,15,31,47,50,65,97,99]:

	# Set seed for rng; tests: 3,6,9,10,12,15,31,47,50,65,97,99
	np.random.seed(seed=aux_info['seed'])
	tensorflow.random.set_seed(aux_info['seed'])


	# randomly select training and test years
	yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
	yrs_idx_training = sorted(yrs_idx_rng[:n_training])
	# yrs_idx_training = [0, 1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 16]
	yrs_idx_test = sorted(yrs_idx_rng[n_training:])
	# yrs_idx_test = [3, 8, 9, 10, 15]

	print("Years Training: %s"%(np.asarray(yrs)[yrs_idx_training]))
	print("Years Testing: %s"%(np.asarray(yrs)[yrs_idx_test]))


	# split training and test data:
	data_files_training = list()
	data_files_test = list()
	for yyyy in yrs_idx_training:
		data_files_training.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])

	if aux_info['nya_test_data']:
		data_files_test = sorted(glob.glob("/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/" + 
											"rt_nya_*.nc"))
	else:
		for yyyy in yrs_idx_test:
			data_files_test.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])

	if aux_info['nya_eval_data']:
		data_files_eval = sorted(glob.glob("/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/" + 
											"rt_nya_*.nc"))

	# Define noise strength dictionary for the function add_TB_noise in class radiometers:
	if aux_info['add_TB_noise']:
		noise_dict = {	'183.91':	0.75,
						'184.81':	0.75,
						'185.81':	0.75,
						'186.81':	0.75,
						'188.31':	0.75,
						'190.81':	0.75,
						'243.00':	4.20,
						'340.00':	4.50}

		# Load radiometer TB data (independent predictor):
		predictor_training = radiometers(data_files_training, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc, 
											add_TB_noise=aux_info['add_TB_noise'],
											noise_dict=noise_dict)
		predictor_test = radiometers(data_files_test, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc,
											add_TB_noise=aux_info['add_TB_noise'],
											noise_dict=noise_dict)

		if aux_info['nya_eval_data']:
			predictor_eval = radiometers(data_files_eval, instrument='synthetic',
											include_pres_sfc=include_pres_sfc,
											add_TB_noise=aux_info['add_TB_noise'],
											noise_dict=noise_dict)

		# additional 'correlated noise' for G band channels:
		if aux_info['TB_G_corr_err']:
			G_corr_err = np.random.normal(0,0.5)
			frq_idx = select_MWR_channels(predictor_training.TB, predictor_training.freq, 'G', 2)
			predictor_training.TB[:,frq_idx] = predictor_training.TB[:,frq_idx] + G_corr_err

			frq_idx = select_MWR_channels(predictor_test.TB, predictor_test.freq, 'G', 2)
			predictor_test.TB[:,frq_idx] = predictor_test.TB[:,frq_idx] + G_corr_err

			if aux_info['nya_eval_data']:
				frq_idx = select_MWR_channels(predictor_eval.TB, predictor_eval.freq, 'G', 2)
				predictor_eval.TB[:,frq_idx] = predictor_eval.TB[:,frq_idx] + G_corr_err
			

	else:
		# Load radiometer TB data (independent predictor):
		predictor_training = radiometers(data_files_training, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc)
		predictor_test = radiometers(data_files_test, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc)

		if aux_info['nya_eval_data']:
			predictor_eval = radiometers(data_files_eval, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc)


	# Load predictand data:
	if aux_info['predictand'] in ["iwv", "q_profile"]:
		predictand_training = radiosondes(data_files_training, s_version=rs_version)
		predictand_test = radiosondes(data_files_test, s_version=rs_version)

		aux_info['n_training'] = len(predictand_training.launch_time)
		aux_info['n_test'] = len(predictand_test.launch_time)

		if aux_info['nya_eval_data']:
			predictand_eval = radiosondes(data_files_eval, s_version=rs_version)
			aux_info['n_eval'] = len(predictand_eval.launch_time)

	else:
		predictand_training = radiosondes(data_files_training, s_version=rs_version, with_lwp=True)
		predictand_test = radiosondes(data_files_test, s_version=rs_version, with_lwp=True)

		pdb.set_trace()

		del predictand_training.iwv
		del predictand_test.iwv

		aux_info['n_training'] = len(predictand_training.launch_time)
		aux_info['n_test'] = len(predictand_test.launch_time)

		if aux_info['nya_eval_data']:
			predictand_eval = radiosondes(data_files_eval, s_version=rs_version, with_lwp=True)
			del predictand_eval.iwv
			aux_info['n_eval'] = len(predictand_eval.launch_time)


	print(aux_info['n_training'], aux_info['n_test'])


	# Quality control (can be commented out if this part of the script has been performed successfully):
	if aux_info['predictand'] in ['iwv', 'q_profile']:
		# # # # # simple_quality_control(predictand_training, predictand_test, aux_info)

		# further expand the quality control and check if IWV values are okay:
		# In the Ny Alesund radiosonde training data there are some questionable IWV values 
		# (< -80000 kg m^-2). These values need replacement:
		# also need to repair training TBs at the same spot:
		iwv_broken_training = np.argwhere(predictand_training.iwv < 0).flatten()
		iwv_broken_test = np.argwhere(predictand_test.iwv < 0).flatten()
		if iwv_broken_training.size > 0:
			predictand_training.iwv[iwv_broken_training] = np.asarray([(predictand_training.iwv[ib-1] + 
															predictand_training.iwv[ib+1]) / 2 for ib in iwv_broken_training])
			predictor_training.TB[iwv_broken_training,:] = np.asarray([(predictor_training.TB[ib-1,:] + 
															predictor_training.TB[ib+1,:]) / 2 for ib in iwv_broken_training])

		if iwv_broken_test.size > 0:
			predictand_test.iwv[iwv_broken_test] = np.asarray([(predictand_test.iwv[ib-1] + 
															predictand_test.iwv[ib+1]) / 2 for ib in iwv_broken_test])
			predictor_test.TB[iwv_broken_test,:] = np.asarray([(predictor_test.TB[ib-1,:] + 
															predictor_test.TB[ib+1,:]) / 2 for ib in iwv_broken_test])

		if aux_info['nya_eval_data']:
			iwv_broken_eval = np.argwhere(predictand_eval.iwv < 0).flatten()
			if iwv_broken_eval.size > 0:
				predictand_eval.iwv[iwv_broken_eval] = np.asarray([(predictand_eval.iwv[ib-1] + 
																predictand_eval.iwv[ib+1]) / 2 for ib in iwv_broken_eval])
				predictor_eval.TB[iwv_broken_eval,:] = np.asarray([(predictor_eval.TB[ib-1,:] + 
																predictor_eval.TB[ib+1,:]) / 2 for ib in iwv_broken_eval])


	# Start building input vector for training and test data:
	predictor_training.input = predictor_training.TB
	predictor_test.input = predictor_test.TB
	if aux_info['nya_eval_data']: predictor_eval.input = predictor_eval.TB


	# If chosen, add surface pressure to input vector:
	if "pres_sfc" in aux_info['predictors']:
		predictor_training.input = np.concatenate((predictor_training.input,
													np.reshape(predictor_training.pres, (aux_info['n_training'],1))),
													axis=1)
		predictor_test.input = np.concatenate((predictor_test.input,
													np.reshape(predictor_test.pres, (aux_info['n_test'],1))),
													axis=1)
		if aux_info['nya_eval_data']:
			predictor_eval.input = np.concatenate((predictor_eval.input,
													np.reshape(predictor_eval.pres, (aux_info['n_eval'],1))),
													axis=1)

	# Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
	if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
		predictor_training.DOY_1, predictor_training.DOY_2 = compute_DOY(predictor_training.time, return_dt=False, reshape=True)
		predictor_test.DOY_1, predictor_test.DOY_2 = compute_DOY(predictor_test.time, return_dt=False, reshape=True)

		predictor_training.input = np.concatenate((predictor_training.input, 
													predictor_training.DOY_1,
													predictor_training.DOY_2), axis=1)
		predictor_test.input = np.concatenate((predictor_test.input,
													predictor_test.DOY_1,
													predictor_test.DOY_2), axis=1)

		if aux_info['nya_eval_data']:
			predictor_eval.DOY_1, predictor_eval.DOY_2 = compute_DOY(predictor_eval.time, return_dt=False, reshape=True)
			predictor_eval.input = np.concatenate((predictor_eval.input,
													predictor_eval.DOY_1,
													predictor_eval.DOY_2), axis=1)


	"""
		Define and build Neural Network model: Start with Multilayer Perceptron Model (MLP)
		which has got fully connected (Dense) layers only. Input_shape depends on whether or not
		DOY and surface pressure are included (and another bias term might also be included).
		One hidden layer with activation fct. tanh (try others), n_neurons = 8 (or 9 depending 
		on inclusion of yet another bias term). Last layer (output layer) has got 1 node (IWV), 
		linear activation function. Eventually, the inclusion of the second bias term 
		from hidden layer to output layer requires the use of the Functional API of Keras.

		Eventually more complex architectures might be tested (Convolutional Neural Networks (CNN))
		to better capture nonlinearities 
		Example: https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm

		Loss function: MSE, optimiser: adam (both options might also be changed during testing and
		build phase)

		Fit model: try different training data combinations (TB only, TB + bias, TB + DOY, 
		TB + DOY + pres_sfc, TB + bias + DOY + pres_sfc). try different batch sizes (1, 2**2 - 2**9,
		n_training) and learning rates (lr; lower values for small batch sizes)
		Eventually add validation_dataset=(predictor_test, predictand_test).
		Avoid overfitting by applying Early Stop: callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
	"""


	print(aux_info['activation'], aux_info['feature_range'])

	# Rescale input: Use MinMaxScaler:
	scaler = MinMaxScaler(feature_range=aux_info['feature_range']).fit(predictor_training.input)
	predictor_training.input_scaled = scaler.transform(predictor_training.input)
	predictor_test.input_scaled = scaler.transform(predictor_test.input)
	if aux_info['nya_eval_data']: predictor_eval.input_scaled = scaler.transform(predictor_eval.input)

	# Rescale obs predictor:
	mwr_dict['input_scaled'] = scaler.transform(mwr_dict['input'])


	"""
		Plot relations of input to output (predictor to predictand)
		# # plot_DOY_predictand(predictor_training, predictand_training, aux_info, save_figures=True)
		# # plot_pres_sfc_predictand(predictor_training, predictand_training, aux_info, save_figures=True)
	"""


	print("(batch_size, epochs, seed)=", aux_info['batch_size'], aux_info['epochs'], aux_info['seed'])
	print("learning_rate=", aux_info['learning_rate'])

	error_dict, error_dict_syn = NN_retrieval(predictor_training, predictand_training, predictor_test, 
													predictand_test, mwr_dict, sonde_dict, aux_info,
													save_figures=True, predictor_evaluation=predictor_eval,
													predictand_evaluation=predictand_eval)

	# save information: 
	for ek in error_dict.keys():
		if ek in ['test_loss_cv', 'test_loss_cv_mean', 'test_loss_cv_std', 'stddev']:
			continue
		elif ek in ['test_loss']:
			retrieval_stats[ek].append(error_dict[ek])
		else:
			retrieval_stats[ek].append(error_dict[ek])
			retrieval_stats_syn[ek].append(error_dict_syn[ek])

	for ek in aux_info_stats:
		retrieval_stats[ek].append(aux_info[ek])


# Save retrieval stats to xarray dataset, then to netcdf:
nc_output_name = "Retrieval_stat_test_" + "train70_%s_BS%i_E%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs']) + "_"
if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
	nc_output_name += "LEG%i_"%(aux_info['mosaic_leg'])
elif aux_info['considered_period'] == 'mosaic':
	nc_output_name += "MOSAiC_"
if aux_info['downsample_obs']:
	nc_output_name += "speedrun_"

feature_range_0 = np.asarray([fr[0] for fr in retrieval_stats['feature_range']])
feature_range_1 = np.asarray([fr[1] for fr in retrieval_stats['feature_range']])

if aux_info['predictand'] == 'iwv':
	RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':	(['test_id'], np.asarray(retrieval_stats['test_loss']),
													{'description': "Test data loss, mean square error",
													'units': "mm^2"}),
									'rmse_tot':		(['test_id'], np.asarray(retrieval_stats['rmse_tot']),
													{'description': "Root Mean Square Error (RMSE) of observed and predicted IWV",
													'units': "mm"}),
									'rmse_bot':		(['test_id'], np.asarray(retrieval_stats['rmse_bot']),
													{'description': "Like rmse_tot but confined to IWV range [0,5) mm",
													'units': "mm"}),
									'rmse_mid':		(['test_id'], np.asarray(retrieval_stats['rmse_mid']),
													{'description': "Like rmse_tot but confined to IWV range [5,10) mm",
													'units': "mm"}),
									'rmse_top':		(['test_id'], np.asarray(retrieval_stats['rmse_top']),
													{'description': "Like rmse_tot but confined to IWV range [10,100) mm",
													'units': "mm"}),
									'test_rmse_tot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_tot']),
														{'description': "Test data Root Mean Square Error (RMSE) of target and predicted IWV",
														'units': "mm"}),
									'test_rmse_bot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_bot']),
														{'description': "Like test_rmse_tot but confined to IWV range [0,5) mm",
														'units': "mm"}),
									'test_rmse_mid':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_mid']),
														{'description': "Like test_rmse_tot but confined to IWV range [5,10) mm",
														'units': "mm"}),
									'test_rmse_top':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_top']),
														{'description': "Like test_rmse_tot but confined to IWV range [10,100) mm",
														'units': "mm"}),
									'bias_tot':		(['test_id'], np.asarray(retrieval_stats['bias_tot']),
													{'description': "Bias of predicted - observed IWV",
													'units': "mm"}),
									'bias_bot':		(['test_id'], np.asarray(retrieval_stats['bias_bot']),
													{'description': "Like bias_tot but confined to IWV range [0,5) mm",
													'units': "mm"}),
									'bias_mid':		(['test_id'], np.asarray(retrieval_stats['bias_mid']),
													{'description': "Like bias_tot but confined to IWV range [5,10) mm",
													'units': "mm"}),
									'bias_top':		(['test_id'], np.asarray(retrieval_stats['bias_top']),
													{'description': "Like bias_tot but confined to IWV range [10,100) mm",
													'units': "mm"}),
									'test_bias_tot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_tot']),
														{'description': "Bias of test data predicted - target IWV",
														'units': "mm"}),
									'test_bias_bot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_bot']),
														{'description': "Like test_bias_tot but confined to IWV range [0,5) mm",
														'units': "mm"}),
									'test_bias_mid':	(['test_id'], np.asarray(retrieval_stats_syn['bias_mid']),
														{'description': "Like test_bias_tot but confined to IWV range [5,10) mm",
														'units': "mm"}),
									'test_bias_top':	(['test_id'], np.asarray(retrieval_stats_syn['bias_top']),
														{'description': "Like test_bias_tot but confined to IWV range [10,100) mm",
														'units': "mm"}),
									'batch_size':	(['test_id'], np.asarray(retrieval_stats['batch_size']),
													{'description': "Neural Network training batch size"}),
									'epochs':		(['test_id'], np.asarray(retrieval_stats['epochs']),
													{'description': "Neural Network training epoch number"}),
									'activation':	(['test_id'], np.asarray(retrieval_stats['activation']),
													{'description': "Neural Network activation function from input to hidden layer"}),
									'seed':			(['test_id'], np.asarray(retrieval_stats['seed']),
													{'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
									'learning_rate':(['test_id'], np.asarray(retrieval_stats['learning_rate']),
													{'description': "Learning rate of NN optimizer"}),
									'feature_range0': (['test_id'], feature_range_0,
													{'description': "Lower end of feature range of tensorflow's MinMaxScaler"}),
									'feature_range1': (['test_id'], feature_range_1,
													{'description': "Upper end of feature range of tensorflow's MinMaxScaler"})},
									coords=			{'test_id': (['test_id'], np.arange(len(retrieval_stats['test_loss'])),
													{'description': "Test number"})})

elif aux_info['predictand'] == 'lwp':
		RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':	(['test_id'], np.asarray(retrieval_stats['test_loss']),
													{'description': "Test data loss, mean square error",
													'units': "(kg m^-2)^2"}),
									'rmse_tot':		(['test_id'], np.asarray(retrieval_stats['rmse_tot']),
													{'description': "Root Mean Square Error (RMSE) of observed and predicted LWP",
													'units': "kg m^-2"}),
									'rmse_bot':		(['test_id'], np.asarray(retrieval_stats['rmse_bot']),
													{'description': "Like rmse_tot but confined to LWP range [0,25) g m^-2",
													'units': "kg m^-2"}),
									'rmse_mid':		(['test_id'], np.asarray(retrieval_stats['rmse_mid']),
													{'description': "Like rmse_tot but confined to LWP range [25,100) g m^-2",
													'units': "kg m^-2"}),
									'rmse_top':		(['test_id'], np.asarray(retrieval_stats['rmse_top']),
													{'description': "Like rmse_tot but confined to LWP range [100,inf) g m^-2",
													'units': "kg m^-2"}),
									'test_rmse_tot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_tot']),
														{'description': "Test data Root Mean Square Error (RMSE) of target and predicted LWP",
														'units': "kg m^-2"}),
									'test_rmse_bot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_bot']),
														{'description': "Like test_rmse_tot but confined to LWP range [0,25) g m^-2",
														'units': "kg m^-2"}),
									'test_rmse_mid':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_mid']),
														{'description': "Like test_rmse_tot but confined to LWP range [25,100) g m^-2",
														'units': "kg m^-2"}),
									'test_rmse_top':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_top']),
														{'description': "Like test_rmse_tot but confined to LWP range [100,inf) g m^-2",
														'units': "kg m^-2"}),
									'bias_tot':		(['test_id'], np.asarray(retrieval_stats['bias_tot']),
													{'description': "Bias of predicted - observed LWP",
													'units': "kg m^-2"}),
									'bias_bot':		(['test_id'], np.asarray(retrieval_stats['bias_bot']),
													{'description': "Like bias_tot but confined to LWP range [0,25) g m^-2",
													'units': "kg m^-2"}),
									'bias_mid':		(['test_id'], np.asarray(retrieval_stats['bias_mid']),
													{'description': "Like bias_tot but confined to LWP range [25,100) g m^-2",
													'units': "kg m^-2"}),
									'bias_top':		(['test_id'], np.asarray(retrieval_stats['bias_top']),
													{'description': "Like bias_tot but confined to LWP range [100,inf) g m^-2",
													'units': "kg m^-2"}),
									'test_bias_tot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_tot']),
														{'description': "Bias of test data predicted - target LWP",
														'units': "kg m^-2"}),
									'test_bias_bot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_bot']),
														{'description': "Like test_bias_tot but confined to LWP range [0,25) g m^-2",
														'units': "kg m^-2"}),
									'test_bias_mid':	(['test_id'], np.asarray(retrieval_stats_syn['bias_mid']),
														{'description': "Like test_bias_tot but confined to LWP range [25,100) g m^-2",
														'units': "kg m^-2"}),
									'test_bias_top':	(['test_id'], np.asarray(retrieval_stats_syn['bias_top']),
														{'description': "Like test_bias_tot but confined to LWP range [100,inf) g m^-2",
														'units': "kg m^-2"}),
									'batch_size':	(['test_id'], np.asarray(retrieval_stats['batch_size']),
													{'description': "Neural Network training batch size"}),
									'epochs':		(['test_id'], np.asarray(retrieval_stats['epochs']),
													{'description': "Neural Network training epoch number"}),
									'activation':	(['test_id'], np.asarray(retrieval_stats['activation']),
													{'description': "Neural Network activation function from input to hidden layer"}),
									'seed':			(['test_id'], np.asarray(retrieval_stats['seed']),
													{'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
									'learning_rate':(['test_id'], np.asarray(retrieval_stats['learning_rate']),
													{'description': "Learning rate of NN optimizer"}),
									'feature_range0': (['test_id'], feature_range_0,
													{'description': "Lower end of feature range of tensorflow's MinMaxScaler"}),
									'feature_range1': (['test_id'], feature_range_1,
													{'description': "Upper end of feature range of tensorflow's MinMaxScaler"})},
									coords=			{'test_id': (['test_id'], np.arange(len(retrieval_stats['test_loss'])),
													{'description': "Test number"})})

if site == 'pol':
	RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
	if aux_info['nya_test_data']:
		RETRIEVAL_STAT_DS.attrs['test_data'] = "Ny Alesund radiosondes 2006-2017"
	else:
		RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
elif site == 'nya':
	RETRIEVAL_STAT_DS.attrs['training_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
	RETRIEVAL_STAT_DS.attrs['test_data'] = "Subset of Ny Alesund radiosondes 2006-2017"

RETRIEVAL_STAT_DS.attrs['input_vector'] = ("(TB_183.31+/-0.6, TB_183.31+/-1.5, TB_183.31+/-2.5, TB_183.31+/-3.5, TB_183.31+/-5.0, " +
											"TB_183.31+/-7.5, TB_243.00, TB_340.00")
if 'pres_sfc' in aux_info['predictors']:
	RETRIEVAL_STAT_DS.attrs['input_vector'] = RETRIEVAL_STAT_DS.input_vector + ", pres_sfc"

	nc_output_name = nc_output_name + "pres_sfc_"

if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
	RETRIEVAL_STAT_DS.attrs['input_vector'] = RETRIEVAL_STAT_DS.input_vector + ", cos(DayOfYear), sin(DayOfYear)"
RETRIEVAL_STAT_DS.attrs['input_vector'] = RETRIEVAL_STAT_DS.input_vector + ")"

RETRIEVAL_STAT_DS.attrs['test_purpose'] = test_purpose
RETRIEVAL_STAT_DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
datetime_utc = dt.datetime.utcnow()
RETRIEVAL_STAT_DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")


RETRIEVAL_STAT_DS.to_netcdf(path_output + nc_output_name + datetime_utc.strftime("%Y%m%d_%H%M") + ".nc", mode='w', format="NETCDF4")
RETRIEVAL_STAT_DS.close()


print("Done....")
print(datetime_utc - ssstart)