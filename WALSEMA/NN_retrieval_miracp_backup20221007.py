import pdb
import glob
import copy
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gc
import os
import sys

from matplotlib.ticker import PercentFormatter

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from import_data import import_PS_mastertrack, import_mirac_level1b_daterange_pangaea, import_mirac_level1b_daterange
from my_classes_walsema import radiosondes, radiometers, era_i
from data_tools import compute_retrieval_statistics, compute_DOY, select_MWR_channels, datetime_to_epochtime

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
	path_output,
	prediction,
	mwr_dict,
	aux_info):

	"""
	Save the Neural Network prediction to a netCDF file. Variables to be included:
	time, flag, output variable (prediction), standard error (std. dev. (bias corrected!),
	lat, lon, zsl (altitude above mean sea level),

	Parameters:
	-----------
	path_output : str
		Path where output is saved to.
	prediction : array of floats
		Array that contains the predicted output and is to be saved.
	mwr_dict : dict
		Dictionary containing data of the MiRAC-P.
	retrieval_stats_syn : dict
		Dictionary containing information about the errors (RMSE, bias).
	aux_info : dict
		Dictionary containing additional information about the NN.
	"""

	path_output_l1 = path_output + "l1/"
	path_output_l2 = path_output + "l2/"


	# Add geoinfo data: Load it, interpolate it on the MWR time axis, set the right attribute (source of
	# information), 
	# # # ps_track_dict = load_geoinfo_MOSAiC_polarstern()

	# # # # interpolate Polarstern track data on mwr data time axis:
	# # # for ps_key in ['lat', 'lon', 'time']:
		# # # ps_track_dict[ps_key + "_ip"] = np.interp(np.rint(mwr_dict['time']), ps_track_dict['time'], ps_track_dict[ps_key])

	# extract day, month and year from start date:
	date_start = dt.datetime.strptime(aux_info['date_start'], "%Y-%m-%d")
	date_end = dt.datetime.strptime(aux_info['date_end'], "%Y-%m-%d")



	# Save the data on a daily basis:
	# Also set flag bits to 1024 (which means adding 1024) when retrieved quantity is beyond thresholds:
	import sklearn
	import netCDF4 as nc
	l1_var = 'tb'
	l1_var_units = "K"
	l1_version = "v00"
	l2_version = "v00"
	if aux_info['predictand'] == 'iwv':
		output_var = 'prw'
		output_units = "kg m-2"

		prediction_thresh = [0, 100]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		mwr_dict['flag'][idx_beyond] += 1024

	elif aux_info['predictand'] == 'lwp':
		output_var = 'clwvi'
		output_units = "kg m-2"

		prediction_thresh = [-0.2, 3.0]		# kg m-2
		idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]
		mwr_dict['flag'][idx_beyond] += 1024

	now_date = date_start
	while now_date <= date_end:

		path_addition = f"{now_date.year:04}/{now_date.month:02}/{now_date.day:02}/"

		# check if path exists:
		path_output_dir = os.path.dirname(path_output_l1 + path_addition)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)
		path_output_dir = os.path.dirname(path_output_l2 + path_addition)
		if not os.path.exists(path_output_dir):
			os.makedirs(path_output_dir)

		print(now_date)

		# filter time:
		now_date_epoch = datetime_to_epochtime(now_date)
		now_date_epoch_plus = now_date_epoch + 86399		# plus one day minus one second
		time_idx = np.where((mwr_dict['time'] >= now_date_epoch) & (mwr_dict['time'] <= now_date_epoch_plus))[0]

		if len(time_idx) > 0:
			# Save primary data (level 1) to xarray dataset, then to netcdf: (TBs, cos(DOY), sin(DOY)):
			nc_output_name = f"WALSEMA_uoc_lhumpro-243-340_l1_{l1_var}_{l1_version}_{dt.datetime.strftime(now_date, '%Y%m%d%H%M%S')}"
			
			DS = xr.Dataset({'lat':			(['time'], np.full_like(mwr_dict['time'][time_idx], 78.0).astype(np.float32),
											{'units': "degree_north",
											'standard_name': "latitude",
											'long_name': "latitude of the RV Polarstern"}),
							'lon':			(['time'], np.full_like(mwr_dict['time'][time_idx], 5.0).astype(np.float32),
											{'units': "degree_east",
											'standard_name': "longitude",
											'long_name': "longitude of the RV Polarstern"}),
							'zsl':			(['time'], np.full_like(mwr_dict['time'][time_idx], 21.0).astype(np.float32),
											{'units': "m",
											'standard_name': "altitude",
											'long_name': "altitude above mean sea level"}),
							'freq_sb':		(['n_freq'], mwr_dict['freq_sb'].astype(np.float32),
											{'units': "GHz",
											'standard_name': "sensor_band_central_radiation_frequency",
											'long_name': "frequency of microwave channels"}),
							'azi':			(['time'], np.zeros_like(mwr_dict['time'][time_idx]).astype(np.float32),
											{'units': "degree",
											'standard_name': "sensor_azimuth_angle",
											'comment': "0=North, 90=East, 180=South, 270=West"}),
							'ele':			(['time'], np.full_like(mwr_dict['time'][time_idx], 89.97).astype(np.float32),
											{'units': "degree",
											'long_name': "sensor elevation angle"}),
							l1_var:			(['time', 'n_freq'], mwr_dict['tb'][time_idx,:].astype(np.float32),
											{'units': l1_var_units,
											'standard_name': "brightness_temperature",
											'long_name': "brightness temperatures"}),
						'tb_bias_estimate': (['time', 'n_freq'], mwr_dict['tb_bias_estimate'][time_idx,:].astype(np.float32),
											{'units': l1_var_units,
											'long_name': "brightness temperature offset subtracted from measured brightness temperature",
											'comment': ("Some types of MWR require a systemmatic adjustement of the measured TBs. This variable " +
														"gives the offset which was subtracted from each measurement. The offset was deteremined " +
														"from not applicable.")}),
							'freq_shift':	(['n_freq'], mwr_dict['freq_shift'].astype(np.float32),
											{'units': "GHz",
											'long_name': ("frequency shift applied to correct measured brightness temperature for frequency offset " +
															"of microwave radiometer channel"),
											'comment': ("RPG offers a frequency shift within the radiometer software. This shift will modify " +
														"the TBs calculated. Original TBs cannot be reconstructed. The variable given here is " +
														"intended to inform the user which frequency shifts were applied to the given TBs")}),
					'tb_absolute_accuracy': (['n_freq'], mwr_dict['tb_absolute_accuracy'].astype(np.float32),
											{'units': l1_var_units,
											'long_name': "total calibration uncertainty of brightness temperature, one standard deviation",
											'comment': ("This variable is an estimate of the one-standard-deviation calibration error to be " +
														"expected from an absolute system calibration, i.e. the likely systematic error of " +
														"brightness temperature. As a reference see Maschwitz et al. 2012, AMT (Tab. 5). " +
														"However, these numbers differ from instrument to instrument and should be adapted " +
														"accordingly. Values only valid for elevation angles larger than 20deg.")}),
							'tb_cov':		(['n_freq', 'n_freq'], mwr_dict['tb_cov'].astype(np.float32),
											{'units': "K*K",
											'long_name': "error covariance matrix of brightness temperature channels",
											'comment': ("This variable is calculated from brightness temperature observations of an internal " +
														"black body whose physical temperature is known. The square root of the matrix diagonal " +
														"gives the brightness temperature random error of each frequency channel. " +
														"Values only valid for elevation angles larger than 20deg.")}),
							'cos_doy':		(['time'], mwr_dict['DOY_1'][time_idx].flatten().astype(np.float32),
											{'units': "dimensionless",
											'long_name': "COS of the day of the year",
											'comment': "Used for level 2 data."}),
							'sin_doy':		(['time'], mwr_dict['DOY_2'][time_idx].flatten().astype(np.float32),
											{'units': "dimensionless",
											'long_name': "SIN of the day of the year",
											'comment': "Used for level 2 data."}),
							'ta':			(['time'], mwr_dict['ta'][time_idx].astype(np.float32),
											{'units': "K",
											'standard_name': "air_temperature",
											'long_name': "air temperature",
											'comments': "ambient air temperature measured by sensor on microwave radiometer",
											'valid_min': np.array([200.]).astype(np.float32)[0],
											'valid_max': np.array([330.]).astype(np.float32)[0]}),
							'pa':			(['time'], mwr_dict['pa'][time_idx].astype(np.float32),
											{'units': "Pa",
											'standard_name': "air_pressure",
											'long_name': "air pressure",
											'comment': "ambient air pressure measured by sensor on microwave radiometer",
											'valid_min': np.array([90000.]).astype(np.float32)[0],
											'valid_max': np.array([104000.]).astype(np.float32)[0]}),
							'hur':			(['time'], mwr_dict['hur'][time_idx].astype(np.float32),
											{'units': "1",
											'standard_name': "relative_humidity",
											'long_name': "relative humidity",
											'comment': "ambient relative humidity measured by sensor on microwave radiometer",
											'valid_min': np.array([0.]).astype(np.float32)[0],
											'valid_max': np.array([1.1]).astype(np.float32)[0]}),
							'flag':			(['time'], mwr_dict['flag'][time_idx].astype(np.short),
											{'long_name': "quality control flags",
											'flag_masks': np.array([1,2,4,8,16,32,64,128,256,512], dtype=np.short),
											'flag_meanings': ("visual_inspection_filter_band_1 visual_inspection_filter_band2 visual_inspection_filter_band3 " +
															"rain_flag sanity_receiver_band1 sanity_receiver_band1 sun_in_beam unused " +
															"unused tb_threshold_band1"),
											'comment': ("Flags indicate data that the user should only use with care. In cases of doubt, please refer " +
														"to the contact person. A Fillvalue of 0 means that data has not been flagged. " +
														"Bands refer to the measurement ranges (if applicable) of the microwave radiometer; " +
														"i.e band 1: all lhumpro frequencies (170-200, 243, and 340 GHz); tb valid range: " +
														"[  2.70, 330.00] in K; ")})},
							coords=			{'time': 	(['time'], mwr_dict['time'][time_idx].astype(np.float64),
														{'units': "seconds since 1970-01-01 00:00:00 UTC",
														'standard_name': "time"})})

			# adapt fill values:
			# Make sure that _FillValue is not added to certain variables:
			exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl', 'freq_sb']
			for kk in exclude_vars_fill_value:
				DS[kk].encoding["_FillValue"] = None

			# add fill values to remaining variables:
			vars_fill_value = ['azi', 'ele', 'tb', 'tb_bias_estimate', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov',
								'cos_doy', 'sin_doy', 'ta', 'pa', 'hur', 'flag']
			for kk in vars_fill_value:
				if kk != 'flag':
					DS[kk].encoding["_FillValue"] = float(-999.)
				else:
					DS[kk].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]


			DS.attrs['Title'] = "Microwave radiometer brightness temperatures"
			DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
			DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
			DS.attrs['Source'] = "RPG LHUMPRO-243-340 G5 microwave radiometer"
			DS.attrs['Dependencies'] = "external"
			DS.attrs['Conventions'] = "CF-1.6"
			datetime_utc = dt.datetime.utcnow()
			DS.attrs['Processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
			DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
			DS.attrs['Comments'] = ""
			DS.attrs['License'] = "For non-commercial use only."
			DS.attrs['Measurement_site'] = "RV Polarstern"


			# encode time:
			DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
			DS['time'].encoding['dtype'] = 'double'

			DS.to_netcdf(path_output_l1 + path_addition + nc_output_name + ".nc", mode='w', format="NETCDF4")
			DS.close()


			# Save predictions (level 2) to xarray dataset, then to netcdf:
			nc_output_name = f"WALSEMA_uoc_lhumpro-243-340_l2_{output_var}_{l2_version}_{dt.datetime.strftime(now_date, '%Y%m%d%H%M%S')}"

			# create Dataset:
			if aux_info['predictand'] == 'iwv':
				DS = xr.Dataset({'lat':			(['time'], np.full_like(mwr_dict['time'][time_idx], 78.0).astype(np.float32),
												{'units': "degree_north",
												'standard_name': "latitude",
												'long_name': "latitude of the RV Polarstern"}),
								'lon':			(['time'], np.full_like(mwr_dict['time'][time_idx], 5.0).astype(np.float32),
												{'units': "degree_east",
												'standard_name': "longitude",
												'long_name': "longitude of the RV Polarstern"}),
								'zsl':			(['time'], np.full_like(mwr_dict['time'][time_idx], 21.0).astype(np.float32),
												{'units': "m",
												'standard_name': "altitude",
												'long_name': "altitude above mean sea level"}),
								'azi':			(['time'], np.zeros_like(mwr_dict['time'][time_idx]).astype(np.float32),
												{'units': "degree",
												'standard_name': "sensor_azimuth_angle",
												'comment': "0=North, 90=East, 180=South, 270=West"}),
								'ele':			(['time'], np.full_like(mwr_dict['time'][time_idx], 89.97).astype(np.float32),
												{'units': "degree",
												'long_name': "sensor elevation angle"}),
								output_var:		(['time'], prediction.flatten()[time_idx].astype(np.float32),
												{'units': output_units,
												'standard_name': "atmosphere_mass_content_of_water_vapor",
												'comment': ("These values denote the vertically integrated amount of water vapor from the surface to TOA. " +
															"The (bias corrected) standard error of atmosphere mass content of water vapor is " +
															f"0.55 {output_units}. More specifically, the " +
															f"standard error of {output_var} in the ranges [0, 5), [5, 10), [10, 100) {output_units} is " +
															f"0.07, 0.36, 1.11 {output_units}.")}),
						output_var + "_offset": (['time'], np.full_like(mwr_dict['time'][time_idx], 0.0).astype(np.float32),
												{'units': output_units,
												'long_name': "atmosphere_mass_content_of_water_vapor offset correction based on brightness temperature offset",
												'comment': ("This value has been subtracted from the original prw value to account for instrument " +
															"calibration drifts. The information is designated for expert user use.")}),
								'flag':			(['time'], mwr_dict['flag'][time_idx].astype(np.short),
												{'long_name': "quality control flags",
												'flag_masks': np.array([1,2,4,8,16,32,64,128,256,512,1024], dtype=np.short),
												'flag_meanings': ("visual_inspection_filter_band_1 visual_inspection_filter_band2 visual_inspection_filter_band3 " +
																	"rain_flag sanity_receiver_band1 sanity_receiver_band1 sun_in_beam unused " +
																	"unused tb_threshold_band1 iwv_lwp_threshold"),
												'comment': ("Flags indicate data that the user should only use with care. In cases of doubt, please refer " +
															"to the contact person. A Fillvalue of 0 means that data has not been flagged. " +
															"Bands refer to the measurement ranges (if applicable) of the microwave radiometer; " +
															"i.e band 1: all lhumpro frequencies (170-200, 243, and 340 GHz); tb valid range: " +
															"[  2.70, 330.00] in K; prw valid range: [0.,  100.] in kg m-2; clwvi (not considering " +
															"clwvi offset correction) valid range: [-0.2, 3.0] in k gm-2; ")})},
								coords=			{'time': (['time'], mwr_dict['time'][time_idx].astype(np.float64),
															{'units': "seconds since 1970-01-01 00:00:00 UTC",
															'standard_name': "time"})})

				# adapt fill values:
				# Make sure that _FillValue is not added to certain variables:
				exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl']
				for kk in exclude_vars_fill_value:
					DS[kk].encoding["_FillValue"] = None

				# add fill values to remaining variables:
				vars_fill_value = ['azi', 'ele', 'prw', 'prw_offset', 'flag']
				for kk in vars_fill_value:
					if kk != 'flag':
						DS[kk].encoding["_FillValue"] = float(-999.)
					else:
						DS[kk].encoding["_FillValue"] = np.array([0]).astype(np.short)[0]

				DS.attrs['Title'] = f"Microwave radiometer retrieved {output_var}"
				DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
				DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
				DS.attrs['Source'] = "RPG LHUMPRO-243-340 G5 microwave radiometer"
				DS.attrs['Dependencies'] = f"WALSEMA_mirac-p_l1_tb"
				DS.attrs['Conventions'] = "CF-1.6"
				datetime_utc = dt.datetime.utcnow()
				DS.attrs['Processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
				DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
				DS.attrs['Comments'] = ""
				DS.attrs['License'] = "For non-commercial use only."
				DS.attrs['Measurement_site'] = "RV Polarstern"

				DS.attrs['retrieval_type'] = "Neural Network"
				DS.attrs['python_packages'] = (f"python version: 3.8.10, tensorflow: {tensorflow.__version__}, keras: {keras.__version__}, " +
												f"numpy: {np.__version__}, sklearn: {sklearn.__version__}, netCDF4: {nc.__version__}")
				DS.attrs['retrieval_batch_size'] = f"{str(aux_info['batch_size'])}"
				DS.attrs['retrieval_epochs'] = f"{str(aux_info['epochs'])}"
				DS.attrs['retrieval_learning_rate'] = f"{str(aux_info['learning_rate'])}"
				DS.attrs['retrieval_activation_function'] = f"{aux_info['activation']} (from input to hidden layer)"
				DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_info['feature_range'])}"
				DS.attrs['retrieval_rng_seed'] = str(aux_info['seed'])
				DS.attrs['retrieval_hidden_layers_nodes'] = f"1: 32 (kernel_initializer={aux_info['kernel_init']})"
				DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
				DS.attrs['retrieval_callbacks'] = "EarlyStopping(monitor=val_loss, patience=20, restore_best_weights=True)"

			if site == 'pol':
				DS.attrs['training_data'] = "ERA-Interim"
				DS.attrs['training_data_years'] = "2001, 2002, 2004, 2006, 2007, 2008, 2009, 2011, 2012, 2013, 2015, 2017"

				if aux_info['nya_test_data']:
					DS.attrs['test_data'] = "Ny-Alesund radiosondes 2006-2017"
				else:
					DS.attrs['test_data'] = "ERA-Interim"
					DS.attrs['test_data_years'] = "2003, 2005, 2010, 2014, 2016"

				DS.attrs['n_training_samples'] = aux_info['n_training']
				DS.attrs['n_test_samples'] = aux_info['n_test']


			DS.attrs['training_test_TB_noise_std_dev'] = ("TB_183.31+/-0.6GHz: 0.75, TB_183.31+/-1.5GHz: 0.75, TB_183.31+/-2.5GHz: 0.75, " +
															"TB_183.31+/-3.5GHz: 0.75, TB_183.31+/-5.0GHz: 0.75, " +
															"TB_183.31+/-7.5GHz: 0.75, TB_243.00GHz: 4.2")

			DS.attrs['input_vector'] = ("(TB_183.31+/-0.6GHz, TB_183.31+/-1.5GHz, TB_183.31+/-2.5GHz, TB_183.31+/-3.5GHz, TB_183.31+/-5.0GHz, " +
													"TB_183.31+/-7.5GHz, TB_243.00GHz")
			if 'pres_sfc' in aux_info['predictors']:
				DS.attrs['input_vector'] = DS.input_vector + ", pres_sfc"

			if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
				DS.attrs['input_vector'] = DS.input_vector + ", cos(DayOfYear), sin(DayOfYear)"
			DS.attrs['input_vector'] = DS.input_vector + ")"
			DS.attrs['output_vector'] = f"({output_var})"


			# encode time:
			DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
			DS['time'].encoding['dtype'] = 'double'

			DS.to_netcdf(path_output_l2 + path_addition + nc_output_name + ".nc", mode='w', format="NETCDF4")
			DS.close()

		# update date:
		now_date += dt.timedelta(days=1)


def save_test_predictions(
	predictor_training,
	predictand_training,
	predictor_test,
	predictand_test,
	predictor_eval,
	predictand_eval,
	prediction_syn, 
	error_dict_syn, 
	aux_info):

	"""
	Save the Neural Network prediction of the evaluation data to a netCDF file. 
	Variables to be included: prediction, target, predictor_training, predictor_test,
	predictand_training, predictand_test, bias, rmse, stddev (all four categories).

	Parameters:
	-----------
	predictor_training : class radiometers
		Belongs to class 'radiometers' and contains training data of the predictor (and
		some auxiliary information).
	predictand_training : class radiosondes
		Belongs to class radiosondes and contains training data of the predictand (i.e., prw,
		clwvi (iwv, lwp)).
	predictor_test : class radiometers
		Belongs to class 'radiometers' and contains test (validation) data of the predictor 
		(and some auxiliary information).
	predictand_test : class radiosondes
		Belongs to class radiosondes and contains test (validation) data of the predictand 
		(i.e., prw, clwvi (iwv, lwp)).
	predictor_eval : class radiometers
		Belongs to class 'radiometers' and contains evaluation data of the predictor 
		(and some auxiliary information).
	predictand_eval : class radiosondes
		Belongs to class radiosondes and contains evaluation data of the predictand 
		(i.e., prw, clwvi (iwv, lwp)).
	prediction_syn : array of floats
		Array that contains the predicted output of the evaluation data.
	error_dict_syn : dict
		Dictionary containing information about the errors (RMSE, bias).
	aux_info : dict
		Dictionary containing additional information about the NN.
	"""

	# Save predictions to xarray dataset, then to netcdf:
	if aux_info['predictand'] == 'iwv':
		output_var = 'prw'
		output_units = "kg m-2"
	elif aux_info['predictand'] == 'lwp':
		output_var = 'clwvi'
		output_units = "kg m-2"

	# input vector description:
	input_vector_descr = ("(TB_183.31+/-0.6, TB_183.31+/-1.5, TB_183.31+/-2.5, TB_183.31+/-3.5, TB_183.31+/-5.0, " +
							"TB_183.31+/-7.5, TB_243.00, TB_340.00")
	if 'pres_sfc' in aux_info['predictors']:
		input_vector_descr += ", pres_sfc"

	if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
		input_vector_descr += ", cos(DayOfYear), sin(DayOfYear)"
	input_vector_descr += ")"

	nc_output_name = f"WALSEMA_mirac-p_NN_eval_{output_var}_nya_radiosondes_seed_{aux_info['seed']:03}"

	# create Dataset:
	if aux_info['predictand'] == 'iwv':
		DS = xr.Dataset({'prediction':	(['time_ev'], prediction_syn.flatten(),
										{'description': f"predicted {output_var} (aka {aux_info['predictand']}) of the evaluation data",
										'units': output_units}),
						'target':		(['time_ev'], predictand_eval.output,
										{'description': f"target predictand ({output_var}, aka {aux_info['predictand']}) of the evaluation data",
										'units': output_units}),
						'predictor_ev': (['time_ev', 'input'], predictor_eval.input,
										{'description': "predictors of evaluation data (unscaled) used to derive prediction",
										'comments': input_vector_descr,
										'units': "si units"}),
						'predictor_tr': (['time_tr', 'input'], predictor_training.input,
										{'description': "predictors of training data (unscaled)",
										'comments': input_vector_descr,
										'units': "si units"}),
						'predictor_te': (['time_te', 'input'], predictor_test.input,
										{'description': "predictors of test (validation) data (unscaled)",
										'comments': input_vector_descr,
										'units': "si units"}),
						'predictand_tr':(['time_tr'], predictand_training.output,
										{'description': f"predictand of training data ({output_var})",
										'units': output_units}),
						'predictand_te':(['time_te'], predictand_test.output,
										{'description': f"predictand of test (validation) data ({output_var})",
										'units': output_units}),
						'prw_rmse':		([], error_dict_syn['rmse_tot'],
										{'description': "root mean square error (rmse) of prediction and target over all evaluated test cases",
										'comments': f"values can be large due to high rmse for large atmosphere mass content of water vapour {output_var}",
										'units': output_units}),
						'prw_rmse_0':	([], error_dict_syn['rmse_bot'],
										{'description': f"rmse of prediction and target in range [0, 5) {output_units}",
										'units': output_units}),
						'prw_rmse_1':	([], error_dict_syn['rmse_mid'],
										{'description': f"rmse of prediction and target in range [5, 10) {output_units}",
										'units': output_units}),
						'prw_rmse_2':	([], error_dict_syn['rmse_top'],
										{'description': f"rmse of prediction and target in range [10, 100) {output_units}",
										'units': output_units}),
						'prw_err':		([], error_dict_syn['stddev'],
										{'description': f"standard deviation (bias corrected rmse) of predicted {output_var} ({aux_info['predictand']})",
										'comments': f"values can be large due to misfit for atmosphere mass content of water vapour (prw) > 10 {output_units}",
										'units': output_units}),
						'prw_err_0':	([], error_dict_syn['stddev_bot'],
										{'description': f"standard deviation (bias corrected rmse) of predicted {output_var} ({aux_info['predictand']}) in range [0, 5) {output_units}",
										'units': output_units}),
						'prw_err_1':	([], error_dict_syn['stddev_mid'],
										{'description': f"standard deviation (bias corrected rmse) of predicted {output_var} ({aux_info['predictand']}) in range [5, 10) {output_units}",
										'units': output_units}),
						'prw_err_2':	([], error_dict_syn['stddev_top'],
										{'description': f"standard deviation (bias corrected rmse) of predicted {output_var} ({aux_info['predictand']}) in range [10, 100) {output_units}",
										'units': output_units}),
						'prw_bias':		([], error_dict_syn['bias_tot'],
										{'description': "bias (prediction - target) over all evaluated test cases",
										'comments': "values can be large due to high biases for large atmosphere mass content of water vapour (prw)",
										'units': output_units}),
						'prw_bias_0':	([], error_dict_syn['bias_bot'],
										{'description': f"bias (prediction - target) in range [0, 5) {output_units}",
										'units': output_units}),
						'prw_bias_1':	([], error_dict_syn['bias_mid'],
										{'description': f"bias (prediction - target) in range [5, 10) {output_units}",
										'units': output_units}),
						'prw_bias_2':	([], error_dict_syn['bias_top'],
										{'description': f"bias (prediction - target) in range [10, 100) {output_units}",
										'units': output_units}),
						'n_training':	([], aux_info['n_training'],
										{'description': "number of training cases"}),
						'n_test':		([], aux_info['n_test'],
										{'description': "number of test (validation) cases"}),
						'n_eval':		([], aux_info['n_eval'],
										{'description': "number of evaluation cases"})},
						coords=			{'time_ev': (['time_ev'], predictor_eval.time.astype(np.int64),
													{'description': "time of evaluation data",
													'units': "seconds since 1970-01-01 00:00:00 UTC"}),
										'time_tr':	(['time_tr'], predictor_training.time.astype(np.int64),
													{'description': "time of training data",
													'units': "seconds since 1970-01-01 00:00:00 UTC"}),
										'time_te':	(['time_te'], predictor_test.time.astype(np.int64),
													{'description': "time of test (validation) data",
													'units': "seconds since 1970-01-01 00:00:00 UTC"}),
										'input':	(['input'], input_vector_descr[1:-1].split(", "),
													{'units': "si units"})})


		DS.attrs['retrieval_type'] = "Neural Networks (Tensorflow V2.5.0, Keras V2.5.0), Python Version: 3.8.10"
		DS.attrs['retrieval_batch_size'] = f"{str(aux_info['batch_size'])}"
		DS.attrs['retrieval_epochs'] = f"{str(aux_info['epochs'])}"
		DS.attrs['retrieval_learning_rate'] = f"{str(aux_info['learning_rate'])}"
		DS.attrs['retrieval_activation_function'] = f"{aux_info['activation']} (from input to hidden layer)"
		DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_info['feature_range'])}"
		DS.attrs['retrieval_rng_seed'] = str(aux_info['seed'])
		DS.attrs['retrieval_hidden_layers_nodes'] = f"1: 32 (kernel_initializer={aux_info['kernel_init']})"
		DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
		DS.attrs['retrieval_callbacks'] = "EarlyStopping(monitor=val_loss, patience=20, restore_best_weights=True)"

	if site == 'pol':
		DS.attrs['training_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
		if aux_info['nya_test_data']:
			DS.attrs['test_data'] = "Ny Alesund radiosondes 2006-2017"
		else:
			DS.attrs['test_data'] = "Subset of ERA-Interim 2001-2017, 8 virtual stations north of 84.5 deg N"
	elif site == 'nya':
		DS.attrs['training_data'] = "Subset of Ny Alesund radiosondes 2006-2017"
		DS.attrs['test_data'] = "Subset of Ny Alesund radiosondes 2006-2017"

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
	DS.attrs['output_vector'] = f"({output_var})"

	DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
	datetime_utc = dt.datetime.utcnow()
	DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

	# encode time:
	DS['time_ev'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	DS['time_tr'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	DS['time_te'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
	DS['time_ev'].encoding['dtype'] = 'int64'
	DS['time_tr'].encoding['dtype'] = 'int64'
	DS['time_te'].encoding['dtype'] = 'int64'

	DS.to_netcdf(path_output + nc_output_name + ".nc", mode='w', format="NETCDF4")
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


def scatterplot_comparison_syn(prediction, predictand_test, aux_info, save_figures=True):

	# For test data evaluation (prediction of synthetic data):

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
	x_stuff_bot = x_stuff[mask_bot]
	x_stuff_mid = x_stuff[mask_mid]
	x_stuff_top = x_stuff[mask_top]
	y_stuff_bot = y_stuff[mask_bot]
	y_stuff_mid = y_stuff[mask_mid]
	y_stuff_top = y_stuff[mask_top]
	stats_dict_bot = compute_retrieval_statistics(x_stuff_bot, y_stuff_bot, compute_stddev=True)
	stats_dict_mid = compute_retrieval_statistics(x_stuff_mid, y_stuff_mid, compute_stddev=True)
	stats_dict_top = compute_retrieval_statistics(x_stuff_top, y_stuff_top, compute_stddev=True)

	error_dict = {	'rmse_tot': sc_rmse,
					'rmse_bot': stats_dict_bot['rmse'],
					'rmse_mid': stats_dict_mid['rmse'],
					'rmse_top': stats_dict_top['rmse'],
					'stddev': 	stats_dict['stddev'],
					'stddev_bot': 	stats_dict_bot['stddev'],
					'stddev_mid': 	stats_dict_mid['stddev'],
					'stddev_top': 	stats_dict_top['stddev'],
					'bias_tot': sc_bias,
					'bias_bot': stats_dict_bot['bias'],
					'bias_mid': stats_dict_mid['bias'],
					'bias_top': stats_dict_top['bias']}


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

		# fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()
	gc.collect()

	return error_dict


def NN_retrieval(predictor_training, predictand_training, predictor_test,
					predictand_test, aux_info, save_figures=True, 
					predictor_eval=None, predictand_eval=None):

	# specify output:
	if aux_info['predictand'] == 'iwv':
		predictand_training.output = predictand_training.iwv
		predictand_test.output = predictand_test.iwv

	input_shape = predictor_training.input.shape
	model = Sequential()
	model.add(Dense(32, input_dim=input_shape[1], activation=aux_info['activation'], kernel_initializer=aux_info['kernel_init']))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=aux_info['learning_rate']))

	# train the NN:
	history = model.fit(predictor_training.input_scaled, predictand_training.output, batch_size=aux_info['batch_size'],
				epochs=aux_info['epochs'], verbose=1,
				validation_data=(predictor_test.input_scaled, predictand_test.output),
				callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
				)

	test_loss = history.history['val_loss'][-1]			# test data MSE
	print("n_epochs executed: ", len(history.history['loss']))
	print("Test loss: ", test_loss)


	# Predict non-training data:
	if aux_info['finalized']:
		return model

	else:
		# in this case both test data (validation data) and obs can be predicted
		prediction_syn = model.predict(predictor_test.input_scaled)

		# some plots for evaluation: (scatter plot):
		error_dict_syn = scatterplot_comparison_syn(prediction_syn, predictand_test, aux_info, save_figures=True)
		error_dict_syn['test_loss'] = test_loss

		if aux_info['save_test_predictions']:
			save_test_predictions(prediction_syn, error_dict_syn, aux_info)

		return error_dict_syn


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

# sys.argv[1] determines if 20 random numbers shall be cycled through ("20_runs" or whether the finalized
# form ("finalized") of the retrieval, which creates the published MiRAC-P data products, shall be run
if sys.argv[1] not in ['20_runs', 'finalized']:
	raise ValueError("Script must be called with either 'python3 NN_retrieval_miracp.py " + '"20_runs"' +
						"' or 'python3 NN_retrieval_miracp.py " + '"finalized"' + "'!")


aux_info = dict()	# dictionary that collects additional information
site = 'pol'		# options of training and test data: 'nya' for Ny-Alesund radiosondes
					# 'pol': ERA-Interim grid points north of 84.5 deg N (recommended)
rs_version = 'mwr_pro'	# radiosonde type: 'mwr_pro' means that the structure is built
								# so that mwr_pro retrieval can read it (for Ny-Alesund radiosondes and unconcatenated ERA-I)
test_purpose = "Retrieval: WALSEMA" # specify the intention of a test (used for the retrieval statistics output .nc file)
aux_info['file_descr'] = "retrieval_walsema"				# file name addition (of some plots and netCDF output
aux_info['predictors'] = ["TBs", "DOY_1", "DOY_2"]	# specify input vector (predictors): options: TBs, DOY_1, DOY_2, pres_sfc
													# TBs: all MiRAC-P channels
													# DOY_1: cos(day_of_year)
													# DOY_2: sin(day_of_year)
													# pres_sfc: surface pressure (not recommended)
aux_info['predictand'] = "iwv"						# output variable / predictand: options: "iwv", ("lwp", ("q_profile"))


yrs = {'pol': ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
				"2012", "2013", "2014", "2015", "2016", "2017"],
		'nya': ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
				"2015", "2016", "2017"]}		# available years of data
yrs = yrs[site]
n_yrs = len(yrs)
n_training = round(0.7*n_yrs)			# number of training years; default: 0.7
n_test = n_yrs - n_training

path_output = "/data/obs/campaigns/WALSEMA/atm/mirac-p/"			# path where output is saved to
path_data = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
			'pol': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/retrieval_training/mirac-p/"}		# path of training/test data
path_data = path_data[site]
path_mirac_level1 = "/data/obs/campaigns/WALSEMA/atm/mirac-p/l1/"					# path of mirac-p tb data

aux_info['add_TB_noise'] = True				# if True, random noise will be added to training and test data. 
											# Remember to define a noise dictionary if True
aux_info['TB_G_corr_err'] = False			# option to add another 'correlated error' to G band frequencies
aux_info['nya_test_data'] = False			# If True, Ny Alesund data will be used for validation (test data)

if sys.argv[1] == '20_runs':
	aux_info['finalized'] = False				# if True: finalized version of retrieval to create MiRAC-P derived product
	aux_info['save_obs_predictions'] = False		# if True, predictions made from MiRAC-P observations will be saved
												# to a netCDF file (i.e., for finalized retrieval)
elif sys.argv[1] == 'finalized':
	aux_info['finalized'] = True				# if True: finalized version of retrieval to create MiRAC-P derived product
	aux_info['save_obs_predictions'] = True		# if True, predictions made from MiRAC-P observations will be saved
												# to a netCDF file (i.e., for finalized retrieval)
aux_info['save_test_predictions'] = False	# if True, predictions made from test (or eval) data will be saved to a 
											# netCDF file
aux_info['considered_period'] = 'walsema'	#f"leg{aux_info['mosaic_leg']}"	# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2022-07-01 - 2022-08-12
									# 'walsema': entire period: 2022-06-28 - 2022-08-12
daterange_options = {'mwr_range': ["2022-07-07", "2022-08-12"],
					'walsema': ["2022-06-28", "2022-08-18"],
					'user': ["2020-01-01", "2020-01-10"]}
aux_info['date_start'] = daterange_options[aux_info['considered_period']][0]	# def: "2019-09-20"
aux_info['date_end'] = daterange_options[aux_info['considered_period']][1]		# def: "2020-10-12"

# Training data months to be limited to certain season / months:
aux_info['training_data_months'] = []	# as list of integers; empty list if all months to be used


# eventually load observed surface pressure data and continue building input vector:
include_pres_sfc = False
if 'pres_sfc' in aux_info['predictors']: include_pres_sfc = True

# set radiosonde version automatically if Ny-Alesund radiosondes == Test data
if aux_info['nya_test_data']:
	rs_version = 'mwr_pro'


if sys.argv[1] == 'finalized':
	# Load observed MiRAC-P data:
	# mwr_dict = import_mirac_level1b_daterange_pangaea(path_mirac_level1, aux_info['date_start'], aux_info['date_end'])
	mwr_dict = import_mirac_level1b_daterange(path_mirac_level1, aux_info['date_start'], aux_info['date_end'], vers='v00')

	mwr_dict['flag'] = mwr_dict['flag'].astype(np.short)
	mwr_dict['time'] = mwr_dict['time'].astype(np.float64)

	aux_info['n_obs'] = len(mwr_dict['time'])

	# start building input vector
	mwr_dict['input'] = mwr_dict['tb'][:,:-1]


	if 'pres_sfc' in aux_info['predictors']:
		mwr_dict['pres'] = mwr_dict['pa']
		# into input vector:
		mwr_dict['input'] = np.concatenate((mwr_dict['input'], mwr_dict['pres']), axis=1)


	# Compute DOY_1 and DOY_2 if needed and further build input vector:
	if ("DOY_1" in aux_info['predictors']) and ("DOY_2" in aux_info['predictors']):
		mwr_dict['DOY_1'], mwr_dict['DOY_2'] = compute_DOY(mwr_dict['time'], return_dt=False, reshape=True)

		mwr_dict['input'] = np.concatenate((mwr_dict['input'], mwr_dict['DOY_1'], mwr_dict['DOY_2']),
											axis=1)


# NN settings:
aux_info['activation'] = "exponential"			# default or best estimate: exponential
aux_info['feature_range'] = (-3.0,1.0)				# best est. with exponential (-3.0, 1.0)
aux_info['batch_size'] = 64
aux_info['epochs'] = 100
aux_info['learning_rate'] = 0.001			# default: 0.001
aux_info['kernel_init'] = 'glorot_uniform'			# default: 'glorot_uniform'

# dict which will save information about each test
retrieval_stats_syn = {'test_loss': list(),
					'rmse_tot': list(),
					'rmse_bot': list(),
					'rmse_mid': list(),
					'rmse_top': list(),
					'stddev': list(),
					'stddev_bot': list(),
					'stddev_mid': list(),
					'stddev_top': list(),
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
aux_info_stats = ['batch_size', 'epochs', 'activation', 
					'seed', 'learning_rate', 'feature_range']

if sys.argv[1] == '20_runs':
	some_seeds = [423, 558, 59, 421, 424, 250, 791, 12, 454, 479, 314, 
					263, 763, 6, 665, 321, 13, 947, 929, 368]
elif sys.argv[1] == 'finalized':
	some_seeds = [558]
for aux_info['seed'] in some_seeds:

	# set rng seeds
	np.random.seed(seed=aux_info['seed'])
	tensorflow.random.set_seed(aux_info['seed'])

	# randomly select training and test years
	yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
	yrs_idx_training = sorted(yrs_idx_rng[:n_training])
	yrs_idx_test = sorted(yrs_idx_rng[n_training:])

	yrs_training = np.asarray(yrs)[yrs_idx_training]
	yrs_testing = np.asarray(yrs)[yrs_idx_test]

	print("Years Training: %s"%(yrs_training))
	print("Years Testing: %s"%(yrs_testing))


	# split training and test data:
	data_files_training = list()
	data_files_test = list()
	data_files_training = sorted(glob.glob(path_data + "MOSAiC_mirac-p_retrieval*.nc"))
	data_files_test = data_files_training

	if aux_info['nya_test_data']:
		data_files_test = sorted(glob.glob("/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/" + 
											"rt_nya_*.nc"))


	# Define noise strength dictionary for the function add_TB_noise in class radiometers:
	if aux_info['add_TB_noise']:
		noise_dict = {	'183.91':	0.75,
						'184.81':	0.75,
						'185.81':	0.75,
						'186.81':	0.75,
						'188.31':	0.75,
						'190.81':	0.75,
						'243.00':	4.20}
						# '340.00':	4.50}	# deactivated for WALSEMA

		# Load radiometer TB data (independent predictor):
		predictor_training = radiometers(data_files_training, instrument='syn_mwr_pro', 
											include_pres_sfc=include_pres_sfc, 
											add_TB_noise=aux_info['add_TB_noise'],
											noise_dict=noise_dict, 
											subset=yrs_training,
											subset_months=aux_info['training_data_months'])

		if aux_info['nya_test_data']:
			predictor_test = radiometers(data_files_test, instrument='synthetic', 
												include_pres_sfc=include_pres_sfc,
												add_TB_noise=aux_info['add_TB_noise'],
												noise_dict=noise_dict)
		else:
			predictor_test = radiometers(data_files_test, instrument='syn_mwr_pro', 
												include_pres_sfc=include_pres_sfc,
												add_TB_noise=aux_info['add_TB_noise'],
												noise_dict=noise_dict,
												subset=yrs_testing,
												subset_months=aux_info['training_data_months'])


		# additional 'correlated noise' for G band channels:
		if aux_info['TB_G_corr_err']:
			G_corr_err = np.random.normal(0,0.5)
			frq_idx = select_MWR_channels(predictor_training.TB, predictor_training.freq, 'G', 2)
			predictor_training.TB[:,frq_idx] = predictor_training.TB[:,frq_idx] + G_corr_err

			frq_idx = select_MWR_channels(predictor_test.TB, predictor_test.freq, 'G', 2)
			predictor_test.TB[:,frq_idx] = predictor_test.TB[:,frq_idx] + G_corr_err
			

	else: # without noise:
		# Load radiometer TB data (independent predictor):
		predictor_training = radiometers(data_files_training, instrument='synthetic', 
											include_pres_sfc=include_pres_sfc,
											subset=yrs_training,
											subset_months=aux_info['training_data_months'])

		if aux_info['nya_test_data']:
			predictor_test = radiometers(data_files_test, instrument='synthetic', 
												include_pres_sfc=include_pres_sfc)
		else:
			predictor_test = radiometers(data_files_test, instrument='syn_mwr_pro', 
												include_pres_sfc=include_pres_sfc,
												subset=yrs_testing,
												subset_months=aux_info['training_data_months'])


	# Load predictand data: (e.g., Ny-Alesund radiosondes or ERA-I)
	if aux_info['predictand'] in ["iwv", "q_profile"]:
		predictand_training = era_i(data_files_training, subset=yrs_training, subset_months=aux_info['training_data_months'])
		if aux_info['nya_test_data']:
			predictand_test = radiosondes(data_files_test, s_version=rs_version)
		else:
			predictand_test = era_i(data_files_test, subset=yrs_testing, subset_months=aux_info['training_data_months'])

		aux_info['n_training'] = len(predictand_training.launch_time)
		aux_info['n_test'] = len(predictand_test.launch_time)


	print(aux_info['n_training'], aux_info['n_test'])


	# Quality control (can be commented out if this part of the script has been performed successfully)
	# The quality control of the ERA-I data has already been performed on the files uploaded to ZENODO.
	if aux_info['predictand'] in ['iwv', 'q_profile']:
		# simple_quality_control(predictand_training, predictand_test, aux_info)

		# further expand the quality control and check if IWV values are okay:
		# In the Ny Alesund radiosonde training data there are some questionable IWV values 
		# (< -80000 kg m^-2). These values need replacement:
		# also need to repair training TBs at the same spot:
		if site == 'nya':
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



	# Start building input vector for training and test data:
	predictor_training.input = predictor_training.TB[:,:-1]
	predictor_test.input = predictor_test.TB[:,:-1]


	# If chosen, add surface pressure to input vector:
	if "pres_sfc" in aux_info['predictors']:
		predictor_training.input = np.concatenate((predictor_training.input,
													np.reshape(predictor_training.pres, (aux_info['n_training'],1))),
													axis=1)
		predictor_test.input = np.concatenate((predictor_test.input,
													np.reshape(predictor_test.pres, (aux_info['n_test'],1))),
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


	"""
		Define and build Neural Network model: Start with Multilayer Perceptron Model (MLP)
		which has got fully connected (Dense) layers only. Input_shape depends on whether or not
		DOY and surface pressure are included.

		Loss function: MSE, optimiser: adam (these options (among others) might also be changed during testing and
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

	# Rescale obs predictor:
	if sys.argv[1] == 'finalized':
		mwr_dict['input_scaled'] = scaler.transform(mwr_dict['input'])

	print("(batch_size, epochs, seed)=", aux_info['batch_size'], aux_info['epochs'], aux_info['seed'])
	print("learning_rate=", aux_info['learning_rate'])


	if sys.argv[1] == 'finalized':
		model = NN_retrieval(predictor_training, predictand_training, predictor_test, 
								predictand_test, aux_info, save_figures=True)

	elif sys.argv[1] == '20_runs':
		error_dict_syn = NN_retrieval(predictor_training, predictand_training, predictor_test, 
										predictand_test, aux_info, save_figures=True)

		# save information: 
		for ek in error_dict_syn.keys():
			if ek in ['test_loss_cv', 'test_loss_cv_mean', 'test_loss_cv_std']:
				continue
			else:
				retrieval_stats_syn[ek].append(error_dict_syn[ek])

		for ek in aux_info_stats:
			retrieval_stats_syn[ek].append(aux_info[ek])


if sys.argv[1] == 'finalized':

	# Predict actual observations:
	prediction_obs = model.predict(mwr_dict['input_scaled'])
	if aux_info['save_obs_predictions']:
		save_obs_predictions(path_output, prediction_obs, mwr_dict, aux_info)

elif sys.argv[1] == '20_runs':

	# Save retrieval stats to xarray dataset, then to netcdf:
	nc_output_name = "Retrieval_stat_test_" + "train70_%s_BS%i_E%i"%(aux_info['file_descr'], aux_info['batch_size'], aux_info['epochs']) + "_"
	if aux_info['considered_period'] == 'mosaic':
		nc_output_name += "WALSEMA_"

	feature_range_0 = np.asarray([fr[0] for fr in retrieval_stats_syn['feature_range']])
	feature_range_1 = np.asarray([fr[1] for fr in retrieval_stats_syn['feature_range']])

	if aux_info['predictand'] == 'iwv':
		RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':	(['test_id'], np.asarray(retrieval_stats_syn['test_loss']),
														{'description': "Test data loss, mean square error",
														'units': "mm^2"}),
										'rmse_tot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_tot']),
														{'description': "Test data Root Mean Square Error (RMSE) of target and predicted IWV",
														'units': "mm"}),
										'rmse_bot':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_bot']),
														{'description': "Like rmse_tot but confined to IWV range [0,5) mm",
														'units': "mm"}),
										'rmse_mid':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_mid']),
														{'description': "Like rmse_tot but confined to IWV range [5,10) mm",
														'units': "mm"}),
										'rmse_top':	(['test_id'], np.asarray(retrieval_stats_syn['rmse_top']),
														{'description': "Like rmse_tot but confined to IWV range [10,100) mm",
														'units': "mm"}),
										'bias_tot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_tot']),
														{'description': "Bias of test data predicted - target IWV",
														'units': "mm"}),
										'bias_bot':	(['test_id'], np.asarray(retrieval_stats_syn['bias_bot']),
														{'description': "Like bias_tot but confined to IWV range [0,5) mm",
														'units': "mm"}),
										'bias_mid':	(['test_id'], np.asarray(retrieval_stats_syn['bias_mid']),
														{'description': "Like bias_tot but confined to IWV range [5,10) mm",
														'units': "mm"}),
										'bias_top':	(['test_id'], np.asarray(retrieval_stats_syn['bias_top']),
														{'description': "Like bias_tot but confined to IWV range [10,100) mm",
														'units': "mm"}),
										'batch_size':	(['test_id'], np.asarray(retrieval_stats_syn['batch_size']),
														{'description': "Neural Network training batch size"}),
										'epochs':		(['test_id'], np.asarray(retrieval_stats_syn['epochs']),
														{'description': "Neural Network training epoch number"}),
										'activation':	(['test_id'], np.asarray(retrieval_stats_syn['activation']),
														{'description': "Neural Network activation function from input to hidden layer"}),
										'seed':			(['test_id'], np.asarray(retrieval_stats_syn['seed']),
														{'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
										'learning_rate':(['test_id'], np.asarray(retrieval_stats_syn['learning_rate']),
														{'description': "Learning rate of NN optimizer"}),
										'feature_range0': (['test_id'], feature_range_0,
														{'description': "Lower end of feature range of tensorflow's MinMaxScaler"}),
										'feature_range1': (['test_id'], feature_range_1,
														{'description': "Upper end of feature range of tensorflow's MinMaxScaler"})},
										coords=			{'test_id': (['test_id'], np.arange(len(retrieval_stats_syn['test_loss'])),
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
datetime_utc = dt.datetime.utcnow()
print(datetime_utc - ssstart)