import numpy as np
import datetime as dt
import glob
import os
import pdb
import sys
import shutil
import xarray as xr


"""
	This script is used to merge ELE90 and normal zenith mode MWR data into a daily file.
	- identify correct files
"""


def modify_DS(
	DS,
	ffc,
	aux_dict):

	"""
	Additional modifications to achieve a data set that is only merged through time dependent
	variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset to be exported.
	ffc : str
		files_for_copy.
	aux_dict : dict
		Dictionary containing additional information.
	"""

	if ffc == 'BLB':

		# first, repair existing variables, then adapt to newer generation RPG file system
		DS['time_reference'] = DS.time_reference[0]
		DS['minimum_TBs'] = DS.minimum_TBs.min('time')
		DS['maximum_TBs'] = DS.maximum_TBs.max('time')
		DS['frequencies'] = DS.frequencies[0,:]
		DS['elevation_scan_angles'] = DS.elevation_scan_angles[0,:]

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['Min_TBs'] = DS.minimum_TBs
		DS['Min_TBs'].attrs['long_name'] = "Minimum TBs in File"
		DS['Min_TBs'].attrs['units'] = "K"

		DS['Max_TBs'] = DS.maximum_TBs
		DS['Max_TBs'].attrs['long_name'] = "Maximum TBs in File"
		DS['Max_TBs'].attrs['units'] = "K"

		DS['ElAngs'] = DS.elevation_scan_angles
		del DS['ElAngs'].attrs['units']
		DS['ElAngs'].attrs['long_name'] = "Elevation Scan Angles"
		DS['ElAngs'].attrs['units'] = "degree (0-90)"

		DS['AzAng'] = DS.azimuth_angle
		DS['AzAng'].attrs['long_name'] = "Azimuth Viewing Angle"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Freq'] = xr.DataArray(np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000], dtype="float32"),
									dims=['number_frequencies'])
		DS['Freq'].attrs['long_name'] = "Channel Center Frequency"
		DS['Freq'].attrs['units'] = "GHz"

		DS['TBs2'] = DS.TBs
		del DS['TBs2'].attrs['units']
		DS['TBs2'].attrs['long_name'] = "Brightness Temperatures"
		DS['TBs2'].attrs['units'] = "K"

		# delete redundant variables:
		del_vars = ['time_reference', 'minimum_TBs', 'maximum_TBs', 'frequencies', 'elevation_scan_angles', 'azimuth_angle',
					'rain_flag', 'TBs']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['TBs'] = DS.TBs2
		DS = DS.drop('TBs2')

	elif ffc == 'BLH':
		DS['time_reference'] = DS.time_reference[0]
		DS['number_integrated_samples'] = DS.number_integrated_samples[0]
		DS['minimum'] = DS.minimum[0]
		DS['maximum'] = DS.maximum[0]

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = DS.number_integrated_samples
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Min_BLH'] = DS.minimum
		del DS['Min_BLH'].attrs['units']
		DS['Min_BLH'].attrs['long_name'] = "Minimum Boundary Layer Height in File"
		DS['Min_BLH'].attrs['units'] = "m"

		DS['Max_BLH'] = DS.maximum
		del DS['Max_BLH'].attrs['units']
		DS['Max_BLH'].attrs['long_name'] = "Maximum Boundary Layer Height in File"
		DS['Max_BLH'].attrs['units'] = "m"

		DS['BLH'] = DS.BLH_data
		del DS['BLH'].attrs['units']
		DS['BLH'].attrs['long_name'] = "Boundary Layer Height DAta"
		DS['BLH'].attrs['units'] = "m"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'rain_flag', 'BLH_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'BRT':
		DS['time_reference'] = DS.time_reference[0]
		DS['integration_time_per_sample'] = DS.integration_time_per_sample[0]
		DS['frequencies'] = DS.frequencies[0,:]

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['Freq'] = xr.DataArray(np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000], dtype="float32"),
									dims=['number_frequencies'])
		DS['Freq'].attrs['long_name'] = "Channel Center Frequency"
		DS['Freq'].attrs['units'] = "GHz"

		DS['Min_TBs'] = DS.TBs.min('time')
		DS['Min_TBs'].attrs['long_name'] = "Minimum TBs in File"
		DS['Min_TBs'].attrs['units'] = "K"

		DS['Max_TBs'] = DS.TBs.max('time')
		DS['Max_TBs'].attrs['long_name'] = "Maximum TBs in File"
		DS['Max_TBs'].attrs['units'] = "K"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['ElAng'] = DS.elevation_angle
		DS['ElAng'].attrs['long_name'] = "Elevation Viewing Angle"
		DS['ElAng'].attrs['comment'] = "-90 is blackbody view, 0 is horizontal view (red arrow), 90 is zenith view, 180 is horizontal view (2nd quadrant)"
		DS['ElAng'].attrs['units'] = "degrees (-90 - 180)"

		DS['AziAng'] = DS.azimuth_angle
		del DS['AziAng'].attrs['units']
		DS['AziAng'].attrs['long_name'] = "Azimuth Viewing Angle"
		DS['AziAng'].attrs['units'] = "DEG (0-360)"

		DS['TBs2'] = DS.TBs
		del DS['TBs2'].attrs['units']
		DS['TBs2'].attrs['long_name'] = "TB Map"
		DS['TBs2'].attrs['units'] = "K"

		# delete redundant variables:
		del_vars = ['time_reference', 'integration_time_per_sample', 'frequencies', 'rain_flag', 'elevation_angle', 'azimuth_angle', 'TBs']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['TBs'] = DS.TBs2
		DS = DS.drop('TBs2')

	elif ffc == 'CBH':
		DS['time_reference'] = DS.time_reference[0]
		DS['number_integrated_samples'] = DS.number_integrated_samples[0]
		DS['minimum'] = DS.minimum.min('time').astype("float32")
		DS['maximum'] = DS.maximum.max('time').astype("float32")

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = DS.number_integrated_samples
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Min_CBH'] = DS.minimum
		DS['Min_CBH'].attrs['long_name'] = "Minimum Cloud Base Height in File"
		DS['Min_CBH'].attrs['units'] = "m"

		DS['Max_CBH'] = DS.maximum
		DS['Max_CBH'].attrs['long_name'] = "Maximum Cloud Base Height in File"
		DS['Max_CBH'].attrs['units'] = "m"

		DS['CBH'] = DS.CBH_data
		del DS['CBH'].attrs['units']
		DS['CBH'].attrs['long_name'] = "Cloud Base Height DAta"
		DS['CBH'].attrs['units'] = "m"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'rain_flag', 'CBH_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)
		
	elif ffc == 'HKD':
		DS['time_reference'] = DS.time_reference[0]
		DS['enable_flags'] = DS.enable_flags[0]

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['EnaFl'] = DS.enable_flags
		DS['EnaFl'].attrs['long_name'] = "HKD Enable Flags"
		DS['EnaFl'].attrs['units'] = "unitless"

		DS['AlFl'] = DS.alarm_flag
		del DS['AlFl'].attrs['Info']
		DS['AlFl'].attrs['long_name'] = "Alarm Flag indicating critical system status"
		DS['AlFl'].attrs['comment'] = "0: Radiometer HW-Status OK, 1: HW-Problem has occurred"

		DS['AT1_T'] = DS.ambient_Target1_temperature
		del DS['AT1_T'].attrs['units']
		DS['AT1_T'].attrs['long_name'] = "Ambient Target Sensor1 Temperature"
		DS['AT1_T'].attrs['units'] = "K"

		DS['AT2_T'] = DS.ambient_Target2_temperature
		del DS['AT2_T'].attrs['units']
		DS['AT2_T'].attrs['long_name'] = "Ambient Target Sensor2 Temperature"
		DS['AT2_T'].attrs['units'] = "K"

		DS['Rec1_T'] = DS.receiver1_temperature
		del DS['Rec1_T'].attrs['units']
		DS['Rec1_T'].attrs['long_name'] = "Receiver1 Temperature"
		DS['Rec1_T'].attrs['units'] = "K"

		DS['Rec2_T'] = DS.receiver2_temperature
		del DS['Rec2_T'].attrs['units']
		DS['Rec2_T'].attrs['long_name'] = "Receiver2 Temperature"
		DS['Rec2_T'].attrs['units'] = "K"

		DS['Rec1_Stab'] = DS.stability_rec1
		del DS['Rec1_Stab'].attrs['units']
		DS['Rec1_Stab'].attrs['long_name'] = "Receiver1 Temperature Stability"
		DS['Rec1_Stab'].attrs['units'] = "K"

		DS['Rec2_Stab'] = DS.stability_rec2
		del DS['Rec2_Stab'].attrs['units']
		DS['Rec2_Stab'].attrs['long_name'] = "Receiver2 Temperature Stability"
		DS['Rec2_Stab'].attrs['units'] = "K"

		DS['QualFl'] = DS.quality_flags
		DS['QualFl'].attrs['long_name'] = "Quality Flags Bit Field"
		DS['QualFl'].attrs['units'] = 'unitless'
		DS['QualFl'].attrs['comment'] = "meaning of the quality bits can be found in appendix A of the operational manual"

		DS['StatFl'] = DS.status_flags
		DS['StatFl'].attrs['long_name'] = "Radiometer Status Flags"
		DS['StatFl'].attrs['units'] = "unitless"
		DS['StatFl'].attrs['comment'] = "meaning of the status flags can be found in appendix A of the operational manual"

		# delete redundant variables:
		del_vars = ['time_reference', 'enable_flags', 'alarm_flag', 'longitude', 'latitude', 'ambient_Target1_temperature',
					'ambient_Target2_temperature', 'receiver1_temperature', 'receiver2_temperature', 'stability_rec1', 
					'stability_rec2', 'quality_flags', 'status_flags']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'IRT':
		DS['time_reference'] = DS.time_reference[0]
		DS['minimum'] = DS.minimum[0]
		DS['maximum'] = DS.maximum[0]
		DS['frequencies'] = DS.frequencies[0,:]

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntTime'] = DS.number_integrated_samples
		DS['IntTime'].attrs['long_name'] = "integration time for each sample"
		DS['IntTime'].attrs['units'] = "seconds"

		DS['Min_IRT'] = DS.minimum
		del DS['Min_IRT'].attrs['units']
		DS['Min_IRT'].attrs['long_name'] = "Minimum IR TB in file"
		DS['Min_IRT'].attrs['units'] = "C"

		DS['Max_IRT'] = DS.maximum
		del DS['Max_IRT'].attrs['units']
		DS['Max_IRT'].attrs['long_name'] = "Maximum IR TB in file"
		DS['Max_IRT'].attrs['units'] = "C"

		DS['WavLens'] = DS.frequencies
		DS['WavLens'].attrs['long_name'] = "IR Channel Center Wavelength"
		DS['WavLens'].attrs['units'] = "um"

		DS['ElAng'] = DS.elevation_angle
		del DS['ElAng'].attrs['units'], DS['ElAng'].attrs['comment'], DS['ElAng'].attrs['long_name']
		DS['ElAng'].attrs['long_name'] = "Elevation Viewing Angle"
		DS['ElAng'].attrs['comment'] = "-90 is blackbody view, 0 is horizontal view (red arrow), 90 is zenith view, 180 is horizontal view (2nd quadrant)"
		DS['ElAng'].attrs['units'] = "degrees (-90 - 180)"

		DS['AziAng'] = DS.azimuth_angle
		del DS['AziAng'].attrs['units']
		DS['AziAng'].attrs['long_name'] = "Azimuth Viewing Angle"
		DS['AziAng'].attrs['units'] = "DEG (0-360)"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info'], DS['RF'].attrs['units']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['IRR_Map'] = DS.IRR_data
		DS['IRR_Map'].attrs['long_name'] = "IRR Data Map"
		DS['IRR_Map'].attrs['units'] = "C"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'frequencies', 'elevation_angle',
					'azimuth_angle', 'rain_flag', 'IRR_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'TPB':
		DS['time_reference'] = DS.time_reference[0]
		DS['number_integrated_samples'] = DS.number_integrated_samples[0]
		DS['retrieval'] = DS.retrieval[0]
		DS['altitude_layers'] = DS.altitude_layers[0,:]
		DS['minimum_T'] = DS.minimum_T.min('time').astype("float32")
		DS['maximum_T'] = DS.maximum_T.max('time').astype("float32")

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['integration_time'] = DS.number_integrated_samples
		DS['integration_time'].attrs['long_name'] = "integration time per profile sample"
		DS['integration_time'].attrs['units'] = "sec"

		DS['retrieval2'] = DS['retrieval']
		del DS['retrieval2'].attrs['Info']
		DS['retrieval2'].attrs['long_name'] = "LV2 retrieval type"
		DS['retrieval2'].attrs['Info'] = "0 = Linear Regr., 1 = Quadr. Regr., 2 = Neural Network"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['units'] = 'unitless'
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['altitude'] = DS.altitude_layers

		DS['min_T'] = DS.minimum_T
		DS['min_T'].attrs['long_name'] = "minimum brightness temperature in data set"
		DS['min_T'].attrs['units'] = "K"

		DS['max_T'] = DS.maximum_T
		DS['max_T'].attrs['long_name'] = "maximum brightness temperature in data set"
		DS['max_T'].attrs['units'] = "K"

		DS['T_prof'] = DS.temperature_profiles
		del DS['T_prof'].attrs['units']
		DS['T_prof'].attrs['long_name'] = "brightness temperature profiles"
		DS['T_prof'].attrs['units'] = "K"

		DS = DS.rename_dims({'number_altitude_layers': 'altitude_layer'})

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'retrieval', 'rain_flag',
					'altitude_layers', 'minimum_T', 'maximum_T', 'temperature_profiles']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['retrieval'] = DS.retrieval2
		DS = DS.drop("retrieval2")

	return DS


def modify_single_DS(
	DS,
	ffc,
	aux_dict):

	"""
	Additional modifications to achieve a data set that is only merged through time dependent
	variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset to be exported.
	ffc : str
		files_for_copy.
	aux_dict : dict
		Dictionary containing additional information.
	"""

	if ffc == 'BLB':

		# first, repair existing variables, then adapt to newer generation RPG file system

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['Min_TBs'] = DS.minimum_TBs
		DS['Min_TBs'].attrs['long_name'] = "Minimum TBs in File"
		DS['Min_TBs'].attrs['units'] = "K"

		DS['Max_TBs'] = DS.maximum_TBs
		DS['Max_TBs'].attrs['long_name'] = "Maximum TBs in File"
		DS['Max_TBs'].attrs['units'] = "K"

		DS['ElAngs'] = DS.elevation_scan_angles
		del DS['ElAngs'].attrs['units']
		DS['ElAngs'].attrs['long_name'] = "Elevation Scan Angles"
		DS['ElAngs'].attrs['units'] = "degree (0-90)"

		DS['AzAng'] = DS.azimuth_angle
		DS['AzAng'].attrs['long_name'] = "Azimuth Viewing Angle"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Freq'] = xr.DataArray(np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000], dtype="float32"),
									dims=['number_frequencies'])
		DS['Freq'].attrs['long_name'] = "Channel Center Frequency"
		DS['Freq'].attrs['units'] = "GHz"

		DS['TBs2'] = DS.TBs
		del DS['TBs2'].attrs['units']
		DS['TBs2'].attrs['long_name'] = "Brightness Temperatures"
		DS['TBs2'].attrs['units'] = "K"

		# delete redundant variables:
		del_vars = ['time_reference', 'minimum_TBs', 'maximum_TBs', 'frequencies', 'elevation_scan_angles', 'azimuth_angle',
					'rain_flag', 'TBs']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['TBs'] = DS.TBs2
		DS = DS.drop('TBs2')

	elif ffc == 'BLH':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = DS.number_integrated_samples
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Min_BLH'] = DS.minimum
		del DS['Min_BLH'].attrs['units']
		DS['Min_BLH'].attrs['long_name'] = "Minimum Boundary Layer Height in File"
		DS['Min_BLH'].attrs['units'] = "m"

		DS['Max_BLH'] = DS.maximum
		del DS['Max_BLH'].attrs['units']
		DS['Max_BLH'].attrs['long_name'] = "Maximum Boundary Layer Height in File"
		DS['Max_BLH'].attrs['units'] = "m"

		DS['BLH'] = DS.BLH_data
		try:
			del DS['BLH'].attrs['units']
		except KeyError:
			DS['BLH'].attrs['long_name'] = "Boundary Layer Height DAta"
			DS['BLH'].attrs['units'] = "m"
		DS['BLH'].attrs['long_name'] = "Boundary Layer Height DAta"
		DS['BLH'].attrs['units'] = "m"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'rain_flag', 'BLH_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'BRT':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['Freq'] = xr.DataArray(np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000], dtype="float32"),
									dims=['number_frequencies'])
		DS['Freq'].attrs['long_name'] = "Channel Center Frequency"
		DS['Freq'].attrs['units'] = "GHz"

		DS['Min_TBs'] = DS.TBs.min('time')
		DS['Min_TBs'].attrs['long_name'] = "Minimum TBs in File"
		DS['Min_TBs'].attrs['units'] = "K"

		DS['Max_TBs'] = DS.TBs.max('time')
		DS['Max_TBs'].attrs['long_name'] = "Maximum TBs in File"
		DS['Max_TBs'].attrs['units'] = "K"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['ElAng'] = DS.elevation_angle
		DS['ElAng'].attrs['long_name'] = "Elevation Viewing Angle"
		DS['ElAng'].attrs['comment'] = "-90 is blackbody view, 0 is horizontal view (red arrow), 90 is zenith view, 180 is horizontal view (2nd quadrant)"
		DS['ElAng'].attrs['units'] = "degrees (-90 - 180)"

		DS['AziAng'] = DS.azimuth_angle
		del DS['AziAng'].attrs['units']
		DS['AziAng'].attrs['long_name'] = "Azimuth Viewing Angle"
		DS['AziAng'].attrs['units'] = "DEG (0-360)"

		DS['TBs2'] = DS.TBs
		del DS['TBs2'].attrs['units']
		DS['TBs2'].attrs['long_name'] = "TB Map"
		DS['TBs2'].attrs['units'] = "K"

		# delete redundant variables:
		del_vars = ['time_reference', 'integration_time_per_sample', 'frequencies', 'rain_flag', 'elevation_angle', 'azimuth_angle', 'TBs']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['TBs'] = DS.TBs2
		DS = DS.drop('TBs2')

	elif ffc == 'CBH':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntSampCnt'] = DS.number_integrated_samples
		DS['IntSampCnt'].attrs['long_name'] = "Number of Integration Samples"
		DS['IntSampCnt'].attrs['units'] = 'unitless'

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['Min_CBH'] = DS.minimum
		DS['Min_CBH'].attrs['long_name'] = "Minimum Cloud Base Height in File"
		DS['Min_CBH'].attrs['units'] = "m"

		DS['Max_CBH'] = DS.maximum
		DS['Max_CBH'].attrs['long_name'] = "Maximum Cloud Base Height in File"
		DS['Max_CBH'].attrs['units'] = "m"

		DS['CBH'] = DS.CBH_data
		del DS['CBH'].attrs['units']
		DS['CBH'].attrs['long_name'] = "Cloud Base Height DAta"
		DS['CBH'].attrs['units'] = "m"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'rain_flag', 'CBH_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)
		
	elif ffc == 'HKD':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['EnaFl'] = DS.enable_flags
		DS['EnaFl'].attrs['long_name'] = "HKD Enable Flags"
		DS['EnaFl'].attrs['units'] = "unitless"

		DS['AlFl'] = DS.alarm_flag
		del DS['AlFl'].attrs['Info']
		DS['AlFl'].attrs['long_name'] = "Alarm Flag indicating critical system status"
		DS['AlFl'].attrs['comment'] = "0: Radiometer HW-Status OK, 1: HW-Problem has occurred"

		DS['AT1_T'] = DS.ambient_Target1_temperature
		del DS['AT1_T'].attrs['units']
		DS['AT1_T'].attrs['long_name'] = "Ambient Target Sensor1 Temperature"
		DS['AT1_T'].attrs['units'] = "K"

		DS['AT2_T'] = DS.ambient_Target2_temperature
		del DS['AT2_T'].attrs['units']
		DS['AT2_T'].attrs['long_name'] = "Ambient Target Sensor2 Temperature"
		DS['AT2_T'].attrs['units'] = "K"

		DS['Rec1_T'] = DS.receiver1_temperature
		del DS['Rec1_T'].attrs['units']
		DS['Rec1_T'].attrs['long_name'] = "Receiver1 Temperature"
		DS['Rec1_T'].attrs['units'] = "K"

		DS['Rec2_T'] = DS.receiver2_temperature
		del DS['Rec2_T'].attrs['units']
		DS['Rec2_T'].attrs['long_name'] = "Receiver2 Temperature"
		DS['Rec2_T'].attrs['units'] = "K"

		DS['Rec1_Stab'] = DS.stability_rec1
		del DS['Rec1_Stab'].attrs['units']
		DS['Rec1_Stab'].attrs['long_name'] = "Receiver1 Temperature Stability"
		DS['Rec1_Stab'].attrs['units'] = "K"

		DS['Rec2_Stab'] = DS.stability_rec2
		del DS['Rec2_Stab'].attrs['units']
		DS['Rec2_Stab'].attrs['long_name'] = "Receiver2 Temperature Stability"
		DS['Rec2_Stab'].attrs['units'] = "K"

		DS['QualFl'] = DS.quality_flags
		DS['QualFl'].attrs['long_name'] = "Quality Flags Bit Field"
		DS['QualFl'].attrs['units'] = 'unitless'
		DS['QualFl'].attrs['comment'] = "meaning of the quality bits can be found in appendix A of the operational manual"

		DS['StatFl'] = DS.status_flags
		DS['StatFl'].attrs['long_name'] = "Radiometer Status Flags"
		DS['StatFl'].attrs['units'] = "unitless"
		DS['StatFl'].attrs['comment'] = "meaning of the status flags can be found in appendix A of the operational manual"

		# delete redundant variables:
		del_vars = ['time_reference', 'enable_flags', 'alarm_flag', 'longitude', 'latitude', 'ambient_Target1_temperature',
					'ambient_Target2_temperature', 'receiver1_temperature', 'receiver2_temperature', 'stability_rec1', 
					'stability_rec2', 'quality_flags', 'status_flags']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'IRT':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['IntTime'] = DS.number_integrated_samples
		DS['IntTime'].attrs['long_name'] = "integration time for each sample"
		DS['IntTime'].attrs['units'] = "seconds"

		DS['Min_IRT'] = DS.minimum
		del DS['Min_IRT'].attrs['units']
		DS['Min_IRT'].attrs['long_name'] = "Minimum IR TB in file"
		DS['Min_IRT'].attrs['units'] = "C"

		DS['Max_IRT'] = DS.maximum
		del DS['Max_IRT'].attrs['units']
		DS['Max_IRT'].attrs['long_name'] = "Maximum IR TB in file"
		DS['Max_IRT'].attrs['units'] = "C"

		DS['WavLens'] = DS.frequencies
		DS['WavLens'].attrs['long_name'] = "IR Channel Center Wavelength"
		DS['WavLens'].attrs['units'] = "um"

		DS['ElAng'] = DS.elevation_angle
		del DS['ElAng'].attrs['units'], DS['ElAng'].attrs['comment'], DS['ElAng'].attrs['long_name']
		DS['ElAng'].attrs['long_name'] = "Elevation Viewing Angle"
		DS['ElAng'].attrs['comment'] = "-90 is blackbody view, 0 is horizontal view (red arrow), 90 is zenith view, 180 is horizontal view (2nd quadrant)"
		DS['ElAng'].attrs['units'] = "degrees (-90 - 180)"

		DS['AziAng'] = DS.azimuth_angle
		del DS['AziAng'].attrs['units']
		DS['AziAng'].attrs['long_name'] = "Azimuth Viewing Angle"
		DS['AziAng'].attrs['units'] = "DEG (0-360)"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info'], DS['RF'].attrs['units']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['IRR_Map'] = DS.IRR_data
		DS['IRR_Map'].attrs['long_name'] = "IRR Data Map"
		DS['IRR_Map'].attrs['units'] = "C"

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'minimum', 'maximum', 'frequencies', 'elevation_angle',
					'azimuth_angle', 'rain_flag', 'IRR_data']
		for ddd in del_vars:
			DS = DS.drop(ddd)

	elif ffc == 'TPB':

		# adapt variables to newer generation conventions
		DS['file_code'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['file_code'].attrs['long_name'] = "four byte code identifying file type and version"

		DS['Rad_ID'] = xr.DataArray(np.array([-2147483647], dtype='int32')[0])
		DS['Rad_ID'].attrs['long_name'] = "radiometer model ID"
		DS['Rad_ID'].attrs['comment'] = "RPG-HATPRO"

		DS['time'].attrs['comment'] = "time is UTC"

		DS['RSFactor'] = xr.DataArray(np.array([1], dtype='int32')[0])
		DS['RSFactor'].attrs['long_name'] = "rapid sampling multiplier (1 / 2 / 4)"
		DS['RSFactor'].attrs['units'] = 'unitless'
		DS['RSFactor'].attrs['Comment'] = "Sampling interval: 1: 1 sec, 2: 0.5 sec , 4: 0.25 sec"

		DS['integration_time'] = DS.number_integrated_samples
		DS['integration_time'].attrs['long_name'] = "integration time per profile sample"
		DS['integration_time'].attrs['units'] = "sec"

		DS['retrieval2'] = DS['retrieval']
		del DS['retrieval2'].attrs['Info']
		DS['retrieval2'].attrs['long_name'] = "LV2 retrieval type"
		DS['retrieval2'].attrs['Info'] = "0 = Linear Regr., 1 = Quadr. Regr., 2 = Neural Network"

		DS['RF'] = DS.rain_flag
		del DS['RF'].attrs['Info']
		DS['RF'].attrs['long_name'] = "Rain Flag"
		DS['RF'].attrs['units'] = 'unitless'
		DS['RF'].attrs['Info'] = "0 = No Rain, 1 = Raining"

		DS['altitude'] = DS.altitude_layers

		DS['min_T'] = DS.minimum_T
		DS['min_T'].attrs['long_name'] = "minimum brightness temperature in data set"
		DS['min_T'].attrs['units'] = "K"

		DS['max_T'] = DS.maximum_T
		DS['max_T'].attrs['long_name'] = "maximum brightness temperature in data set"
		DS['max_T'].attrs['units'] = "K"

		DS['T_prof'] = DS.temperature_profiles
		del DS['T_prof'].attrs['units']
		DS['T_prof'].attrs['long_name'] = "brightness temperature profiles"
		DS['T_prof'].attrs['units'] = "K"

		DS = DS.rename_dims({'number_altitude_layers': 'altitude_layer'})

		# delete redundant variables:
		del_vars = ['time_reference', 'number_integrated_samples', 'retrieval', 'rain_flag',
					'altitude_layers', 'minimum_T', 'maximum_T', 'temperature_profiles']
		for ddd in del_vars:
			DS = DS.drop(ddd)

		DS['retrieval'] = DS.retrieval2
		DS = DS.drop("retrieval2")

	return DS


###################################################################################################
###################################################################################################


aux_dict = dict()		# contains extra info
paths = dict()

if len(sys.argv) == 1:
	aux_dict['instr'] = 'hatpro'
elif len(sys.argv) == 2:
	aux_dict['instr'] = sys.argv[1]


# paths:
paths['source'] = f"/data/obs/campaigns/WALSEMA/atm/{aux_dict['instr']}/l1/"
paths['dest'] = f"/data/obs/campaigns/WALSEMA/atm2/{aux_dict['instr']}/l1/"


# start and end date:
date0 = "2022-07-07"
date1 = "2022-08-12"
date0_dt = dt.datetime.strptime(date0, "%Y-%m-%d")
date1_dt = dt.datetime.strptime(date1, "%Y-%m-%d")


# Variables that may not have fill values:
exclude_vars_fill_value = {	'BLB': ['minimum_TBs', 'maximum_TBs', 'azimuth_angle', 'frequencies', 'elevation_scan_angles', 'TBs',
									'Min_TBs', 'Max_TBs', 'ElAngs', 'AzAng', 'Freq'],
							'BLH': ['BLH_data', 'minimum', 'maximum', 'Min_BLH', 'Max_BLH', 'BLH'],
							'BRT': ['frequencies', 'elevation_angle', 'azimuth_angle', 'TBs', 'Freq', 'AziAng', 'ElAng', 'Max_TBs', 'Min_TBs'],
							'CBH': ['minimum', 'maximum', 'CBH_data', 'Min_CBH', 'Max_CBH', 'CBH'],
							'HKD': ['longitude', 'latitude', 'ambient_Target1_temperature', 'ambient_Target2_temperature', 'receiver1_temperature', 
									'receiver2_temperature', 'stability_rec1', 'stability_rec2', 'AT1_T', 'AT2_T', 'Rec1_T', 'Rec2_T', 'Rec1_Stab', 
									'Rec2_Stab'],
							'IRT': ['minimum', 'maximum', 'frequencies', 'elevation_angle', 'azimuth_angle', 'IRR_data', 'Min_IRT',
									'Max_IRT', 'WavLens', 'ElAng', 'AziAng', 'IRR_Map'],
							'TPB': ['minimum_T', 'maximum_T', 'temperature_profiles', 'min_T', 'max_T', 'T_prof'],
						}
time_encoding = {	'BLB': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'BLH': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'BRT': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'CBH': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'HKD': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'IRT': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
					'TPB': {'units': "seconds since 1.1.2001, 00:00:00", 'dtype': "int32"},
				}


# loop through date range:
files_for_copy = ["BLB", "BLH", "BRT", "CBH", "HKD", "IRT", "TPB"]
date_now = date0_dt
while date_now <= date1_dt:
	print(date_now.strftime("%Y-%m-%d"))

	# extract month, day and set correct source and destination file path:
	yyyy_str = f"{date_now.year:04}"
	mm_str = f"{date_now.month:02}"
	dd_str = f"{date_now.day:02}"
	path_source = paths['source'] + f"{yyyy_str}/{mm_str}/{dd_str}/"
	path_dest = paths['dest'] + f"{yyyy_str}/{mm_str}/{dd_str}/"

	# create path_dest if not existing:
	path_dest_dir = os.path.dirname(path_dest)
	if not os.path.exists(path_dest_dir):
		os.makedirs(path_dest_dir)


	# find correct files:
	files_dict = dict()
	for ffc in files_for_copy:
		files_dict[ffc] = sorted(glob.glob(path_source + f"{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}.NC"))
		ele90_file = sorted(glob.glob(path_source + f"ELE90_{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}.NC"))
		if len(ele90_file) == 1:
			files_dict[ffc].append(ele90_file[0])
		elif len(ele90_file) > 1:
			pdb.set_trace()

		# open files if more than one exists. If only one exists, no actions needed:
		if len(files_dict[ffc]) > 1:
			DS = xr.open_mfdataset(files_dict[ffc], concat_dim='time', combine='nested', decode_times=False)
			DS = DS.sortby('time')

			# make sure that nothing else is changed:
			DS = modify_DS(DS, ffc, aux_dict)

			# export:
			# adapt fill values: Make sure that _FillValue is not added to certain variables!
			for kk in DS.variables:
				if kk in exclude_vars_fill_value[ffc]:
					DS[kk].encoding["_FillValue"] = None

			# encode time:
			DS['time'].attrs['units'] = time_encoding[ffc]['units']
			DS['time'].encoding['units'] = time_encoding[ffc]['units']
			DS['time'].encoding['dtype'] = time_encoding[ffc]['dtype']

			DS.to_netcdf(path_dest + f"{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}.NC", mode='w', format="NETCDF4")
			DS.close()

			print("Merged " + path_dest + f"{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}.NC")

		elif len(files_dict[ffc]) == 1: # RENAME ELE90 file
			DS = xr.open_dataset(files_dict[ffc][0], decode_times=False)
			DS = modify_single_DS(DS, ffc, aux_dict)

			# export:
			# adapt fill values: Make sure that _FillValue is not added to certain variables!
			for kk in DS.variables:
				if kk in exclude_vars_fill_value[ffc]:
					DS[kk].encoding["_FillValue"] = None

			# encode time:
			DS['time'].attrs['units'] = time_encoding[ffc]['units']
			DS['time'].encoding['units'] = time_encoding[ffc]['units']
			DS['time'].encoding['dtype'] = time_encoding[ffc]['dtype']

			DS.to_netcdf(path_dest + f"{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}.NC", mode='w', format="NETCDF4")
			DS.close()


	date_now += dt.timedelta(days=1)