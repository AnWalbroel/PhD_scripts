import numpy as np
import xarray as xr
import datetime as dt
from import_data import import_mvr_pro_rt
from data_tools import compute_DOY
import pdb
import glob
import os


"""
	This script combines the training data files for HATPRO's MWR_PRO retrieval and MiRAC-P's NN
	retrievalinto one file for each instrument.
"""

aux_info = dict()					# dictionary saving auxiliary information for this script

path_data = {'hatpro': "/net/blanc/awalbroe/Data/TTT/hatpro_mvr/",
				'mirac-p': "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/combined/"}
path_output = {'hatpro': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/retrieval_training/hatpro/",
				'mirac-p': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/retrieval_training/mirac-p/"}

# for aux_info['instrument'] in ['hatpro', 'mirac-p']:	# specify which instrument shall be processed
for aux_info['instrument'] in ['mirac-p']:	# specify which instrument shall be processed


	# Load data:
	if aux_info['instrument'] == 'hatpro':
		keys = ['station_id', 'cap_height_above_sea_level', 'frequency', 'elevation_angle', 
				'height_grid', 'date', 'atmosphere_temperature', 'atmosphere_humidity', 'integrated_water_vapor', 
				'liquid_water_path', 'brightness_temperatures']
		DS = import_mvr_pro_rt(path_data[aux_info['instrument']], keys, instrument=aux_info['instrument'])

		# compose new dataset with (mainly) new variable names:
		DS_new = xr.Dataset({'freq_sb': (['n_freq'], DS.frequency.values,
										{'units': "GHz",
										'standard_name': "sensor_band_central_radiation_frequency",
										'long_name': "simulated frequency"}),
							'ele':		(['n_angle'], DS.elevation_angle.values,
										{'units': "degree",
										'long_name': "elevation angle"}),
							'height':	(['height'], DS.height_grid.values,
										{'units': "m",
										'standard_name': "height"}),
							'ta':		(['time', 'height'], DS.atmosphere_temperature.values,
										{'units': "K",
										'standard_name': "air_temperature",
										'long_name': "air temperature on height grid"}),
							'hua':		(['time', 'height'], DS.atmosphere_humidity.values,
										{'units': "kg m-3",
										'long_name': "absolute humidity on height grid"}),
							'prw':		(['time'], DS.integrated_water_vapor.values,
										{'units': "kg m-2",
										'standard_name': "atmosphere_mass_content_of_water_vapor",
										'long_name': "integrated water vapor or precipitable water"}),
							'clwvi':	(['time'], DS.liquid_water_path.values,
										{'units': "kg m-2",
										'standard_name': "atmosphere_mass_content_of_cloud_liquid_water_content",
										'long_name': "liquid water path"}),
							'tb':		(['time', 'n_angle', 'n_freq'], DS.brightness_temperatures.values,
										{'units': "K",
										'standard_name': "brightness_temperature",
										'long_name': "brightness temperatures"})},
							coords=		{'time': 	(['time'], DS.time.values,
													{'units': "seconds since 1970-01-01 00:00:00 UTC",
													'standard_name': "time"})})

		# attributes:
		DS_new.attrs['Title'] = "Training and test data for HATPRO retrievals"
		DS_new.attrs['Data_based_on'] = "Ny-Alesund radiosondes"
		DS_new.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
		DS_new.attrs['Contact_person'] = "Kerstin Ebell (kebell@meteo.uni-koeln.de), Andreas Walbröl (a.walbroel@uni-koeln.de)"
		DS_new.attrs['Conventions'] = "CF-1.6"
		DS_new.attrs['License'] = "For non-commercial use only."
		DS_new.attrs['Station_id'] = "nya"
		DS_new.attrs['Location'] = "Ny-Alesund"
		DS_new.attrs['Latitude'] = "78.92"
		DS_new.attrs['Longitude'] = "11.92"
		DS_new.attrs['Altitude'] = "15.7"
		DS_new.attrs['System'] = "radiatve transfer calculations using STP RT modules"
		DS_new.attrs['Predictors'] = "tb"
		DS_new.attrs['Predictands'] = "prw, clwvi, ta, hua"
		DS_new.attrs['Years_training'] = "2006-2017"
		DS_new.attrs['Years_testing'] = "2006-2017"

		# Make sure that _FillValue is not added:
		for kk in DS_new.variables:
			DS_new[kk].encoding["_FillValue"] = None

		# encode time:
		DS_new['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
		DS_new['time'].encoding['dtype'] = 'double'

		DS_new.to_netcdf(path_output[aux_info['instrument']] + "MOSAiC_hatpro_retrieval_nya_v00" + ".nc", mode='w', format="NETCDF4")
		DS_new.close()
		DS.close()


	if aux_info['instrument'] == 'mirac-p':
		keys = ['station_id', 'cap_height_above_sea_level', 'frequency', 'elevation_angle', 
				'height_grid', 'date', 'integrated_water_vapor', 'brightness_temperatures']
		DS = import_mvr_pro_rt(path_data[aux_info['instrument']], keys, instrument=aux_info['instrument'])

		DOY_1, DOY_2 = compute_DOY(DS.time.values, return_dt=False, reshape=False)

		# compose new dataset with (mainly) new variable names:
		DS_new = xr.Dataset({'freq_sb': (['n_freq'], DS.frequency.values,
										{'units': "GHz",
										'standard_name': "sensor_band_central_radiation_frequency",
										'long_name': "simulated frequency"}),
							'ele':		(['n_angle'], DS.elevation_angle.values,
										{'units': "degree",
										'long_name': "elevation angle"}),
							'prw':		(['time'], DS.integrated_water_vapor.values,
										{'units': "kg m-2",
										'standard_name': "atmosphere_mass_content_of_water_vapor",
										'long_name': "integrated water vapor or precipitable water"}),
							'tb':		(['time', 'n_angle', 'n_freq'], DS.brightness_temperatures.values,
										{'units': "K",
										'standard_name': "brightness_temperature",
										'long_name': "brightness temperatures"}),
							'cos_doy':	(['time'], DOY_1,
										{'units': "dimensionless",
										'long_name': "COS of the day of the year"}),
							'sin_doy':	(['time'], DOY_2,
										{'units': "dimensionless",
										'long_name': "SIN of the day of the year"})},
							coords=		{'time': 	(['time'], DS.time.values,
													{'units': "seconds since 1970-01-01 00:00:00 UTC",
													'standard_name': "time"})})

		# attributes:
		DS_new.attrs['Title'] = "Training and test data for MiRAC-P (LHUMPRO-243-340) retrievals"
		DS_new.attrs['Data_based_on'] = "ERA-Interim"
		DS_new.attrs['Institution'] = "RPG Radiometer Physics GmbH, Meckenheim, Germany; Insititute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
		DS_new.attrs['Contact_person'] = ("Emiliano Orlandi (Emiliano.Orlandi@radiometer-physics.de), " +
											"Andreas Walbröl (a.walbroel@uni-koeln.de)")
		DS_new.attrs['Conventions'] = "CF-1.6"
		DS_new.attrs['License'] = "For non-commercial use only."
		DS_new.attrs['Station_id'] = "pol"
		DS_new.attrs['Location'] = "Arctic Sea"
		DS_new.attrs['Latitude'] = "84.75, 84.75, 84.75, 84.75, 87.00, 87.00, 89.25, 89.25"
		DS_new.attrs['Longitude'] = "5.25, 90.00, 180.00, 270.00, 45.00, 225.00, 5.25, 5.25"
		DS_new.attrs['Altitude'] = "0.0"
		DS_new.attrs['System'] = "radiatve transfer calculations have been carried out by RPG Radiometer Physics GmbH"
		DS_new.attrs['Predictors'] = "tb, cos_doy, sin_doy"
		DS_new.attrs['Predictands'] = "prw"
		DS_new.attrs['Years_training'] = "2001, 2002, 2004, 2006, 2007, 2008, 2009, 2011, 2012, 2013, 2015, 2017"
		DS_new.attrs['Years_testing'] = "2003, 2005, 2010, 2014, 2016"

		# Make sure that _FillValue is not added:
		for kk in DS_new.variables:
			DS_new[kk].encoding["_FillValue"] = None

		# encode time:
		DS_new['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
		DS_new['time'].encoding['dtype'] = 'double'

		DS_new.to_netcdf(path_output[aux_info['instrument']] + "MOSAiC_mirac-p_retrieval_pol_v00" + ".nc", mode='w', format="NETCDF4")
		DS_new.close()
		DS.close()