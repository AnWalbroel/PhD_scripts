import numpy as np
import os
import datetime
import glob
import netCDF4 as nc
import pdb
import pandas as pd
import xarray as xr

from CSSC_src import CSSC


os.environ['OPENBLAS_NUM_THREADS'] = "1" # workaround for pamtra and openblas, required if PAMTRA was compiled with multithreadding version of openBLAS

'''
	This is the control room for the Clear Sky Sonde Comparison (CSSC) code package where
	you assign the required variables (mostly paths of data).
	More information about each to be manually assigned variable can be found in
	'README.md'.

	This version of main uses unified MWR data and JOANNE drop sonde data
'''


# Define paths:
path_data = {	'BAH': "/data/obs/campaigns/halo-ac3/halo/BAHAMAS/unified/",	# BAHAMAS data path
				'mwr_concat': "/net/blanc/awalbroe/Data/HALO_AC3/HALO/HAMP/unified/", # This is the unified MWR data
				'radar': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/hamp_mira/",	# RADAR data path; OPTIONAL
				'dropsonde': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/dropsondes/",
				'dropsonde_rep': "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/gap_filled/", # repaired dropsonde path
				'sst': "/net/blanc/awalbroe/Data/HALO_AC3/sst_slice/", # SST data path (data is downloaded here)
				'dropsonde_sim': "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/fwd_sim_dropsondes/",  # output path of pamtra
				'cssc_output': "/net/blanc/awalbroe/Data/HALO_AC3/HALO/CSSC/",		# output for sonde - mwr comparison
			}
path_plot = "/net/blanc/awalbroe/Plots/HALO_AC3/CSSC/"


# dictionary of settings to control CSSC:
set_dict = {'sonde_dataset_type': "raw",		# dropsonde dataset type; options, see README
			'sonde_height_grid': np.arange(0.0, 16000.0001, 10.0),		# height grid in m
			'sst_lat': [65, 82],		# lat boundaries of sst data (for download)
			'sst_lon': [-30, 30],		# lon boundaries of sst data (for download)
			'start_date': "2022-03-11",	# first date sst data should cover (for download), in yyyy-mm-dd
			'end_date': "2022-04-14",	# last date sst data should cover (for download), in yyyy-mm-dd
			}


# initialise CSSC:
cssc_main = CSSC(path_data, path_plot, set_dict)


# 1. repair dropsondes with dropsonde_gap_filler:
print("Repairing dropsondes (filling gaps)....\n")
cssc_main.dropsonde_gap_filler()


# 2. Either manually select and download the SST data from the website given in
# the README file or assign latitude and longitude boundaries and required dates
# for an automated download:
print("Getting SST data....\n")
cssc_main.download_sst_data()


# 3. Simulate repaired dropsondes with PAMTRA to obtain brightness temperatures for clear sky scenes:
print("Forward simulating dropsondes with PAMTRA....\n")
cssc_main.fwd_sim_dropsondes_to_TB()


# 4. Identify clear sky scenes and compare simulated and measured brightness temperatures (TBs) 
# for each research flight (time boundaries given by radiometer time).
print("Comparing simulated and measured TBs....\n")
cssc_main.TB_comparison()


# 5. Compute offsets (and slopes) of measured TBs based on the output of TB_comparison for each
# research flight.
print("Computing offsets for measured TBs....\n")
cssc_main.get_TB_offsets()


# 6. (optional): Visualise TB comparison for corrected TB measurements:
print("Visualising offset corrected TBs compared to simulated TBs....\n")
cssc_main.visualise_TB_offsets()