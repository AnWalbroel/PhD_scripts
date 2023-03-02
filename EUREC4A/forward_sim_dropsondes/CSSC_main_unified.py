from __future__ import print_function, division
import numpy as np
import os
import datetime
import glob
import netCDF4 as nc
import pdb
import pandas as pd
from dropsonde_raw_gap_filler import run_dropsonde_gap_filler
from get_sst_data import download_sst_data
from HALO_raw_dropsonde_to_TB import run_HALO_raw_dropsonde_to_TB

os.environ['OPENBLAS_NUM_THREADS'] = "1" # workaround for pamtra and openblas, required if PAMTRA was compiled with multithreadding version of openBLAS

'''
	This is the control room for the Clear Sky Sonde Comparison (CSSC) code package where
	you assign the required variables (mostly paths of data).
	More information about each to be manually assigned variable can be found in
	'README.md'.

	This version of main uses unified MWR data and JOANNE drop sonde data
'''


#	1. "concat_mwr.py": not required when using unified MWR data


#	2. "dropsonde_raw_gap_filler.py":
path_BAH_data = "/data/obs/campaigns/eurec4a/HALO/unified/v0.9/"	# BAHAMAS data path
path_joanne3_sondes = "/data/obs/campaigns/eurec4a/HALO/JOANNE/EUREC4A_JOANNE*"
dropsonde_dataset = "joanne_level_3"
path_halo_dropsonde = "/net/blanc/awalbroe/Data/EUREC4A/forward_sim_dropsondes/sonde_v01/" # output dir

# # # print("Running dropsonde_raw_gap_filler.py ..........\n")
# # # run_dropsonde_gap_filler(path_joanne3_sondes, path_halo_dropsonde, dropsonde_dataset, path_BAH_data)


#	3. Either manually select and download the SST data from the website given in
#	the README file or assign latitude and longitude boundaries and required dates
#	for an automated download:
path_sst = "/net/blanc/awalbroe/Data/EUREC4A/forward_sim_dropsondes/sst_slice/" # SST data path (data is downloaded here)
lat_bound = [0, 40]
lon_bound = [-70, 0]
start_date = "2020-01-19"
end_date = "2020-02-18"
# # # print("Running get_sst_data.py ..........\n")
# # # download_sst_data(path_sst, lat_bound, lon_bound, start_date, end_date)



#	4. "HALO_raw_dropsonde_to_TB.py":
path_halo_dropsonde = path_halo_dropsonde	# interpolated dropsonde output path
path_pam_ds = "/net/blanc/awalbroe/Data/EUREC4A/forward_sim_dropsondes/pam_out_sonde/"  # output path of pamtra
print("Running HALO_raw_dropsonde_to_TB.py ..........\n")
run_HALO_raw_dropsonde_to_TB(path_halo_dropsonde, path_sst, path_pam_ds,
	obs_height='BAHAMAS', path_BAH_data=path_BAH_data)

