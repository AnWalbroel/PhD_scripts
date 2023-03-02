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
	This code package aims to forward simulate dropsondes thrown out of HALO to brightness temperatures
	in the microwave spectrum (i.e., HAMP frequencies).
	- loop through campaign days (research flights)
	- load dropsonde data and fill measurement gaps for PAMTRA
	- simulate dropsondes
'''


# Dates:
which_dates = [
				"20220225",
				"20220311",
				"20220312",
				"20220313",
				"20220314",
				"20220315",
				"20220316",
				"20220320",
				"20220321",
				"20220328",
				"20220329",
				"20220330",
				"20220401",
				]

# Start and end dates:
date_start = datetime.datetime.strptime(which_dates[0], "%Y%m%d").date()
date_end = datetime.datetime.strptime(which_dates[-1], "%Y%m%d").date()
with_mirs_data = False


# Dictionary translating Research Flight numbers and dates:
RF_dict = {
			# '20220225': "RF00",
			# "20220311": "RF01",
			"20220312": "RF02",
			"20220313": "RF03",
			"20220314": "RF04",
			"20220315": "RF05",
			"20220316": "RF06",
			"20220320": "RF07",
			"20220321": "RF08",
			}


# Either manually select and download the SST data from the website given in
# the README file or assign latitude and longitude boundaries and required dates
# for an automated download:
path_sst = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/sst_slice/" # SST data path (data is downloaded here)
lat_bound = [63, 89]
lon_bound = [-30, 30]
# # # # # print("Running get_sst_data.py ..........\n")
# # # # # download_sst_data(path_sst, lat_bound, lon_bound, date_start, date_end)

if with_mirs_data:
	path_sfc_props = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/surface_props/"
else:
	path_sfc_props = None

# data set versions:
dropsonde_dataset = "raw"		# raw: QC Level 1 files after flight day available
BAH_version = "raw"				# raw: like coming out of HALO as netCDF (QL_HALO-AC3_HALO_BAHAMAS_yyyymmdd_RFnn_v1.nc)
								# unified: Heike Konow's unified data set product


now_date = date_start
for now_date in which_dates:

	# Identify RF:
	now_date_str = now_date
	if now_date_str in RF_dict.keys():
		RF_now = RF_dict[now_date_str]
	else:
		print(f"No RF on {now_date_str}.")
		continue

	# Paths:
	path_BAH_data = ("/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/" +
						f"HALO-AC3_HALO_BAHAMAS_{now_date_str}_{RF_now}/")
	path_dropsondes = ("/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/dropsondes/" +
						f"HALO-AC3_HALO_Dropsondes_{now_date_str}_{RF_now}/Level_1/")
	path_dropsondes_repaired = ("/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/repaired/" +
								f"HALO-AC3_HALO_Dropsondes_{now_date_str}_{RF_now}/")		# output dir
	path_pam_ds = ("/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/pam_out_sonde_art_cloud/" +
					f"HALO-AC3_HALO_Dropsondes_{now_date_str}_{RF_now}/")# output path of pamtra

	# check if dirs exist:
	if not os.path.exists(path_dropsondes_repaired):
		os.makedirs(path_dropsondes_repaired)
	if not os.path.exists(path_pam_ds):
		os.makedirs(path_pam_ds)

	# # # # # # "dropsonde_raw_gap_filler.py":
	# # # # # print("Running dropsonde_raw_gap_filler.py ..........\n")
	# # # # # run_dropsonde_gap_filler(path_dropsondes, path_dropsondes_repaired, dropsonde_dataset, path_BAH_data, BAH_version)


	# Fwd simulate dropsondes:
	print("Running HALO_raw_dropsonde_to_TB.py ..........\n")
	run_HALO_raw_dropsonde_to_TB(path_dropsondes_repaired, path_sst, path_pam_ds,
		obs_height='BAHAMAS', path_BAH_data=path_BAH_data, BAH_version=BAH_version,
		path_sfc_props=path_sfc_props,
		)
