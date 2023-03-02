from __future__ import print_function, division
import numpy as np
import os
import datetime
import glob
import netCDF4 as nc
import pdb
import pandas as pd
from concat_mwr import run_concat_mwr
from dropsonde_raw_gap_filler import run_dropsonde_gap_filler
from get_sst_data import download_sst_data
from HALO_raw_dropsonde_to_TB import run_HALO_raw_dropsonde_to_TB
from TB_statistics_raw import run_TB_statistics_raw
from TB_statistics_daily import run_TB_statistics_daily

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
path_BAH_data = "/data/obs/campaigns/halo-ac3/halo/BAHAMAS/unified/"	# BAHAMAS data path
path_sonde_data = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/unified/"	# may contain subfolders
sonde_dataset_type = "unified"
path_sonde_repaired = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/gap_filled/" # output dir

print("Running dropsonde_raw_gap_filler.py ..........\n")
run_dropsonde_gap_filler(path_sonde_data, path_sonde_repaired, path_sonde_data, path_BAH_data)


#	3. Either manually select and download the SST data from the website given in
#	the README file or assign latitude and longitude boundaries and required dates
#	for an automated download:
path_sst = "/net/blanc/awalbroe/Data/HALO_AC3/sst_slice/" # SST data path (data is downloaded here)
lat_bound = [65, 82]
lon_bound = [-30, 30]
start_date = "2022-03-11"
end_date = "2022-04-14"
print("Running get_sst_data.py ..........\n")
download_sst_data(path_sst, lat_bound, lon_bound, start_date, end_date)


#	4. "HALO_raw_dropsonde_to_TB.py":
path_sonde_repaired = path_sonde_repaired	# interpolated dropsonde output path
path_sonde_sim = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/fwd_sim_dropsondes/"  # output path of pamtra
print("Running HALO_raw_dropsonde_to_TB.py ..........\n")
run_HALO_raw_dropsonde_to_TB(path_sonde_repaired, path_sst, path_sonde_sim,
	obs_height='BAHAMAS', path_BAH_data=path_BAH_data)



#	5. "TB_statistics_raw.py":
out_path = "/work/mjacob/CSSC_test/sonde_comparison_half_raw/"
plot_path = "/work/mjacob/CSSC_test/plots/TB_stat/"


bias_ev_plotname = "TB_abs_biasevolution_ALL_J3v0.9.2_radar"
output_filename = "clear_sky_sonde_comparison_ALL_J3v0.9.2_radar"
scatterplot_name = "TB_scatterplot_ALL_J3v0.9.2_radar"
mwr_concat_path = "/data/hamp/flights/EUREC4A/unified/radiometer_*_v0.8" # This is the unified MWR data

path_RADAR_data = "/data/hamp/flights/EUREC4A/unified/radar_*v0.6"	# RADAR data path; OPTIONAL
print("Running TB_statistics_raw.py ..........\n")
run_TB_statistics_raw(mwr_concat_path, path_sonde_sim, out_path, plot_path, scatterplot_name,
	bias_ev_plotname, output_filename,
	obs_height='BAHAMAS',
	path_BAH_data=path_BAH_data,
	path_RADAR_data=path_RADAR_data,
)

print("Running TB_statistics_daily.py ..........\n")
run_TB_statistics_daily(
	out_path+output_filename+'.nc',
	out_path+output_filename+'_daily.nc'
)

dates = [
	'20200119',
	'20200122',
	'20200124',
	'20200126',
	'20200128',
	'20200130',
	'20200131',
	'20200202',
	'20200205',
	'20200207',
	'20200209',
	'20200211',
	'20200213',
	'20200215',
	'20200218',
]
print("Running TB_statistics_raw.py for each day ..........\n")

for date in dates:
	bias_ev_plotname = f"TB_abs_biasevolution_ALL_J3v0.9.2_radar_{date}"
	output_filename = f"clear_sky_sonde_comparison_ALL_J3v0.9.2_radar_{date}"
	scatterplot_name = f"TB_scatterplot_ALL_J3v0.9.2_radar_{date}"

	# restrict dropsonde and MWR data to one day
	path_sonde_sim = f"/net/gharra/mjacob/CSSC_test/pam_out_sonde_JOANNE_interpolated_v0.9.2/pamtra_*{date}"
	mwr_concat_path = f"/data/hamp/flights/EUREC4A/unified/radiometer_{date}_v0.8"

	print(f"Running TB_statistics_raw.py for {date} ..........\n")
	run_TB_statistics_raw(mwr_concat_path, path_sonde_sim, out_path, plot_path, scatterplot_name,
		bias_ev_plotname, output_filename,
		obs_height='BAHAMAS',
		#obs_height=9600,
		path_BAH_data=path_BAH_data,
		path_RADAR_data=path_RADAR_data,
	)
