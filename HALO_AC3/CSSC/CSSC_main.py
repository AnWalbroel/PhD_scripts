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

os.environ['OPENBLAS_NUM_THREADS'] = "1" # workaround for pamtra and openblas, required if PAMTRA was compiled with multithreadding version of openBLAS

'''
	This is the control room for the Clear Sky Sonde Comparison (CSSC) code package where
	you assign the required variables (mostly paths of data).
	More information about each to be manually assigned variable can be found in
	'README.md'.
'''


# 	1. "concat_mwr.py": Due to changing conventions how the MWR data is saved you have to
# 	manually edit some lines in "concat_mwr.py" (marked with "##################" and
# 	explained in the README file).
mwr_base_path = "/data/hamp/flights/EUREC4A/"
day_folders = sorted(glob.glob(mwr_base_path + "*"))[0:17]	# all day folders will be saved here
mwr_concat_path = "/net/sever/walbroel/CSSC_test/halo/mwr/"
print("Running concat_mwr.py ..........\n")
run_concat_mwr(day_folders, mwr_concat_path)



# 2. "dropsonde_raw_gap_filler.py":
path_raw_sondes = "/net/sever/walbroel/data/sonde_half_raw/"
path_halo_dropsonde = "/net/sever/walbroel/CSSC_test/sonde_qc_interpolated_v01_BAH/"
dropsonde_dataset = "raw"
path_BAH_data = "/data/hamp/flights/EUREC4A/unified/"	# BAHAMAS data path; OPTIONAL
print("Running dropsonde_raw_gap_filler.py ..........\n")
run_dropsonde_gap_filler(path_raw_sondes, path_halo_dropsonde, dropsonde_dataset, path_BAH_data)



#	3. Either manually select and download the SST data from the website given in
#	the README file or assign latitude and longitude boundaries and required dates
#	for an automated download:
path_sst = "/net/sever/walbroel/CSSC_test/sst_slice/"
lat_bound = [0, 40]
lon_bound = [-70, 0]
start_date = "2020-01-19"
end_date = "2020-02-18"
print("Running get_sst_data.py ..........\n")
download_sst_data(path_sst, lat_bound, lon_bound, start_date, end_date)



#	4. "HALO_raw_dropsonde_to_TB.py":
path_halo_dropsonde = path_halo_dropsonde	# interpolated dropsonde output path
path_sst = path_sst		# SST data path
path_pam_ds = "/net/sever/walbroel/CSSC_test/pam_out_BAH/"  # output path of pamtra
print("Running HALO_raw_dropsonde_to_TB.py ..........\n")
run_HALO_raw_dropsonde_to_TB(path_halo_dropsonde, path_sst, path_pam_ds,
	obs_height='BAHAMAS', path_BAH_data=path_BAH_data)
	# obs_height=10e3)



#	5. "TB_statistics_raw.py":
out_path = "/net/sever/walbroel/CSSC_test/sonde_comparison_half_raw_BAH/"
plot_path = "/net/sever/walbroel/CSSC_test/plots/TB_stat_BAH/"
scatterplot_name = "TB_scatterplot"
bias_ev_plotname = "TB_abs_biasevolution"
output_filename = "clear_sky_sonde_comparison"
print("Running TB_statistics_raw.py ..........\n")
run_TB_statistics_raw(mwr_concat_path, path_pam_ds, out_path, plot_path, scatterplot_name,
	bias_ev_plotname, output_filename,
	obs_height='BAHAMAS', path_BAH_data=path_BAH_data)
	# obs_height=10e3,)