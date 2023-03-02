import numpy as np
import glob
import pdb
import os
import datetime as dt
from import_data import *
from data_tools import *

"""
	In this program, output files from the mwr_pro tool are imported and lat, lon
	coordinates from Polarstern are added to them and saved into the output files.
	Input files can be netCDF files of level 1 (TBs) or level 2 (e.g. IWV, LWP,
	temperature, humidity profile, ...).
"""

# Data paths:
path_hatpro_level1 = "/data/obs/campaigns/mosaic/hatpro/l1/"
path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"
path_ps_track = "/data/obs/campaigns/mosaic/polarstern_track/"

# Specify date range:
date_start = "2019-09-20"
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


# the plan: first: consider level 1b and level 1c files. Afterwards, go through each level 2b and 2c file.
# When starting with 1b: cycle through all days and months within the date_start and date_end date.
# Then load the level 1b file as xarray dataset and interpolate the lat, lon from Polarstern track onto
# the time axis of the level 1b file. Finally, insert the lat and lon data and save it as netCDF again
# without changing the structure or touching anything else!
write_polarstern_track_data_into_mwr_pro_output(path_hatpro_level1, ps_track_dict, date_start, 
												date_end, instrument='hatpro', level='level_1',
												verbose=1)

# For level 2 hatpro data:
write_polarstern_track_data_into_mwr_pro_output(path_hatpro_level2, ps_track_dict, date_start, 
												date_end, instrument='hatpro', level='level_2',
												verbose=1)
# pdb.set_trace()