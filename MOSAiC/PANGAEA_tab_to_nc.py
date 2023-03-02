import numpy as np
import glob
import pdb
import os
import datetime as dt
from import_data import *
from data_tools import *

"""
	Convert PANGAEA .tab files to netcdf.
"""

path_radiosondes = "/net/blanc/awalbroe/Data/MOSAiC_radiosondes/"
path_pstrack = "/net/blanc/awalbroe/Data/MOSAiC_polarstern_track/"


# radiosonde data:
radiosonde_files = sorted(glob.glob(path_radiosondes + "*.tab"))
for rs_file in radiosonde_files:
	print(rs_file)
	rs_dict, rs_att_info = import_MOSAiC_Radiosondes_PS122_Level2_tab(rs_file)

	# Save each radiosonde in a single file:
	n_sondes = len(rs_dict.keys())
	for k in range(n_sondes):
		launch_time_str = dt.datetime.strftime(rs_dict[str(k)]['time'][0], "%Y%m%d_%H%M%SZ")
		export_file = path_radiosondes + "PS122_mosaic_radiosonde_level2_" + launch_time_str + ".nc"
		save_MOSAiC_Radiosondes_PS122_Level2_as_nc(export_file, rs_dict[str(k)], rs_att_info)

# # Polarstern master track data:
# pstrack_files = sorted(glob.glob(path_pstrack + "*.tab"))	# should only be one file
# for pstrack_file in pstrack_files:
	# pstrack_dict, pstrack_att_info = import_PS_mastertrack_tab(pstrack_file)

	# ps_export_filename = os.path.join(path_pstrack, os.path.basename(pstrack_file)[:-4] + ".nc")
	# save_PS_mastertrack_as_nc(ps_export_filename, pstrack_dict, pstrack_att_info)