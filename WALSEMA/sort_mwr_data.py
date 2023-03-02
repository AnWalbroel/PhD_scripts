import numpy as np
import datetime as dt
import shutil
import glob
import os
import pdb
import sys


"""
	Script to copy and sort MWR netCDF data into atmosphere and ice mode. Call script via 
	"python3 sort_mwr_data.py" or "python3 sort_mwr_data.py 'hatpro'" or
	"python3 sort_mwr_data.py 'mirac-p'".
	- Specify source and destination directories
"""


aux_dict = dict()		# contains extra info
paths = dict()

if len(sys.argv) == 1:
	aux_dict['instr'] = 'hatpro'
elif len(sys.argv) == 2:
	aux_dict['instr'] = sys.argv[1]


# source and destination paths:
paths['source'] = f"/data/obs/campaigns/WALSEMA/{aux_dict['instr']}/l0/"		# source of TB data
paths['dest'] = f"/data/obs/campaigns/WALSEMA/atm/{aux_dict['instr']}/l1/"		# dest of TB data


# start and end date:
date0 = "2022-06-28"
date1 = "2022-08-12"
date0_dt = dt.datetime.strptime(date0, "%Y-%m-%d")
date1_dt = dt.datetime.strptime(date1, "%Y-%m-%d")


# source path has got sub-directories (each day) that must be looped over:
date_now = date0_dt
if aux_dict['instr'] == 'hatpro':
	files_for_copy = ["BLB", "BLH", "CBH", "HKD", "BRT", "IRT", "TPB"]
else:
	files_for_copy = ["BLB", "BLH", "CBH", "HKD", "BRT", "IRT", "TPB", "MET", "LV0"]
while date_now <= date1_dt:
	print(date_now.strftime("%Y-%m-%d"))

	# extract month, day and set correct source and destination file path:
	yyyy_str = f"{date_now.year:04}"
	mm_str = f"{date_now.month:02}"
	dd_str = f"{date_now.day:02}"
	path_source = paths['source'] + f"Y{yyyy_str}/M{mm_str}/D{dd_str}/"
	path_dest = paths['dest'] + f"{yyyy_str}/{mm_str}/{dd_str}/"


	# identify correct atm files:
	files_dict = dict()
	files_filtered_dict = dict()
	for ffc in files_for_copy:
		files_dict[ffc] = sorted(glob.glob(path_source + f"*{yyyy_str[2:]}{mm_str}{dd_str}.{ffc}"))		# NON-NETCDFs
		# files_dict[ffc] = sorted(glob.glob(path_source + f"*.{ffc}.NC"))	# NETCDFs
		files_filtered_dict[ffc] = list()
		print(f"Found files: {ffc}: {len(files_dict[ffc])}\n")

		# scan for ELE40 (not atm mode):
		for file in files_dict[ffc]:
			if not (("ELE40" in file) or ("ELE100" in file) or ("ELE110" in file) or ("ELE120" in file) or ("ELE130" in file) or ("ELE140" in file) or ("ELE150" in file) or ("ELE160" in file)):
				files_filtered_dict[ffc].append(file)


	# create destin. directory if not existing
	paths_dest_dir = os.path.dirname(path_dest)
	if not os.path.exists(paths_dest_dir):
		os.makedirs(paths_dest_dir)

	# copy files if files were found:
	for ffc in files_for_copy:
		for file in files_filtered_dict[ffc]:
			old_filename = os.path.basename(file)

			shutil.copyfile(file, path_dest + old_filename)
			print(f"Copied to {path_dest}:   {old_filename} \n")

	date_now += dt.timedelta(days=1)