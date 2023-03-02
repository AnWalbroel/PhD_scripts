from shutil import copyfile
import xarray as xr
import datetime as dt
import numpy as np
import glob
import os
import pdb
from data_tools import datetime_to_epochtime


"""
	Script to remove time stamps within the Exclusive Economic Zones (EEZ, 
	data within these regions may not be published).
	- import data
	- find and remove time steps
	- export data
"""


path_data = {'hatpro_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l1_v01/",
				'hatpro_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/",
				'mirac-p_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/MiRAC-P_l1_v01/",
				'mirac-p_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/MiRAC-P_l2_v01/"}
path_output = {'hatpro_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/HATPRO_l1_v01/",
				'hatpro_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/HATPRO_l2_v01/",
				'mirac-p_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/MiRAC-P_l1_v01/",
				'mirac-p_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/MiRAC-P_l2_v01/"}

# check if folders for output exist. if they don't create them
for key in path_output.keys():
	path_output_dir = os.path.dirname(path_output[key])
	if not os.path.exists(path_output_dir):
		os.makedirs(path_output_dir)

# EEZ periods during MOSAiC:
EEZ_periods_dt = {'range0': [dt.datetime(2020,6,3,20,36), 
							dt.datetime(2020,6,8,20,0)],
				'range1': [dt.datetime(2020,10,2,4,0), 
							dt.datetime(2020,10,2,20,0)],
				'range2': [dt.datetime(2020,10,3,3,15), 
							dt.datetime(2020,10,4,17,0)]}
EEZ_periods = dict()
EEZ_periods_npdt = dict()
for key in EEZ_periods_dt:
	EEZ_periods[key] = [datetime_to_epochtime(EEZ_periods_dt[key][0]), 
						datetime_to_epochtime(EEZ_periods_dt[key][1])]
	EEZ_periods_npdt[key] = np.array([EEZ_periods[key][0], EEZ_periods[key][1]]).astype("datetime64[s]")


# create date boundaries to check:
date_start_dt = dt.date(2019,9,20)
date_end_dt = dt.date(2020,10,12)


# choose which data shall be edited:
for this_data in ["hatpro_l1", "hatpro_l2", "mirac-p_l1", "mirac-p_l2"]:
# for this_data in ["mirac-p_l1", "mirac-p_l2"]:


	# Import data:
	# Cycle through days and check if file for that day exists. If true then check if 
	# that day is within EEZ:
	now_date = date_start_dt
	while now_date <= date_end_dt:

		print(now_date)
		# check if now_date is within an EEZ period:
		isitineez = 0			# if 0: outside any eez
		which_eez = -1		# indicates which period; if -1: now_date not within eez
		for k, key in enumerate(EEZ_periods_dt.keys()):
			eez0 = EEZ_periods_dt[key][0]
			eez1 = EEZ_periods_dt[key][1]
			if (now_date >= eez0.date()) and (now_date <= eez1.date()): 
				isitineez += 1
				which_eez = k

		yyyy = now_date.year
		mm = now_date.month
		dd = now_date.day

		if isitineez == 1: # then now_date is in a eez period

			files = glob.glob(path_data[this_data] + f"*_v01_{yyyy:04}{mm:02}{dd:02}*.nc")

			if len(files) > 0:	# then loop over files
				for file in files:
					# import data and find times outside EEZ:
					DS = xr.open_dataset(file, decode_times=False)
					# outsize_eez_idx = np.where((DS.time.values < EEZ_periods_npdt[f"range{which_eez}"][0])
												# | (DS.time.values > EEZ_periods_npdt[f"range{which_eez}"][1]))[0]
					outsize_eez_idx = np.where((DS.time.values < EEZ_periods[f"range{which_eez}"][0])
												| (DS.time.values > EEZ_periods[f"range{which_eez}"][1]))[0]
					
					DS = DS.isel(time=outsize_eez_idx)

					# adapt fill values:
					# Make sure that _FillValue is not added to certain variables:
					exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl', 'freq_sb', 'wl_irp', 'time_bnds',
												]
					for vava in DS.variables:
						if vava in exclude_vars_fill_value:
							DS[vava].encoding["_FillValue"] = None

					# update global attributes:
					DS.attrs['time_start'] = dt.datetime.utcfromtimestamp(DS.time.values[0]).strftime("%Y-%m-%d %H:%M:%SZ")
					DS.attrs['time_end'] = dt.datetime.utcfromtimestamp(DS.time.values[-1]).strftime("%Y-%m-%d %H:%M:%SZ")

					# encode time and give it back its attributes:
					DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00 UTC"
					DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
					DS['time'].encoding['dtype'] = 'double'

					print("Filtering....", file)
					if len(DS.time.values) > 0:		# else: xarray runs into errors
						DS.to_netcdf(path_output[this_data] + os.path.basename(file), mode='w', format="NETCDF4")
						DS.close()

		elif isitineez > 1:
			raise ValueError("Unexpected number of EEZ periods covered.")

		else:	# copy file
			fyles = glob.glob(path_data[this_data] + f"*_v01_{yyyy:04}{mm:02}{dd:02}*.nc")

			if len(fyles) > 0:
				for fyle in fyles: copyfile(fyle, path_output[this_data] + os.path.basename(fyle))
				



		now_date += dt.timedelta(days=1)


print("Done....")