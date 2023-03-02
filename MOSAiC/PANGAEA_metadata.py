import datetime as dt
import numpy as np
import xarray as xr
import os
import pdb
import glob
from data_tools import numpydatetime64_to_datetime


# Event list according to PANGAEA: https://www.pangaea.de/expeditions/byproject/MOSAiC
# also includes start and end times of events:
event_list = {'leg1': ['PS122/1_1-38', dt.datetime(2019,9,30,6,0,0), dt.datetime(2019,12,13,9,4,45)],
				'leg2': ['PS122/2_14-18', dt.datetime(2019,12,13,9,11,0), dt.datetime(2020,2,27,10,56,0)],
				'leg3': ['PS122/3_28-6', dt.datetime(2020,2,27,10,57,0), dt.datetime(2020,6,8,23,59,0)],
				'leg4_1': ['PS122/4_43-11', dt.datetime(2020,6,9,0,0,0), dt.datetime(2020,6,27,7,0,0)],
				'leg4_2': ['PS122/4_43-145', dt.datetime(2020,6,27,7,1,0), dt.datetime(2020,8,12,9,59,0)],
				'leg5': ['PS122/5_58-3', dt.datetime(2020,8,12,10,0,0), dt.datetime(2020,10,12,23,59,59)]}


def extract_metadata(
	files,
	path_output,
	dataset,
	**kwargs):

	"""
	Imports from all files (one after another), identifies the correct 'event',
	extracts metadata, and writes them to a text file.

	Parameters:
	-----------
	files : list of str
		Path and filename of all files from which metadata is to be extracted.
	path_output : str
		Path where the metadata will be saved to (as .txt).
	dataset : str
		Optional dataset specifier (for filename of the saved metadata).

	**kwargs:
	product : str
		Specifies the data product. Valid option depends on the dataset.
	"""

	metadata = {'event': list(), 'filename': list(), 'start_time': list(), 'start_lat': list(),
				'start_lon': list(), 'end_time': list(), 'end_lat': list(), 'end_lon': list()}
	for file in files:
		DS = xr.open_dataset(file)

		n_time = len(DS.time.values)
		time_dt = numpydatetime64_to_datetime(DS.time.values)

		current_date = time_dt[round(n_time/2)].date()

		# catch overlap of more than one event:
		if current_date == dt.date(2019,12,13):
			metadata['event'].append('PS122/1_1-38,PS122/2_14-18')

		elif current_date == dt.date(2020,2,27):
			metadata['event'].append('PS122/2_14-18,PS122/3_28-6')

		elif current_date == dt.date(2020,6,8):
			metadata['event'].append('PS122/3_28-6')

		elif current_date == dt.date(2020,6,9):
			metadata['event'].append('PS122/4_43-11')

		elif current_date == dt.date(2020,6,27):
			metadata['event'].append('PS122/4_43-11,PS122/4_43-145')

		elif current_date == dt.date(2020,8,12):
			metadata['event'].append('PS122/4_43-145,PS122/5_58-3')

		else:
			
			# identify unambiguous event:
			for event in event_list.keys():
				# matches one event:
				if np.all((time_dt >= event_list[event][1]) & (time_dt <= event_list[event][2])):
					metadata['event'].append(event_list[event][0])


		# append metadata:
		metadata['filename'].append(os.path.basename(file))
		metadata['start_time'].append(time_dt[0].strftime("%Y-%m-%dT%H:%M"))
		metadata['start_lat'].append(f"{DS.lat.values[0]:.5f}")
		metadata['start_lon'].append(f"{DS.lon.values[0]:.5f}")
		metadata['end_time'].append(time_dt[-1].strftime("%Y-%m-%dT%H:%M"))
		metadata['end_lat'].append(f"{DS.lat.values[-1]:.5f}")
		metadata['end_lon'].append(f"{DS.lon.values[-1]:.5f}")

		DS.close()


	n_files = len(metadata['filename'])
	for mkey in metadata.keys():
		assert len(metadata[mkey]) == n_files

	# reorganize list for file writing:
	metadata_write = list()
	for idx, file in enumerate(metadata['filename']):
		# list of all metadata for the current file:
		list_temp = [metadata[mkey][idx] for mkey in metadata.keys()]
		metadata_write.append(list_temp)

	# write to file:
	if 'product' in kwargs.keys():
		output_file = path_output + f"metadata_PANGAEA_{dataset}_{product}.txt"
	else:
		output_file = path_output + f"metadata_PANGAEA_{dataset}.txt"
	headerline = [mkey for mkey in metadata.keys()]
	with open(output_file, 'w') as f:
		f.write('\t'.join(headerline) + '\n')
		f.writelines('\t'.join(list_row) + '\n' for list_row in metadata_write)
	

###################################################################################################


"""
	Quick script to read out lat, lon and time at beginning and end of each dataset
	uploaded to PANGAEA (HATPRO L1 and L2, MiRAC-P L1 and L2).

	- Import data
	- read out information
	- save to tab limited text file
"""


path_data = {'hatpro_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/HATPRO_l1_v01/",
				'hatpro_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/HATPRO_l2_v01/",
				'mirac_l1': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/MiRAC-P_l1_v01/",
				'mirac_l2': "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/MiRAC-P_l2_v01/"}

path_output = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/"


# create path if non-existent:
output_dir = os.path.dirname(path_output)
if not os.path.exists(output_dir): os.makedirs(output_dir)


# Import data: Loop through datasets:
for key in path_data.keys():
	if key in ['mirac_l1', 'mirac_l2']:
		print(f"Extracting metadata from {key}.")
		files = sorted(glob.glob(path_data[key] + "*.nc"))
		extract_metadata(files, path_output, dataset=key)

	elif key == 'hatpro_l1':
		# Zenith mode:
		product = 'TB'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwr00_l1_tb_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

		# Boundary layer mode:
		product = 'TB_BL'
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwrBL00_l1_tb_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

	elif key == 'hatpro_l2':
		# LWP:
		product = 'CLWVI'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwr00_l2_clwvi_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

		# humidity profile:
		product = 'HUA'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwr00_l2_hua_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

		# IWV: 
		product = 'PRW'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwr00_l2_prw_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

		# temperature profile (zenith):
		product = 'TA'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwr00_l2_ta_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)

		# temperature profile (boundary layer):
		product = 'TA_BL'
		print(f"Extracting metadata from {key}, {product}.")
		files = sorted(glob.glob(path_data[key] + "ioppol_tro_mwrBL00_l2_ta_v01*.nc"))
		extract_metadata(files, path_output, dataset=key, product=product)


print("Done....")