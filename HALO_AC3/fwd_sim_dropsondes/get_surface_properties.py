import os
import urllib
import pdb
import shutil
import urllib.request as request
from contextlib import closing


"""
	Small script to download orders from NOAA after submitting an order:
	NOAA JPSS Microwave Integrated Retrieval System (MIRS) Advanced 
	Technology Microwave Sounder (ATMS) Precipitation and Surface Products from NDE. 
	[indicate subset used]. NOAA National Centers for Environmental Information. 
	doi:10.7289/V51V5C1X
"""

# information about the paths:
order_id = 8182904089
path_output = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/surface_props/"
path_file_list = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/surface_props/"

host = "ftp.avl.class.noaa.gov"
user = "anonymous"
pw = "user@internet"

if not os.path.exists(path_output):
	os.makedirs(path_output)


# read out the list of files copied from the download link provided via e-mail:
file_list = list()
file_list_file = path_file_list + "noaa_files.txt"
with open(file_list_file, "r") as f_handler:
	lines = f_handler.readlines()
	for line in lines:
		file_list.append(line.split()[0])


# Get data:
n_files = len(file_list)
for ii, file in enumerate(file_list):
	print(f"{100*(ii+1)/n_files:.1f}%, now downloading {file}")
	with closing(request.urlopen(f"ftp://{user}:{pw}@{host}/{order_id}/001/{file}")) as req:
		with open(f"{path_output}{file}", "wb") as ff:
			shutil.copyfileobj(req, ff)