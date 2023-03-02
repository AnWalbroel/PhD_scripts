import urllib.request
import urllib
from urllib.error import HTTPError
import json
import glob
import pdb

"""
	Requires a checksum file that will be created when ordering a MODIS data set.
"""

path_checksum = "/net/blanc/awalbroe/Data/METRS_SS21/MODIS_IWV/"
path_modis_data = path_checksum			# path where to save the MODIS data

checksum_file = glob.glob(path_checksum + "checksums_501593436")
fp = open(checksum_file[0], 'r')

# json.dumps reads the "fp" unformatted. json.loads reads it formatted afterwards. splitlines splits lines
# To split the lines columns, we use "split('\n')".
checksum = fp.read().split('\n')

# Now we need to split the columns of the checksum file:
fname = list()
headersize = 5			# skip header and only run until the second last line (because last line is empty)
for i in range(headersize, len(checksum)-1):
	temp = checksum[i].split()
	fname.append(temp[2])

data_product = fname[0][:8]		# name of the data product: e.g. 'MOD05_L2'
i = 0
for k, x in enumerate(fname): # x contains the string of the list element fo fname.
	urllib.request.urlretrieve('https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/%s/%s/%s/%s'%(
								data_product, x[10:14], x[14:17], x), path_modis_data + x)
	print(x, ': ', k)
