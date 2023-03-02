import xarray as xr
import pdb
import os
import numpy as np
import glob
import sys
import datetime as dt

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_epochtime


"""
	Concatenate the surface property files

	NOAA JPSS Microwave Integrated Retrieval System (MIRS) Advanced 
	Technology Microwave Sounder (ATMS) Precipitation and Surface Products from NDE. 
	[indicate subset used]. NOAA National Centers for Environmental Information. 
	doi:10.7289/V51V5C1X

	and remove unnecessary variables.
	- find files
	- import files
	- filter variables
	- concatenate and form new dataset
	- export new dataset
"""


# Paths:
path_data_old = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/surface_props/old/"	# input
path_data = "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/surface_props/"	# output


# find and import files: Loop through files
files = sorted(glob.glob(path_data_old + "NPR-MIRS-IMG_v11r4*.nc"))
for ii, file in enumerate(files):
	DS = xr.open_dataset(file, decode_times=False)

	# Build time array correctly:
	time_npdt = np.asarray([np.datetime64(f"{int(year):04}-{int(month):02}-{int(day):02}T{int(hour):02}:{int(min):02}:{int(sec):02}") 
							for year, month, day, hour, min, sec in zip(DS.ScanTime_year.values, 
							DS.ScanTime_month.values, DS.ScanTime_dom.values, DS.ScanTime_hour.values,
							DS.ScanTime_minute.values, DS.ScanTime_second.values)])
	time = numpydatetime64_to_epochtime(time_npdt)
	DS['time_npdt'] = xr.DataArray(time_npdt, dims=["Scanline"])
	DS['time'] = xr.DataArray(time, dims=["Scanline"], attrs={'units': "seconds since 1970-01-01 00:00:00",
								'standard_name': "time"})

	# reduce dataset and concatenate:
	if ii == 0:
		DS_smol = DS[['Freq', 'Polo', 'Latitude', 'Longitude', 'Sfc_type', 'Qc', 'SIce', 'Emis', 'time', 'time_npdt']]

	else:
		DS = DS[['Freq', 'Polo', 'Latitude', 'Longitude', 'Sfc_type', 'Qc', 'SIce', 'Emis', 'time', 'time_npdt']]
		DS_smol = xr.concat([DS_smol, DS], dim="Scanline", data_vars="all", coords="all",
							compat="equals") 	# i.e., Polo or Freq must be the same over all time steps

	DS.close()


# export the optimized data set: Set attributes, encoding and save it:
DS_smol.attrs = {'missing_value': -999,
				'notretrievedproduct_value': -888,
				'noretrieval_value': -99,
				'Conventions': "CF-1.5",
				'title': "MIRS IMG",
				'summary': "MIRS imaging products including surface emissivity, TPW, CLW, RWP, IWP, LST.",
				'institution': "DOC/NOAA/NESDIS/NDE > NPOESS Data Exploitation, NESDIS, NOAA, U.S. Department of Commerce",
                'naming_authority': "gov.noaa.nesdis.nde",
				'edited_by': "Andreas Walbroel (a.walbroel@uni-koeln.de)",
				"date_modified": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

# DS_smol['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
# DS_smol['time'].encoding['dtype'] = "double"

# for file name, extract first and last time step:
time_start = dt.datetime.utcfromtimestamp(DS_smol.time.values[0]).strftime("%Y%m%dT%H%M%SZ")
time_end = dt.datetime.utcfromtimestamp(DS_smol.time.values[-1]).strftime("%Y%m%dT%H%M%SZ")

filename_out = path_data + f"MIRS_surf_props_{time_start}-{time_end}.nc"
DS_smol.to_netcdf(filename_out, mode="w", format="NETCDF4")

print(f"Surface props have been saved to {filename_out}")
print("Done....")

