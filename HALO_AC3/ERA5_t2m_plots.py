import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams.update({'font.family': "monospace"})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import glob
import gc
import geopy
import geopy.distance

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_datetime
from met_tools import *
import pdb


"""
	Script to visualize 2 m temperature climatology as time series. Temperature will be
	spatially averaged (i.e., all latitudes and longitudes north of 80 N).
	- import data
	- crop and average data
	- visualize
"""


# Paths:
path_data = {'t2m': "/mnt/d/heavy_data/ERA5_data/single_level/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/HALO_AC3/ERA5_t2m/"

# settings dictionary:
set_dict = {'save_figures': True,
			'weighted_mean': True,
			'time_series': True}


# find and import era 5 data:
file_era5 = path_data['t2m'] + "ERA5_single_level_T2m_march_1979-2022.nc"
ERA5_DS = xr.open_dataset(file_era5)


# crop to certain latitude range:
lat_range = [80.0, 90.0]
ERA5_DS = ERA5_DS.sel(latitude=slice(lat_range[1], lat_range[0]))

if set_dict['weighted_mean']:

	# convert to float64 for increased precision:
	ERA5_DS['t2m'] = ERA5_DS.t2m.astype(np.float64)

	# weighted mean with weights = cos of latitudes:
	weights = np.cos(np.deg2rad(ERA5_DS.latitude.values))
	# weights = np.ones_like(ERA5_DS.latitude.values)		# uniform weights for testing

	# regional avg: flatten weights and values; extend weights to full lat x lon grid, then flatten:
	weights = np.reshape(np.repeat(weights, len(ERA5_DS.longitude.values)), ERA5_DS.t2m.shape[1:]).flatten()

	# compute weighted regional averages (yields the same as ERA5_DS.t2m.mean(['latitude', 'longitude']) if weights == np.ones(...)
	t2m_mean_cr = np.zeros(ERA5_DS.t2m.time.shape)
	for tt in range(len(ERA5_DS.t2m.time)):
		t2m_mean_cr[tt] = np.nansum((ERA5_DS.t2m.values[tt,:,:].flatten()*weights)) / np.sum(weights)

	# put it back into the dataset:
	ERA5_DS['t2m_avg'] = xr.DataArray(t2m_mean_cr - 273.15, dims=['time'])

else:
	ERA5_DS['t2m_avg'] = ERA5_DS.t2m.mean(['longitude', 'latitude']) - 273.15



# Visualize:
fs = 20
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15
dt_fmt = mdates.DateFormatter("%d")
text_colour = (1,1,1)
face_colour = (0.05,0.05,0.05)


if set_dict['time_series']:
	# time series where each year is displayed, mean over 1979-2022 and 2022 itself are stressed:
	f1 = plt.figure(figsize=(14,7), facecolor=face_colour)
	a1 = plt.axes(facecolor=face_colour)
	a1.spines['left'].set_color(text_colour)
	a1.spines['bottom'].set_color(text_colour)
	a1.spines['right'].set_color(face_colour)
	a1.spines['top'].set_color(face_colour)


	# plot each year:
	years = np.arange(1979, 2023)
	time_line = ERA5_DS.t2m_avg.sel(time=f"{years[0]:04}").time
	for year in years:
		data_plot = ERA5_DS.t2m_avg.sel(time=f"{year:04}")
		a1.plot(time_line.values, data_plot.values, linewidth=0.75, color=(1,1,1,0.33))

	# plot average:
	time_line_daily = np.arange(np.datetime64(f"{years[0]}-03-01"), np.datetime64(f"{years[0]}-04-01")).astype("datetime64[ns]")
	data_yearly_groups = ERA5_DS.t2m_avg.groupby('time.day').mean('time')
	a1.plot(time_line_daily, data_yearly_groups, linewidth=2.0, color=text_colour, label=f'mean 1979-2022')

	# plot 2022:
	data_plot = ERA5_DS.t2m_avg.sel(time="2022")
	a1.plot(time_line.values, data_plot.values, color=(1.0,0,0), linewidth=2.0, label='2022')


	# legend:
	lh, ll = a1.get_legend_handles_labels()
	a1.legend(handles=lh, labels=ll, loc='lower right', fontsize=fs_small, facecolor=face_colour, labelcolor=text_colour)


	# set axis limits:
	a1.set_xlim(left=time_line.values[0], right=time_line.values[1])

	# set axis ticks, ticklabels and tick parameters:
	a1.set_xticks(time_line_daily[::2])
	a1.xaxis.set_major_formatter(dt_fmt)
	a1.tick_params(axis='both', labelsize=fs_dwarf, color=text_colour, labelcolor=text_colour)


	# grid:
	a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# labels:
	a1.set_ylabel("T2m (deg C)", fontsize=fs, color=text_colour)
	a1.set_xlabel("Days of March", fontsize=fs, color=text_colour)

	if set_dict['save_figures']:
		plotname = f"ERA5_T2m_climatology_north80N_1979-2022"
		f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
	else:
		plt.show()

		plt.close()