import pyhdf
from pyhdf.SD import *
import numpy as np
import xarray

import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import glob
import gc
import os
import pdb

path_data = "/mnt/d/walbr/Downloads/"

SIC_file = path_data + "asi-AMSR2-n3125-20220401-v5.4.hdf"
lat_lon_file = path_data + "LongitudeLatitudeGrid-n3125-Arctic3125.hdf"


# Load HDF files:
SIC_hdf = pyhdf.SD.SD(SIC_file)
lat_lon_hdf = pyhdf.SD.SD(lat_lon_file)


# inqurie what datasets they contain:
SIC_ds_dict = SIC_hdf.datasets()
lat_lon_ds_dict = lat_lon_hdf.datasets()



# get data as arrays:
SIC_data_dict = dict()
SIC_data_attrs = dict()
lat_lon_data_dict = dict()
lat_lon_data_attrs = dict()
for key in SIC_ds_dict.keys():
	SIC_data_dict[key] = SIC_hdf.select(key).get()
	SIC_data_attrs[key] = SIC_hdf.select(key).attributes()


for key in lat_lon_ds_dict.keys():
	lat_lon_data_dict[key] = lat_lon_hdf.select(key).get()
	lat_lon_data_attrs[key] = lat_lon_hdf.select(key).attributes()


# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15
dt_fmt = mdates.DateFormatter("%d %b")

c_N = (0.2,0.64,0.95)	# colours
c_C = (0.14,0.57,0.27)
c_S = (0.68,0.17,0.14)

# map_settings:
lon_centre = 0.0
lat_centre = 75.0
lon_lat_extent = [-40.0, 40.0, 65.0, 90.0]
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


f1 = plt.figure(figsize=(10,7.5))
a1 = plt.axes(projection=sel_projection)
a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

# add some land marks:
a1.coastlines(resolution="50m")
a1.add_feature(cartopy.feature.BORDERS)
a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


# plot SIC:
levels = np.array([15.0])
cmap = mpl.cm.get_cmap('Blues', len(levels))
var_plot = SIC_data_dict['ASI Ice Concentration']
# pdb.set_trace()
contourf_plot = a1.contour(lat_lon_data_dict['Longitudes'], lat_lon_data_dict['Latitudes'], var_plot, 
									levels=levels, transform=ccrs.PlateCarree())


# # colorbar(s) and legends:
# cb_var = f1.colorbar(mappable=contourf_plot, ax=a1, orientation='horizontal', 
						# fraction=0.06, pad=0.04, shrink=0.8)
# cb_var.set_label(label="SIC (%)", fontsize=fs_small)
# cb_var.ax.tick_params(labelsize=fs_dwarf)

plt.show()

plt.close()
gc.collect()