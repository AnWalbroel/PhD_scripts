import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'monospace'})

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopy
import geopy.distance

import pdb

path = "/net/blanc/awalbroe/Data/HALO_AC3/sst_slice/"
file = path + "20220311120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc.nc4"
DS = xr.open_dataset(file)



# visualise:
fs = 20
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15
c_orbit = (0.71,0.596,0.388)


# map_settings:
lon_centre = 0.0
lat_centre = 80.0
lon_lat_extent = [-30.0, 30.0, 70.0, 90.0]
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)

station_coords = {'Ny-Alesund': [11.928611, 78.924444]}


f1 = plt.figure(figsize=(11,9))
a1 = plt.axes(projection=sel_projection)
a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

# add some land marks:
a1.coastlines(resolution="50m", zorder=2)
a1.add_feature(cartopy.feature.BORDERS, zorder=2)
a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


# plot SST:
levels_0 = np.arange(273.15, 288.15, 0.5)
n_levels = len(levels_0)
cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
cmap = cmap(range(n_levels))
cmap[0,:] = np.array([0.,0.,0.,0.])
cmap = mpl.colors.ListedColormap(cmap)
sst_plot = DS.analysed_sst[0,:,:]
contour_0 = a1.contourf(sst_plot.lon.values, sst_plot.lat.values, sst_plot.values, 
						cmap=cmap, levels=levels_0, extend='both',
						transform=ccrs.PlateCarree())

# place markers and labels:
a1.plot(station_coords['Ny-Alesund'][0], station_coords['Ny-Alesund'][1], color=(1,0,0),
		marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
		transform=ccrs.PlateCarree(), zorder=10000.0)


PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
											x=marker_size*1.75, y=marker_size*1.75)
a1.text(station_coords['Ny-Alesund'][0], station_coords['Ny-Alesund'][1], 'Ny-Alesund',
		ha='left', va='bottom',
		color=(0,0,0), fontsize=fs_small, transform=text_transform, 
		bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
		zorder=10000.0)

# some description:
a1.text(0.02, 0.98, f"{str(DS.time.values[0].astype('datetime64[s]')).replace('T', ' ') + ' UTC'}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
		bbox={'facecolor': (0.8, 0.8, 0.8, 0.9), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
		zorder=10000.0, transform=a1.transAxes)


# colorbar(s) and legends:
cb_ticks = np.arange(levels_0[0], levels_0[-1]+0.00001, 2.5)
cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
						fraction=0.06, pad=0.04, shrink=0.8, ticks=cb_ticks)
cb_var.ax.set_xticklabels([f"{cb_tick - 273.15}" for cb_tick in cb_ticks])
cb_var.set_label(label="SST (deg C)", fontsize=fs_small)
cb_var.ax.tick_params(labelsize=fs_dwarf)

plt.show()

plt.close()
pdb.set_trace()