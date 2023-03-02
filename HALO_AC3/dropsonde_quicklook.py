import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

import pdb
import xarray as xr
import numpy as np
import glob
import gc

import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import sys
sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")


"""
	Quicklooks for HALO-AC3 dropsondes.
"""


# paths:
path_data = {'dropsondes': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/dropsondes/HALO-AC3_HALO_Dropsondes_20220408_RF15/Level_1/"}
path_plots = "/net/blanc/awalbroe/Plots/HALO_AC3_quicklooks/dropsondes/"
save_figures = True


# load data:
files = sorted(glob.glob(path_data['dropsondes'] + "*.nc"))
max_wspd = list()
max_wspd_time = list()
max_wspd_lat = list()
max_wspd_lon = list()
for k, file in enumerate(files):
	DS_DS = xr.open_dataset(file)

	# find max wind speed at 0 - 500 m height:
	hgt_idx = np.where(DS_DS.alt.values <= 500.0)[0]
	if len(hgt_idx) > 0:
		DS_DS = DS_DS.isel(time=hgt_idx)

		max_wspd_temp = np.array([DS_DS.wspd.max().values])[0]
		max_wspd.append(max_wspd_temp)
		max_wspd_time.append(DS_DS.wspd.idxmax().values)

		max_wspd_lat_temp = np.array([DS_DS.wspd.idxmax().lat.values])[0]
		max_wspd_lon_temp = np.array([DS_DS.wspd.idxmax().lon.values])[0]
		if not np.isnan(max_wspd_lat_temp):
			max_wspd_lat.append(max_wspd_lat_temp)
		else:
			max_wspd_lat.append(DS_DS.reference_lat.values[0])
		if not np.isnan(max_wspd_lon_temp):
			max_wspd_lon.append(max_wspd_lon_temp)
		else:
			max_wspd_lon.append(DS_DS.reference_lon.values[0])

	DS_DS.close()


# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# map_settings:
lon_centre = 5.0
lat_centre = 75.0
lon_lat_extent = [-30.0, 30.0, 70.0, 85.0]		# (zoomed in)
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


# some extra info for the plot:
station_coords = {'Kiruna': [20.223, 67.855],
					'Longyearbyen': [15.632, 78.222]}


f1 = plt.figure(figsize=(10,7.5))
a1 = plt.axes(projection=sel_projection)
a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

# add some land marks:
a1.coastlines(resolution="50m", zorder=9999.0)
a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


# plot HALO flight track for the event:
bahamas_file = "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/HALO-AC3_HALO_BAHAMAS_20220408_RF15/QL_HALO-AC3_HALO_BAHAMAS_20220408_RF15_v1.nc"
BAHAMAS_DS = xr.open_dataset(bahamas_file)
a1.plot(BAHAMAS_DS.IRS_LON.values, BAHAMAS_DS.IRS_LAT.values, color=(0,0,0),
		linewidth=1.2, transform=ccrs.PlateCarree())


# plot dropsonde wind speed:
norm = mpl.colors.Normalize(vmin=0.0, vmax=20.0)
cmap = mpl.cm.get_cmap('nipy_spectral')
wspd_plot = a1.scatter(max_wspd_lon, max_wspd_lat, c=max_wspd, s=81, cmap=cmap, norm=norm,
						edgecolors='k', transform=ccrs.PlateCarree())


# place markers and labels:
a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
		marker='^', markersize=marker_size, markeredgecolor=(0,0,0),
		transform=ccrs.PlateCarree(), zorder=10000.0)

PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
											x=marker_size*0.75, y=marker_size*0.75)
a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "Longyearbyen",
		ha='left', va='bottom',
		color=(0,0,0), fontsize=fs_small, transform=text_transform, 
		bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
		zorder=10000.0)


# colorbar(s) and legends:

cb_var = f1.colorbar(mappable=wspd_plot, ax=a1, extend='max', orientation='horizontal', 
						fraction=0.06, pad=0.04, shrink=0.8)
cb_var.set_label(label="Max. wind speed ($\mathrm{m}\,\mathrm{s}^{-1}$)", fontsize=fs_small)
cb_var.ax.tick_params(labelsize=fs_dwarf)

if save_figures:
	plot_file = path_plots + f"HALO-AC3_HALO_dropsondes_Polar_Low_max_wspd_0-500m_height.png"
	f1.savefig(plot_file, dpi=300, bbox_inches='tight')
	print("Saved " + plot_file)
else:
	plt.show()

plt.close()
plt.clf()
gc.collect()