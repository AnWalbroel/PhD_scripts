import numpy as np
import xarray as xr
import sys
import pdb
import glob

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
mpl.rcParams.update({"font.family": "monospace"})

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from import_data import import_DSHIP, import_DSHIP_csv


"""
	Test script to check out the DSHIP importer... and maybe to plot
	the DSHIP track.
"""


# paths:
path_data = "/mnt/d/heavy_data/WALSEMA/DSHIP/"
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/dship/"


# settings:
set_dict = {'ship_track': True,
			'save_figures': False}


# find and import file:
file = glob.glob(path_data + "*.csv")[0]
# dship = import_DSHIP(file)		# for .dat data
dship = import_DSHIP_csv(file)	# for .csv data

date_start = "2022-07-15"
date_end = "2022-07-19"	# excluding the end date 
date_idx = np.where((dship['date time'] >= np.datetime64(date_start)) & (dship['date time'] < np.datetime64(date_end)))


# visualize ship track
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# map_settings:
lon_centre = 0.0
lat_centre = 70.0
lon_lat_extent = [-30.0, 30.0, 51.0, 85.0]
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


# some extra info for the plot:
station_coords = {'Scoresby Sund': [-25.370214, 71.269469],
					'Longyearbyen': [15.631667, 78.222222],
					'79 N Glacier': [-25.0, 79.0],
					'Bremerhaven': [8.583333, 53.55]}

if set_dict['ship_track']:
	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
	a1.add_image(cimgt.Stamen("terrain-background"), 6)

	# add some land marks:
	a1.coastlines(resolution='50m', zorder=999.0)
	a1.add_feature(cartopy.feature.BORDERS, zorder=1000.0, color=(0.7,0.7,0.7,0.5))
	a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0, color=(153./255.,179./255.,204./255.))
	# a1.add_feature(cartopy.feature.LAND, zorder=999.0)#, color=(0.9, 0.85, 0.85))
	gl = a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=999.0)
	gl.xlabel_style['size'] = fs_small
	gl.ylabel_style['size'] = fs_small

	# plot ship track:
	a1.plot(dship['SYS.STR.PosLon'], dship['SYS.STR.PosLat'], linewidth=1.2, color=(0.75,0,0),
			transform=ccrs.PlateCarree())

	# plot a dashed line between end of DSHIP data and Bremerhaven port:
	final_data_point = np.min(np.array([np.where(~np.isnan(dship['SYS.STR.PosLon']))[0][-1], 
										np.where(~np.isnan(dship['SYS.STR.PosLat']))[0][-1]]))
	a1.plot([dship['SYS.STR.PosLon'][final_data_point], station_coords['Bremerhaven'][0]], 
			[dship['SYS.STR.PosLat'][final_data_point], station_coords['Bremerhaven'][1]],
			linewidth=0.8, linestyle='dashed', color=(0,0,0), transform=ccrs.PlateCarree())


	# plot markers and labels:
	for stat in station_coords.keys():
		a1.plot(station_coords[stat][0], station_coords[stat][1], color=(0.75,0,0), marker='.',
				markersize=marker_size, markeredgecolor=(0,0,0), transform=ccrs.PlateCarree(),
				zorder=10000.0)

		if stat in ['79 N Glacier', 'Scoresby Sund']:
			PC_mpl_tranformer = ccrs.PlateCarree()._as_mpl_transform(a1)
			text_transform = mpl.transforms.offset_copy(PC_mpl_tranformer, units='dots',
														x=-0.75*marker_size, y=marker_size*0.75)
			a1.text(station_coords[stat][0], station_coords[stat][1], stat, ha='right', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform,
					bbox={'facecolor': (1,1,1,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)

		else:
			PC_mpl_tranformer = ccrs.PlateCarree()._as_mpl_transform(a1)
			text_transform = mpl.transforms.offset_copy(PC_mpl_tranformer, units='dots',
														x=0.75*marker_size, y=marker_size*0.75)
			a1.text(station_coords[stat][0], station_coords[stat][1], stat, ha='left', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform,
					bbox={'facecolor': (1,1,1,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)


	if set_dict['save_figures']:
		f1.savefig(path_plots + "PS131_DSHIP_ship_track.png", dpi=400, bbox_inches='tight')
	else:
		plt.show()
