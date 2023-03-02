import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import glob
import gc
import geopy
import geopy.distance
import os

import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_datetime
from met_tools import *
import pdb

"""
	Script to visualize ERA5 data of mean sea level pressure (MSLP) to analyze an anomaly over
	Greenland. Following plots will be generated: PLOT_IDX
	0) MSLP for time steps between a start and end date (12h spacing)
	1) Median MSLP for certain period (limits set by start and end date)
	2) Like 2) but for 75th percentile.
	3) Like 3) but for 90th percentile.
	4) 2022 MSLP - 90th percentile for all time steps of 1)
	- import data
	- crop data to correct time and space
	- visualize
"""

# Paths:
path_era5 = "/mnt/f/heavy_data/ERA5_data/single_level/"
path_plots = "/mnt/f/Studium_NIM/work/Plots/HALO_AC3/ERA5_maps_mslp_anomaly/"


# some other settings: Choose date range and location boundaries:
set_dict = {'save_figures': False,		# if True, saved to .png, if False, just show plot
			'date_start': "2022-03-21",
			'date_end': "2022-03-31",
			'temp_spacing': 3,			# temporal spacing of the data in hours
			'PLOT_IDX': 4,
			'quantile': {	'1': 0.5,	# quantile (0.5 == median) for the climatology
							'2': 0.75,
							'3': 0.90,
							'4': 0.90}
			}





# find and import era 5 data:
file_era5 = path_era5 + "ERA5_single_level_MSLP_march_april_1979-2022.nc"
ERA5_DS = xr.open_dataset(file_era5)


# filter time (and space) of ERA5 data:
if set_dict['PLOT_IDX'] == 0:
	ERA5_DS = ERA5_DS.sel(time=slice(set_dict['date_start'], set_dict['date_end']))

elif set_dict['PLOT_IDX'] in [1,2,3,4]:
	# for np.arange, we need an extended date_end:
	set_dict['date_end_plus'] = (dt.datetime.strptime(set_dict['date_end'], "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")

	# to compute climatological mean: first reduce to desired period within a year
	date_range_array = numpydatetime64_to_datetime(np.arange(set_dict['date_start'], set_dict['date_end_plus'], 
						np.timedelta64(set_dict['temp_spacing'], "h"), dtype='datetime64[s]'))
	date_range_array_str = np.array([dra.strftime("%m-%d %H") for dra in date_range_array])
	ERA5_DS['date_str'] = xr.DataArray(ERA5_DS.time.dt.strftime("%m-%d %H").astype("str"), dims=['time'])

	# limit data array to certain periods and areas (also: sea only!):
	is_in_dr = np.array([date_str in date_range_array_str for date_str in ERA5_DS.date_str.values])
	ERA5_clim_DS = ERA5_DS.isel(time=is_in_dr)

	# compute climatology with a certain quantile:
	ERA5_clim_DS['msl_clim'] = ERA5_clim_DS.msl.quantile(set_dict['quantile'][str(set_dict['PLOT_IDX'])], dim='time')

	# limit non-climatology dataset to 2022:
	if set_dict['PLOT_IDX'] == 4:
		ERA5_DS = ERA5_DS.sel(time=slice(set_dict['date_start'], set_dict['date_end']))


# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# map_settings:
lon_centre = 0.0
lat_centre = 75.0
lon_lat_extent = [-40.0, 40.0, 65.0, 90.0]
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


# some extra info for the plot:
station_coords = {'Kiruna': [20.223, 67.855],
					'Longyearbyen': [15.632, 78.222]}


if set_dict['PLOT_IDX'] == 0:
	for k, sel_time in enumerate(ERA5_DS.time):

		if sel_time.dt.hour in [0,12]:
			f1 = plt.figure(figsize=(10,7.5))
			a1 = plt.axes(projection=sel_projection)
			a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

			# add some land marks:
			a1.coastlines(resolution="50m")
			a1.add_feature(cartopy.feature.BORDERS)
			a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
			a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
			a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


			# plot sea level pressure:
			levels_1 = np.arange(920.0, 1050.1, 1.0)
			var_plot = ERA5_DS['msl'][k,:,:].values*0.01	# convert to hPa
			contour_0 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
									levels=levels_1, colors='black', linewidths=1.0, linestyles='solid',
									transform=ccrs.PlateCarree())
			a1.clabel(contour_0, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


			# place markers and labels:
			a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
					marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
					transform=ccrs.PlateCarree(), zorder=10000.0)
			a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
					marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
					transform=ccrs.PlateCarree(), zorder=10000.0)


			PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
			text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
														x=marker_size*0.75, y=marker_size*0.75)
			a1.text(station_coords['Kiruna'][0], station_coords['Kiruna'][1], "Kiruna", 
					ha='left', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform, 
					bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)
			a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "Longyearbyen",
					ha='left', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform, 
					bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)

			# some description:
			a1.text(0.02, 0.98, f"MSLP (hPa), {sel_time.dt.strftime('%Y-%m-%d %H UTC').astype('str').values}", 
					ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
					bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0, transform=a1.transAxes)


			# colorbar(s) and legends:

			if set_dict['save_figures']:
				path_plots += f"PLOT_IDX_{set_dict['PLOT_IDX']}/"
				plot_file = path_plots + f"HALO-AC3_ERA5_CAO_MSLP_{sel_time.dt.strftime('%Y-%m-%d_%HZ').astype('str').values}.png"

				# check if folder exists:
				path_plots_dir = os.path.dirname(path_plots)
				if not os.path.exists(path_plots_dir):
					os.makedirs(path_plots_dir)

				f1.savefig(plot_file, dpi=300, bbox_inches='tight')
				print("Saved " + plot_file)
			else:
				plt.show()

			plt.close()
			gc.collect()


if set_dict['PLOT_IDX'] in [1,2,3]:

	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

	# add some land marks:
	a1.coastlines(resolution="50m")
	a1.add_feature(cartopy.feature.BORDERS)
	a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
	a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
	a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


	# plot data:
	levels_1 = np.arange(920.0, 1050.1, 1.0)
	var_plot = ERA5_clim_DS['msl_clim'].values*0.01		# convert to hPa
	contour_0 = a1.contour(ERA5_clim_DS.msl_clim.longitude.values, ERA5_clim_DS.msl_clim.latitude.values, var_plot, 
							levels=levels_1, colors='black', linewidths=1.0, linestyles='solid',
							transform=ccrs.PlateCarree())
	a1.clabel(contour_0, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


	# place markers and labels:
	a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
			marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
			transform=ccrs.PlateCarree(), zorder=10000.0)
	a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
			marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
			transform=ccrs.PlateCarree(), zorder=10000.0)


	PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
	text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
												x=marker_size*0.75, y=marker_size*0.75)
	a1.text(station_coords['Kiruna'][0], station_coords['Kiruna'][1], "Kiruna", 
			ha='left', va='bottom',
			color=(0,0,0), fontsize=fs_small, transform=text_transform, 
			bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0)
	a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "Longyearbyen",
			ha='left', va='bottom',
			color=(0,0,0), fontsize=fs_small, transform=text_transform, 
			bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0)

	# some description:
	a1.text(0.02, 0.98, f"MSLP (hPa) climatology 1979-2022, {int(set_dict['quantile'][str(set_dict['PLOT_IDX'])]*100)}th percentile", 
			ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
			bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0, transform=a1.transAxes)


	# colorbar(s) and legends:

	if set_dict['save_figures']:
		path_plots += f"PLOT_IDX_{set_dict['PLOT_IDX']}/"
		plot_file = path_plots + f"HALO-AC3_ERA5_CAO_MSLP_1979-2022_percentile_{int(set_dict['quantile'][str(set_dict['PLOT_IDX'])]*100)}.png"

		# check if folder exists:
		path_plots_dir = os.path.dirname(path_plots)
		if not os.path.exists(path_plots_dir):
			os.makedirs(path_plots_dir)

		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)
	else:
		plt.show()

	plt.close()
	gc.collect()


if set_dict['PLOT_IDX'] == 4:
	path_plots += f"PLOT_IDX_{set_dict['PLOT_IDX']}/"
	for k, sel_time in enumerate(ERA5_DS.time):

		if sel_time.dt.hour in [0,12]:

			# compute difference to climatology:
			ERA5_DS['msl_anomaly'] = (ERA5_DS.msl[k,:,:] - ERA5_clim_DS.msl_clim)*0.01		# converted to hPa


			f1 = plt.figure(figsize=(10,7.5))
			a1 = plt.axes(projection=sel_projection)
			a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

			# add some land marks:
			a1.coastlines(resolution="50m")
			a1.add_feature(cartopy.feature.BORDERS)
			a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
			a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
			a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


			# plot climatological data:
			levels_1 = np.arange(920.0, 1050.1, 1.0)
			var_plot = ERA5_clim_DS['msl_clim'].values*0.01		# convert to hPa
			contour_0 = a1.contour(ERA5_clim_DS.msl_clim.longitude.values, ERA5_clim_DS.msl_clim.latitude.values, var_plot, 
									levels=levels_1, colors='black', linewidths=1.0, linestyles='solid',
									transform=ccrs.PlateCarree())
			a1.clabel(contour_0, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

			# set label for legend: only one line needed
			contour_0_leg_handle,_ = contour_0.legend_elements()
			contour_0_leg_label = f"MSLP (hPa) climatology 1979-2022, {int(set_dict['quantile'][str(set_dict['PLOT_IDX'])]*100)}th percentile"
			# contour_0.collections[0].set_label(f"MSLP (hPa) climatology 1979-2022, {int(set_dict['quantile'][str(set_dict['PLOT_IDX'])]*100)}th percentile")


			# plot anomaly:
			levels_2 = np.arange(-40.0, 40.001, 2.0)
			n_levels = len(levels_2)
			cmap = mpl.cm.get_cmap('seismic', n_levels)
			norm = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=levels_2[0], vmax=levels_2[-1])
			contourf_anomaly = a1.contourf(ERA5_DS['msl_anomaly'].longitude.values, ERA5_DS['msl_anomaly'].latitude.values,
											ERA5_DS['msl_anomaly'].values, cmap=cmap, norm=norm, levels=levels_2, extend='both',
											transform=ccrs.PlateCarree())
			# contour_05 = a1.contour(gph_a.longitude.values, gph_a.latitude.values, gph_a.values,
									# levels=[-16.0], colors='black', linewidths=0.8, linestyles='dashed',
									# transform=ccrs.PlateCarree())
			# a1.clabel(contour_05, levels=[-16.0], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


			# place markers and labels:
			a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
					marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
					transform=ccrs.PlateCarree(), zorder=10000.0)
			a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
					marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
					transform=ccrs.PlateCarree(), zorder=10000.0)


			PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
			text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
														x=marker_size*0.75, y=marker_size*0.75)
			a1.text(station_coords['Kiruna'][0], station_coords['Kiruna'][1], "Kiruna", 
					ha='left', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform, 
					bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)
			a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "Longyearbyen",
					ha='left', va='bottom',
					color=(0,0,0), fontsize=fs_small, transform=text_transform, 
					bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0)

			# some description:
			a1.text(0.02, 0.98, (f"MSLP anomaly (hPa) to 1979-2022 {int(set_dict['quantile'][str(set_dict['PLOT_IDX'])]*100)}th percentile," + 
					"\n" + f"{sel_time.dt.strftime('%Y-%m-%d %H UTC').astype('str').values}"), 
					ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
					bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0, transform=a1.transAxes)


			# colorbar(s) and legends:
			# lh, ll = a1.get_legend_handles_labels()
			lol = a1.legend(handles=[contour_0_leg_handle[0]], labels=[contour_0_leg_label], loc='lower left', fontsize=fs_small,
						framealpha=0.8, facecolor=(1.0, 1.0, 1.0, 0.8), edgecolor=(0,0,0))
			lol.set(zorder=10000.0)

			cb_var = f1.colorbar(mappable=contourf_anomaly, ax=a1, extend='both', orientation='horizontal', 
									fraction=0.06, pad=0.04, shrink=0.8)
			cb_var.set_label(label="MSLP anomaly (hPa)", fontsize=fs_small)
			cb_var.ax.tick_params(labelsize=fs_dwarf)



			if set_dict['save_figures']:
				plot_file = path_plots + f"HALO-AC3_ERA5_CAO_MSLP_anomaly_to_1979-2022_{sel_time.dt.strftime('%Y-%m-%d_%HZ').astype('str').values}.png"

				# check if folder exists:
				path_plots_dir = os.path.dirname(path_plots)
				if not os.path.exists(path_plots_dir):
					os.makedirs(path_plots_dir)

				f1.savefig(plot_file, dpi=300, bbox_inches='tight')
				print("Saved " + plot_file)
			else:
				plt.show()

			plt.close()
			gc.collect()