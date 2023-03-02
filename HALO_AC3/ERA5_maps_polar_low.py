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

import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_datetime
from met_tools import *
import pdb

"""
	Script to visualize ERA5 data around the Polar Low occurrence during HALO-AC3.
	Mean sea level pressure, wind gusts, sea surface temperature, ... will be plotted.
"""

# Paths:
path_era5 = "/mnt/f/heavy_data/ERA5_data/single_level/"
path_era5_ml = "/mnt/f/heavy_data/ERA5_data/multi_level/"
path_plots = "/mnt/f/Studium_NIM/work/Plots/HALO_AC3/ERA5_polar_low/"


# some other settings:
solo_maps = False		# solo maps of several quantities
mslp_500hPaGP_plot = False
SST_500hPaT_plot = False
mslp_500hPaGP_SST_500hPaT_plot = True
wind_plot = False
theta_e_850_plot = False
vert_wind_plot = False
temperature_plot = False
rel_vorticity_adv_plot = False
abs_vorticity_adv_plot = False
mtpr_plot = False
SST_T2m_plot = False
RH_layer_plot = False
LR_plot = False
GPH_anomaly_plot = False
wind_dir_clim_plot = False
wind_dir_diff_cao_wai_plot = False
save_figures = False # if True, saved to .png, if False, just show plot


# choose date range and location boundaries:
date_start = "2022-04-08"
date_end = "2022-04-08"
if wind_dir_clim_plot or wind_dir_diff_cao_wai_plot:
	date_start = "2022-03-07"
	# date_start = "2022-03-21"
	# date_end = "2022-04-13"
	date_end = "2022-04-21"


# find and import era 5 data:
# files = sorted(glob.glob(path_era5 + "*.nc"))
file_era5 = path_era5 + "ERA5_single_level_20220406-20220409_HALO-AC3.nc"
ERA5_DS = xr.open_dataset(file_era5)

# same for pressure level era 5 data:
file_era5 = path_era5_ml + "ERA5_pressure_levels_20220406-20220409_HALO-AC3.nc"
ERA5_ML_DS = xr.open_dataset(file_era5)

# additional data about liquid and solid water:
file_era5_add = path_era5_ml + "ERA5_pressure_levels_total_water_20220406-20220409_HALO-AC3.nc"
ERA5_ML_TW_DS = xr.open_dataset(file_era5_add)


# filter time (and space?) of ERA5 data: eventually manually select important longitudes to further 
# reduce memory usage:
ERA5_DS = ERA5_DS.sel(time=slice(date_start, date_end))
ERA5_ML_DS = ERA5_ML_DS.sel(time=slice(date_start, date_end))
ERA5_ML_TW_DS = ERA5_ML_TW_DS.sel(time=slice(date_start, date_end))


# Variables to plot and their respective expected limits and their cmaps:
# var_list = ['i10fg', 'msl', 'sst', 'sp', 'tcc', 'tcwv', 'tp']
var_list = ['i10fg']
var_limit_dir = {'i10fg': np.arange(0.0, 30.01, 1.0),
					'msl': np.arange(980.0, 1030.1, 1.0),
					'sst': np.arange(268.0, 283.01, 0.5),
					'sp': np.arange(980.0, 1030.1, 1.0),
					'tcc': np.arange(0.0, 100.1, 12.5),
					'lcc': np.arange(0.0, 100.1, 12.5),
					'mcc': np.arange(0.0, 100.1, 12.5),
					'hcc': np.arange(0.0, 100.1, 12.5),
					'tcwv': np.arange(0.0, 20.0, 0.5),
					'tp': np.arange(0.0, 3.001, 0.1)}
var_extend_dir = {'i10fg': "max",
					'msl': "both",
					'sst': "both",
					'sp': "both",
					'tcc': "neither",
					'lcc': "neither",
					'mcc': "neither",
					'hcc': "neither",
					'tcwv': "max",
					'tp': "max"}

var_cmap_dir = dict()
for var in var_list:
	if var in ['i10fg', 'mtpr', 'msl', 'sp', 'sst', 'tp', 'p72.162', 'p70.162']:
		var_cmap_dir[var] = mpl.cm.get_cmap('nipy_spectral', len(var_limit_dir[var]))

	elif var in ['tcwv']:
		var_cmap_dir[var] = mpl.cm.get_cmap('gist_earth_r', len(var_limit_dir[var]))

	elif var in ['tcc', 'lcc', 'mcc', 'hcc']:
		var_cmap_dir[var] = mpl.cm.get_cmap('binary', len(var_limit_dir[var]))

	elif var in ['ptype']:
		var_cmap_dir[var] = mpl.cm.get_cmap('Set2')

var_cb_label_dir = {'i10fg': "10$\,$m wind gust ($\mathrm{m}\,\mathrm{s}^{-1}$)",
					'msl': "MSLP (hPa)",
					'mtpr': "Mean total precip. rate ($\mathrm{kg}\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}$)",
					'ptype': "Precipitation type",
					'sst': "Sea surface temperature (K)",
					'sp': "Surface pressure (hPa)",
					'tcc': "Total cloud cover (\%)",
					'lcc': "Low cloud cover (\%)",
					'mcc': "Medium cloud cover (\%)",
					'hcc': "High cloud cover (\%)",
					'tcwv': "IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)",
					'tp': "Total precip. (mm)",
					'p72.162': "Vert. integ. northw. w. v. flux ($\mathrm{kg}\,\mathrm{m}^{-1}\,\mathrm{s}^{-1}$)",
					'p70.162': "Vert. integ. northw. heat flux ($\mathrm{W}\,\mathrm{m}^{-1}$)"}


# Visualize:

fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# map_settings:
lon_centre = 5.0
lat_centre = 75.0
lon_lat_extent = [-40.0, 40.0, 65.0, 90.0]
if mslp_500hPaGP_SST_500hPaT_plot: lon_lat_extent = [-30.0, 30.0, 70.0, 85.0]		# (zoomed in)
sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


# some extra info for the plot:
station_coords = {'Kiruna': [20.223, 67.855],
					'Longyearbyen': [15.632, 78.222]}

# # # # # # # # # # # # # reduce plottables to ocean only:
# # # # # # # # # # # # # ERA5_DS['tcwv'] = xr.where(ERA5_DS['lsm'] > 0.15, np.nan, ERA5_DS['tcwv'])
# # # # # # # # # # # # # # # also visualize sea ice fraction: 'siconc'?????
if solo_maps:
	ERA5_DS = xr.open_dataset(path_era5 + "ERA5_single_level_hourly_10m_gust_20220407-20220408_HALO-AC3.nc")
	ERA5_DS = ERA5_DS.sel(time=slice(date_start, date_end))

	for pvar in var_list:

		for time_idx in range(len(ERA5_DS.time.values)):
			time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

			f1 = plt.figure(figsize=(10,7.5))
			a1 = plt.axes(projection=sel_projection)
			a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
			a1.add_image(cimgt.Stamen('terrain-background'), 4)

			# add some land marks:
			a1.coastlines(resolution="50m", zorder=9999.0)
			a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
			a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

			# plot var:
			levels = var_limit_dir[pvar]
			cmap = var_cmap_dir[pvar]
			var_plot = ERA5_DS[pvar][time_idx,:,:].values
			if pvar in ['msl', 'sp']:
				var_plot *= 0.01
			elif pvar in ['tcc', 'lcc', 'mcc', 'hcc']:
				var_plot *= 100.0
			elif pvar in ['tp']:
				var_plot *= 1000.0
				cmap_new = cmap(range(len(levels)))
				cmap_new[0,:] = np.array([0., 0., 0., 0.])
				cmap = mpl.colors.ListedColormap(cmap_new)

			contourf_plot = a1.contourf(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
									cmap=cmap, levels=levels, extend=var_extend_dir[pvar],
									transform=ccrs.PlateCarree())

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
			a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
					bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
					zorder=10000.0, transform=a1.transAxes)


			# colorbar(s):
			cb_var = f1.colorbar(mappable=contourf_plot, ax=a1, extend='max', orientation='horizontal', 
									fraction=0.06, pad=0.04, shrink=0.8)
			cb_var.set_label(label=var_cb_label_dir[pvar], fontsize=fs_small)
			cb_var.ax.tick_params(labelsize=fs_dwarf)


			if save_figures:
				plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_{pvar}_{time_str}.png"
				f1.savefig(plot_file, dpi=300)
				print("Saved " + plot_file)
			else:
				plt.show()

			plt.close()
			gc.collect()


if mtpr_plot:

	# additional data about liquid and solid water:
	file_era5_mtpr = path_era5 + "ERA5_single_level_20220201-20220430_HALO-AC3.nc"
	ERA5_MTPR_DS = xr.open_dataset(file_era5_mtpr)

	# filter time (and space?) of ERA5 data: eventually manually select important longitudes to further 
	# reduce memory usage:
	ERA5_MTPR_DS = ERA5_MTPR_DS.sel(time=slice(date_start, date_end))

	for time_idx in range(len(ERA5_MTPR_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_MTPR_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot mtpr:
		levels_1 = np.arange(0.0, 3.01, 0.1)
		n_levels = len(levels_1)
		cmap = mpl.cm.get_cmap("seismic", n_levels)
		var_plot = ERA5_MTPR_DS['mtpr'][time_idx,:,:]*3600.0		# converting from mm s-1 to mm h-1
		contour_1 = a1.contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot.values[1,:,:], 
								cmap=cmap, levels=levels_1, extend='both',
								transform=ccrs.PlateCarree())


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		cb_var = f1.colorbar(mappable=contour_1, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="MTPR ($\mathrm{mm}\,\mathrm{h}^{-1}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)


		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mtpr_{time_str}.png"
			f1.savefig(plot_file, dpi=300)
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if mslp_500hPaGP_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 1.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.5, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot 500 hPa GP or GP height:
		levels_2 = np.arange(476.0, 600.01, 2.0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.5, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.90), fontsize=fs_small,
						framealpha=0.75)
		lol.set(zorder=10000.0)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_500hPaGP_{time_str}.png"
			f1.savefig(plot_file, dpi=300)
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if SST_500hPaT_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot stuff:
		levels_1 = np.arange(38.0, 45.1, 0.5)
		n_levels = len(levels_1)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
		cmap = cmap(range(n_levels))
		cmap[0,:] = np.array([0.,0.,0.,0.])
		cmap[-1,:] = np.array([0.4,0.2,0.2,0.75])
		cmap = mpl.colors.ListedColormap(cmap)
		var_plot = (ERA5_DS['sst'][time_idx,:,:] - ERA5_ML_DS['t'][time_idx, hgt_idx,:,:])
		contour_1 = a1.contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot.values, 
								cmap=cmap, levels=levels_1, extend='both',
								transform=ccrs.PlateCarree())


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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		cb_var = f1.colorbar(mappable=contour_1, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="SST - T$_{\mathrm{500\,hPa}}$ (K)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_SST-T500hPa_{time_str}.png"
			f1.savefig(plot_file, dpi=300)
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if mslp_500hPaGP_SST_500hPaT_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0, linewidth=0.5)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot SST-500hPaT:
		levels_0 = np.arange(38.0, 45.1, 0.5)
		n_levels = len(levels_0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
		cmap = cmap(range(n_levels))
		cmap[:,3] = np.full_like(cmap[:,3], 0.65)
		cmap[0,:] = np.array([0.,0.,0.,0.])
		cmap[-2,:] = np.array([0.9,0.4,0.4,0.65])
		cmap[-1,:] = np.array([0.9,0.7,0.7,0.5])
		cmap = mpl.colors.ListedColormap(cmap)
		sst_plot = (ERA5_DS['sst'][time_idx,:,:] - ERA5_ML_DS['t'][time_idx, hgt_idx,:,:])
		contour_0 = a1.contourf(sst_plot.longitude.values, sst_plot.latitude.values, sst_plot.values, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 1.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot 500 hPa GP or GP height:
		levels_2 = np.arange(476.0, 600.01, 2.0)
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# Find coordinates of a 200 km radius circle around a centre point of the Polar Low; then plot it:
		# centre of Polar Low: lowest MSLP:
		time_genesis = 0
		PL_bounds = [[76.0, 80.0], [-12.5, 12.5]]	# rough lat,lon Polar Low boundaries at 2022-04-08 0000Z
		MSLP_PL = ERA5_DS.msl[time_genesis,:,:].sel(longitude=slice(PL_bounds[1][0], PL_bounds[1][1]),
												latitude=slice(PL_bounds[0][1], PL_bounds[0][0]))
		where_min = np.where(MSLP_PL.values == np.min(MSLP_PL.values))
		circ_centre = [MSLP_PL.latitude.values[where_min[0][0]], MSLP_PL.longitude.values[where_min[1][0]]] # lat lon of circle's centre

		# define circle:
		circ_rad = 200.0		# radius in km
		circ_ang = np.arange(360.0)		# angles in deg
		# len_ang = len(circ_ang)
		# circ_lon = np.full((len_ang,), np.nan)	# circle longitudes
		# circ_lat = np.full((len_ang,), np.nan)	# circle latitudes
		# circ_centre_gp = geopy.Point(circ_centre[0], circ_centre[1])
		# dist = geopy.distance.distance(kilometers=circ_rad)
		# for k, ang in enumerate(circ_ang):
			# circ_lat[k] = dist.destination(point=circ_centre_gp, bearing=ang)[0]
			# circ_lon[k] = dist.destination(point=circ_centre_gp, bearing=ang)[1]

		# find ERA5 points within the circle:
		era5_nlon = len(ERA5_ML_DS.longitude)
		era5_nlat = len(ERA5_ML_DS.latitude)
		circle_mask = np.full((era5_nlat, era5_nlon), False)
		for i, lat in enumerate(ERA5_ML_DS.latitude.values):
			for j, lon in enumerate(ERA5_ML_DS.longitude.values):
				if geopy.distance.distance((lat, lon), (circ_centre[0], circ_centre[1])).km <= circ_rad:
					circle_mask[i,j] = True


		# CHECK RADOVAN ET AL. 2019 CONDITIONS: FOR TIME WHEN PL WAS STILL IN DEVELOPMENT: 2022-04-08 00:00Z
		# (SST-T500, SST-T2m, LR, RH(sfc-950,950-850), wind_gust, GPH_anomaly):
		SST_T500_PL = (ERA5_DS['sst'][time_genesis,:,:] - ERA5_ML_DS['t'][time_genesis, hgt_idx,:,:]).values[circle_mask]
		SST_T2m_PL = (ERA5_DS['sst'][time_genesis,:,:] - ERA5_DS['t2m'][time_genesis,:,:]).values[circle_mask]

		# Lapse rate:
		theta = potential_temperature(ERA5_ML_DS.level.values*100.0, ERA5_ML_DS.t.values[time_genesis,:,:,:], 100000.0, height_axis=0)
		ERA5_ML_DS['theta'] = xr.DataArray(theta, dims=['level','latitude','longitude'], 
												coords={'latitude': ERA5_ML_DS.latitude,
														'longitude': ERA5_ML_DS.longitude,
														'level': ERA5_ML_DS.level})

		# compute dtheta/dz:
		theta = ERA5_ML_DS.theta.sel(level=slice(850,1050))
		dtheta = np.diff(theta.values, axis=0)
		dz = np.diff(Z_from_GP(ERA5_ML_DS.z[time_genesis,:,:,:].sel(level=slice(850,1050)).values), axis=0)
		LR = dtheta*1000.0 / dz			# lapse rate in K/km
		LR_mean_PL = np.mean(LR, axis=0)[circle_mask]		# vertical mean
		LR_min_PL = np.min(LR, axis=0)[circle_mask]			# vertical min

		# relative humidity:
		RH_850_950_PL = ERA5_ML_DS.r[time_genesis,:,:,:].sel(level=slice(850,950)).mean('level').values[circle_mask]
		RH_950_sfc_PL = ERA5_ML_DS.r[time_genesis,:,:,:].sel(level=slice(950,1100)).mean('level').values[circle_mask]

		gust_PL = ERA5_DS.i10fg[time_genesis,:,:].values[circle_mask]

		# GPH anomaly:
		# load geopot height climatology:
		file_gp_clim = path_era5_ml + "ERA5_GP_500_850_hPa_april_1979-2022.nc"
		ERA5_GP_DS = xr.open_dataset(file_gp_clim)

		hgt_level = 500.0		# hPa
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - hgt_level))
		hgt_clim_idx = np.argmin(np.abs(ERA5_GP_DS.level.values - hgt_level))

		# compute climatological mean:
		ERA5_GP_DS['GPH'] = Z_from_GP(ERA5_GP_DS.z[:,hgt_clim_idx,:,:])*0.1		# in gpdam
		gph_c = ERA5_GP_DS.GPH.mean('time')
		ERA5_ML_DS['GPH'] = Z_from_GP(ERA5_ML_DS.z[time_genesis,hgt_idx,:,:])*0.1		# in gpdam
		GPH_anomaly_PL = (ERA5_ML_DS.GPH - gph_c).values[circle_mask]


		# 75th percentiles of (SST-T500, SST-T2m, LR, RH(sfc-950,950-850)):
		C1 = np.percentile(SST_T500_PL, q=75)
		C2 = np.percentile(SST_T2m_PL, q=75)
		C3_min = np.percentile(LR_min_PL, q=75)
		C3_mean = np.percentile(LR_mean_PL, q=75)
		C4_i = np.percentile(RH_950_sfc_PL, q=75)
		C4_ii = np.percentile(RH_850_950_PL, q=75)

		# max of wind gust:
		C5 = np.max(gust_PL)

		# mean of GPH anomaly:
		C6 = np.mean(GPH_anomaly_PL)

		# plot HALO flight track for the event:
		bahamas_file = "/mnt/f/heavy_data/HALO_AC3/BAHAMAS/QL_HALO-AC3_HALO_BAHAMAS_20220408_RF15_v1.nc"
		BAHAMAS_DS = xr.open_dataset(bahamas_file)
		a1.plot(BAHAMAS_DS.IRS_LON.values, BAHAMAS_DS.IRS_LAT.values, color=(0,0,0),
				linestyle='dashed', linewidth=1.2,
				label='HALO flight track', transform=ccrs.PlateCarree(), zorder=10000.0)


		# find where max wind speed of Polar Low is located:
		gust_PL = ERA5_DS.i10fg[time_idx+1,:,:].sel(longitude=slice(PL_bounds[1][0], PL_bounds[1][1]), 
									latitude=slice(PL_bounds[0][1], PL_bounds[0][0]))
		where_max = np.where(gust_PL.values == np.max(gust_PL.values))
		where_max_lat, where_max_lon = where_max[0][0], where_max[1][0]
		a1.plot([gust_PL.longitude.values[where_max_lon], gust_PL.longitude.values[where_max_lon]],
				[gust_PL.latitude.values[where_max_lat], gust_PL.latitude.values[where_max_lat]],
				linestyle='none', marker='*', color='k', markersize=marker_size*0.66,
				transform=ccrs.PlateCarree(), zorder=10000.0)

		PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
		text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
													x=-marker_size*0.75, y=marker_size*0.75)
		a1.text(gust_PL.longitude.values[where_max_lon], gust_PL.latitude.values[where_max_lat],
				f"Max. gust {gust_PL[where_max_lat, where_max_lon].values:.1f}" + "$\,\mathrm{m}\,\mathrm{s}^{-1}$",
				ha='right', va='bottom', color=(0,0,0), fontsize=fs_dwarf, transform=text_transform,
				bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle': 'square'},
				zorder=10000.0)


		# place markers and labels:
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

		PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
		text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
													x=marker_size*0.75, y=marker_size*0.75)
		a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "LYR",
				ha='left', va='bottom',
				color=(1,0,0), fontsize=fs_small, transform=text_transform, 
				bbox={'facecolor': (211.0/255.0,211.0/255.0,211.0/255.0), 'edgecolor': (0,0,0), 'boxstyle': 'square'},
				zorder=10000.0)

		# some description:
		a1.text(0.02, 0.98, f"{numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime('%d %B %Y %H:%M UTC')}",
				ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (211.0/255.0,211.0/255.0,211.0/255.0, 0.9), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)

		# text with conditions C1 - C6:
		a1.text(0.80, 0.98, f"C1: {C1:.1f}" + "$\,$K\n" + f"C2: {C2:.1f}" + "$\,$K\n" +
				f"C3: {C3_mean:.1f}" + "$\,\mathrm{K}\,\mathrm{km}^{-1}$\n" + "C4$_{\mathrm{i}}$:" + f" {C4_i:.1f}" + "$\\%$\n" +
				"C4$_{\mathrm{ii}}$:" + f" {C4_ii:.1f}" + "$\\%$\n" + f"C5: {C5:.1f}" + "$\,\mathrm{m}\,\mathrm{s}^{-1}$\n" + 
				f"C6: {int(C6*10.0)}" + "$\,$gpm", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (211.0/255.0,211.0/255.0,211.0/255.0, 0.9), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)

		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.9, facecolor=(211.0/255.0,211.0/255.0,211.0/255.0), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="SST - T$_{\mathrm{500\,hPa}}$ (K)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + "for_paper/" + f"HALO-AC3_ERA5_Polar_Low_mslp_500hPaGP_SST-T500hPa_{time_str}.pdf"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if vert_wind_plot:

	for time_idx in range(len(ERA5_ML_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_ML_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot mslp:
		levels_1 = np.arange(-2.5, 2.51, 0.25)
		n_levels = len(levels_1)
		hgt_lvl = 700.0
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - hgt_lvl))
		cmap = mpl.cm.get_cmap('seismic', n_levels)
		var_plot = ERA5_ML_DS.w.values[time_idx,hgt_idx,:,:]
		contour_1 = a1.contourf(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, var_plot, 
								cmap=cmap, levels=levels_1, extend='both',
								transform=ccrs.PlateCarree())


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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		cb_var = f1.colorbar(mappable=contour_1, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="$\omega_{700\,\mathrm{hPa}}$ ($\mathrm{Pa}\,\mathrm{s}^{-1}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_{int(hgt_lvl)}hPa_omega_{time_str}.png"
			f1.savefig(plot_file, dpi=300)
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if theta_e_850_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)
	ERA5_ML_DS['q_hydtot'] = (ERA5_ML_TW_DS.ciwc + ERA5_ML_TW_DS.clwc +
								ERA5_ML_TW_DS.crwc + ERA5_ML_TW_DS.cswc)
	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot 850 hPa equiv pot T:
		levels_0 = np.arange(-16.0, 16.1, 1.0)
		n_levels = len(levels_0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 850.0))
		theta_e_850 = equiv_pot_temperature(ERA5_ML_DS.t.values[time_idx,hgt_idx,:,:],
															85000.0, 0.01*ERA5_ML_DS.r.values[time_idx,hgt_idx,:,:],
															ERA5_ML_DS.q.values[time_idx,hgt_idx,:,:],
															ERA5_ML_DS.q_hydtot.values[time_idx,hgt_idx,:,:],
															neglect_rtc=False)
		ERA5_ML_DS['theta_e_850'] = xr.DataArray(theta_e_850, dims=['latitude','longitude'], 
												coords={'latitude': ERA5_ML_DS.latitude,
														'longitude': ERA5_ML_DS.longitude})
		theta_e_850 = ERA5_ML_DS['theta_e_850']
		cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
		# cmap = mpl.cm.get_cmap('gist_rainbow_r', n_levels)
		contour_0 = a1.contourf(theta_e_850.longitude.values, theta_e_850.latitude.values, theta_e_850.values - 273.15, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_00 = a1.contour(theta_e_850.longitude.values, theta_e_850.latitude.values, theta_e_850.values - 273.15, 
								levels=levels_0[::2], colors='grey', linewidths=0.5, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_00, levels=levels_0[::2], colors='grey', fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 4.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot 500 hPa GP or GP height:
		levels_2 = np.arange(476.0, 600.01, 2.0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)



		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="$\\theta_{\mathrm{e,850\,hPa}}$ ($^{\circ}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_500hPaGP_850hPatheta_e_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if temperature_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot T:
		levels_0 = np.arange(-46.0, -14.9, 1.0)
		n_levels = len(levels_0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		temperature = ERA5_ML_DS.t[time_idx,hgt_idx,:,:]
		cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
		# cmap = mpl.cm.get_cmap('gist_rainbow_r', n_levels)
		contour_0 = a1.contourf(temperature.longitude.values, temperature.latitude.values, temperature.values - 273.15, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_00 = a1.contour(temperature.longitude.values, temperature.latitude.values, temperature.values - 273.15, 
								levels=levels_0[::2], colors='grey', linewidths=0.5, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_00, levels=levels_0[::2], colors='grey', fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 4.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot 850 hPa GP or GP height:
		levels_2 = np.arange(476.0, 600.01, 2.0)
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - 500.0))
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)



		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="T ($^{\circ}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_500hPaGP_500hPaT_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if rel_vorticity_adv_plot:
	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


		# plot 500 hPa geopotential:
		hgt_level = 500.0
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - hgt_level))
		levels_2 = np.arange(476.0, 600.01, 2.0)
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot rel vorticity advection:
		levels_0 = np.linspace(-1.0e-07,1.0001e-07,32)
		norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_0[0], vmax=levels_0[-1])
		n_levels = len(levels_0)
		rva = relative_vorticity_advection(ERA5_ML_DS.u.values[time_idx, hgt_idx,:,:], 
											ERA5_ML_DS.v.values[time_idx, hgt_idx,:,:],
											ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values)
		ERA5_ML_DS['rva'] = xr.DataArray(rva, dims=['latitude', 'longitude'],
													coords={'latitude': ERA5_ML_DS.latitude,
															'longitude': ERA5_ML_DS.longitude})
		rva = ERA5_ML_DS.rva
		cmap = mpl.cm.get_cmap("seismic", n_levels)
		contour_0 = a1.contourf(rva.longitude.values, rva.latitude.values,
								rva.values, cmap=cmap, norm=norm, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_00 = a1.contour(rva.longitude.values, rva.latitude.values,
								rva.values, levels=levels_0[levels_0 <= 0.0][::2], colors='black',
								linewidths=0.5, linestyles='solid', transform=ccrs.PlateCarree())
		contour_01 = a1.contour(rva.longitude.values, rva.latitude.values,
								rva.values, levels=levels_0[levels_0 > 0.0][::2], colors='black',
								linewidths=0.5, linestyles='dotted', transform=ccrs.PlateCarree())


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="500$\,$hPa Geopot (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="Rel. vorticity advection ($\mathrm{s}^{-2}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_rel_vortic_advec_{str(int(hgt_level))}hPa_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if abs_vorticity_adv_plot:
	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


		# plot 500 hPa geopotential:
		hgt_level = 500.0
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - hgt_level))
		levels_2 = np.arange(476.0, 600.01, 2.0)
		gp_height_plot = Z_from_GP(ERA5_ML_DS['z'].values[time_idx, hgt_idx, :, :])*0.1		# in gpdam
		contour_2 = a1.contour(ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values, gp_height_plot,
								levels=levels_2, colors='black', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot abs vorticity advection:
		levels_0 = np.linspace(-0.5e-07,0.5001e-07,32)
		norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_0[0], vmax=levels_0[-1])
		n_levels = len(levels_0)

		import metpy.calc as mpcalc
		from metpy.units import units
		f = mpcalc.coriolis_parameter(np.deg2rad( np.repeat(np.reshape(ERA5_ML_DS.latitude.values, (len(ERA5_ML_DS.latitude.values),1)), len(ERA5_ML_DS.longitude.values), axis=1) ))
		avor = mpcalc.vorticity(ERA5_ML_DS.u[time_idx, hgt_idx,:,:], ERA5_ML_DS.v[time_idx, hgt_idx,:,:]) + f
		vort_adv = mpcalc.advection(avor, u=ERA5_ML_DS.u[time_idx, hgt_idx,:,:], v=ERA5_ML_DS.v[time_idx, hgt_idx,:,:])
		


		# # # # # # # rva = absolute_vorticity_advection(ERA5_ML_DS.u.values[time_idx, hgt_idx,:,:], 
											# # # # # # # ERA5_ML_DS.v.values[time_idx, hgt_idx,:,:],
											# # # # # # # ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values)
		# # # # # # # ERA5_ML_DS['rva'] = xr.DataArray(rva, dims=['latitude', 'longitude'],
													# # # # # # # coords={'latitude': ERA5_ML_DS.latitude,
															# # # # # # # 'longitude': ERA5_ML_DS.longitude})
		rva = vort_adv
		cmap = mpl.cm.get_cmap("seismic", n_levels)
		contour_0 = a1.contourf(rva.longitude.values, rva.latitude.values,
								rva.values, cmap=cmap, norm=norm, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_00 = a1.contour(rva.longitude.values, rva.latitude.values,
								rva.values, levels=levels_0[levels_0 <= 0.0][::2], colors='black',
								linewidths=0.5, linestyles='solid', transform=ccrs.PlateCarree())
		contour_01 = a1.contour(rva.longitude.values, rva.latitude.values,
								rva.values, levels=levels_0[levels_0 > 0.0][::2], colors='black',
								linewidths=0.5, linestyles='dotted', transform=ccrs.PlateCarree())


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="500$\,$hPa Geopot (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="Abs. vorticity advection ($\mathrm{s}^{-2}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_abs_vortic_advec_{str(int(hgt_level))}hPa_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if wind_plot:
	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 4.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# plot divergence:
		wind_level = 1000.0
		hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - wind_level))
		levels_0 = np.arange(-0.0005,0.0005000001,0.000025)
		norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_0[0], vmax=levels_0[-1])
		n_levels = len(levels_0)

		"""	IS THE EXACT SAME AS MY VERSION
		import metpy.calc as mpcalc
		divergence = mpcalc.divergence(ERA5_ML_DS.u[time_idx, hgt_idx,:,:], ERA5_ML_DS.v[time_idx, hgt_idx,:,:])
		"""

		divergence = compute_divergence(ERA5_ML_DS.u.values[time_idx, hgt_idx,:,:], 
											ERA5_ML_DS.v.values[time_idx, hgt_idx,:,:],
											ERA5_ML_DS.longitude.values, ERA5_ML_DS.latitude.values)
		ERA5_ML_DS['divergence'] = xr.DataArray(divergence, dims=['latitude', 'longitude'],
													coords={'latitude': ERA5_ML_DS.latitude,
															'longitude': ERA5_ML_DS.longitude})
		divergence = ERA5_ML_DS.divergence
		cmap = mpl.cm.get_cmap("seismic", n_levels)
		contour_0 = a1.contourf(divergence.longitude.values, divergence.latitude.values,
								divergence.values, cmap=cmap, norm=norm, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_00 = a1.contour(divergence.longitude.values, divergence.latitude.values,
								divergence.values, levels=levels_0[levels_0 <= 0.0][::2], colors='black',
								linewidths=0.5, linestyles='solid', transform=ccrs.PlateCarree())
		contour_01 = a1.contour(divergence.longitude.values, divergence.latitude.values,
								divergence.values, levels=levels_0[levels_0 > 0.0][::2], colors='black',
								linewidths=0.5, linestyles='dotted', transform=ccrs.PlateCarree())

		# plot wind:
		wind_plot = a1.barbs(ERA5_ML_DS.longitude.values[::10], ERA5_ML_DS.latitude.values[::5],		# IN KNOTS
								ERA5_ML_DS.u.values[time_idx,hgt_idx,::5,::10]*3.6/1.85, ERA5_ML_DS.v.values[time_idx,hgt_idx,::5,::10]*3.6/1.85, 
								barbcolor=(0,0,0), length=4, pivot='middle', flagcolor=(0,0,0),
								transform=ccrs.PlateCarree())


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="Divergence ($\mathrm{s}^{-1}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_{str(int(wind_level))}hPawind_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if SST_T2m_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0, linewidth=0.5)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot SST-T2m:
		levels_0 = np.arange(0.0, 12.1, 0.5)
		n_levels = len(levels_0)
		cmap = mpl.cm.get_cmap('nipy_spectral', n_levels)
		cmap = cmap(range(n_levels))
		cmap[:,3] = np.full_like(cmap[:,3], 0.65)
		cmap[0,:] = np.array([0.,0.,0.,0.])
		cmap[-1,:] = np.array([0.4,0.2,0.2,0.5])
		cmap = mpl.colors.ListedColormap(cmap)
		sst_plot = (ERA5_DS['sst'][time_idx,:,:] - ERA5_DS['t2m'][time_idx,:,:])
		contour_0 = a1.contourf(sst_plot.longitude.values, sst_plot.latitude.values, sst_plot.values, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 4.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="Geop. Height (gpdam)")

		PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
		text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
													x=marker_size*1.75, y=marker_size*1.75)
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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (0.8, 0.8, 0.8, 0.9), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.9, facecolor=(0.8,0.8,0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="SST - T$_{\mathrm{2\,m}}$ (K)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_SST-T2m_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if RH_layer_plot:

	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)
	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot average RH over 1000-950 hPa, and 950-850 hPa:
		levels_0 = np.arange(0.0, 100.1, 2.5)
		n_levels = len(levels_0)
		RH_level_lims = [850, 950]
		ERA5_ML_DS['RH_mean'] = ERA5_ML_DS.r[time_idx,:,:,:].sel(level=slice(RH_level_lims[0],RH_level_lims[1])).mean('level')
		cmap = mpl.cm.get_cmap('terrain_r', n_levels)
		contour_0 = a1.contourf(ERA5_ML_DS['RH_mean'].longitude.values, ERA5_ML_DS['RH_mean'].latitude.values, ERA5_ML_DS['RH_mean'].values, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 1.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=1.5, label="MSLP (hPa)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="RH$_{1000-950}$ ($\%$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_mslp_RH_{RH_level_lims[1]}-{RH_level_lims[0]}hPa_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if LR_plot:

	for time_idx in range(len(ERA5_ML_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_ML_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

		# plot LR:
		theta = potential_temperature(ERA5_ML_DS.level.values*100.0, ERA5_ML_DS.t.values[time_idx,:,:,:], 100000.0, height_axis=0)
		ERA5_ML_DS['theta'] = xr.DataArray(theta, dims=['level','latitude','longitude'], 
												coords={'latitude': ERA5_ML_DS.latitude,
														'longitude': ERA5_ML_DS.longitude,
														'level': ERA5_ML_DS.level})

		# compute dtheta/dz:
		theta = ERA5_ML_DS.theta.sel(level=slice(850,1050))
		dtheta = np.diff(theta.values, axis=0)
		dz = np.diff(Z_from_GP(ERA5_ML_DS.z[time_idx,:,:,:].sel(level=slice(850,1050)).values), axis=0)
		dtheta_dz = dtheta*1000.0 / dz			# in K/km

		levels_1 = np.arange(-10.0, 10.01, 0.5)
		n_levels = len(levels_1)
		norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_1[0], vmax=levels_1[-1])
		cmap = mpl.cm.get_cmap('seismic', n_levels)
		dtheta_dz_plot = np.max(dtheta_dz, axis=0)
		contour_0 = a1.contourf(theta.longitude.values, theta.latitude.values, dtheta_dz_plot, 
								cmap=cmap, levels=levels_1, norm=norm, extend='both',
								transform=ccrs.PlateCarree())

		# plot mslp:
		levels_1 = np.arange(920.0, 1050.1, 1.0)
		var_plot = ERA5_DS['msl'][time_idx,:,:].values*0.01
		contour_1 = a1.contour(ERA5_DS.longitude.values, ERA5_DS.latitude.values, var_plot, 
								levels=levels_1, colors='white', linewidths=1.25, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_1, levels=levels_1[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

		# add contour line for the LR threshold:
		contour_th = a1.contour(theta.longitude.values, theta.latitude.values, dtheta_dz_plot,
					colors='k', levels=[3.0], linestyles='dashed', transform=ccrs.PlateCarree())
		a1.clabel(contour_th, levels=[3.0], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# Find coordinates of a 200 km radius circle around a centre point of the Polar Low; then plot it:
		# centre of Polar Low: lowest MSLP:
		time_genesis = 0
		PL_bounds = [[76.0, 80.0], [-12.5, 12.5]]	# rough lat,lon Polar Low boundaries at 2022-04-08 0000Z
		MSLP_PL = ERA5_DS.msl[time_genesis,:,:].sel(longitude=slice(PL_bounds[1][0], PL_bounds[1][1]),
												latitude=slice(PL_bounds[0][1], PL_bounds[0][0]))
		where_min = np.where(MSLP_PL.values == np.min(MSLP_PL.values))
		circ_centre = [MSLP_PL.latitude.values[where_min[0][0]], MSLP_PL.longitude.values[where_min[1][0]]] # lat lon of circle's centre

		# define and plot circle:
		circ_rad = 200.0		# radius in km
		circ_ang = np.arange(360.0)		# angles in deg
		len_ang = len(circ_ang)
		circ_lon = np.full((len_ang,), np.nan)	# circle longitudes
		circ_lat = np.full((len_ang,), np.nan)	# circle latitudes
		circ_centre_gp = geopy.Point(circ_centre[0], circ_centre[1])
		dist = geopy.distance.distance(kilometers=circ_rad)
		for k, ang in enumerate(circ_ang):
			circ_lat[k] = dist.destination(point=circ_centre_gp, bearing=ang)[0]
			circ_lon[k] = dist.destination(point=circ_centre_gp, bearing=ang)[1]

		a1.plot(circ_lon, circ_lat, linewidth=1.5, color=(0,0,0), transform=ccrs.PlateCarree(),
				zorder=10000.0)


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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.75), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="$\\frac{\mathrm{d}\\theta}{\mathrm{d}z}$ ($\mathrm{K}\,\mathrm{km}^{-1}$)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_LR_sfc-850hPa_max_{time_str}.png"
			f1.savefig(plot_file, dpi=300)
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		gc.collect()


if GPH_anomaly_plot:
	# make sure that time grids are identical:
	assert np.all(ERA5_ML_DS.time.values == ERA5_DS.time.values)

	# load geopot height climatology:
	file_gp_clim = path_era5_ml + "ERA5_GP_500_850_hPa_april_1979-2022.nc"
	ERA5_GP_DS = xr.open_dataset(file_gp_clim)

	hgt_level = 500.0		# hPa
	hgt_idx = np.argmin(np.abs(ERA5_ML_DS.level.values - hgt_level))
	hgt_clim_idx = np.argmin(np.abs(ERA5_GP_DS.level.values - hgt_level))

	# compute climatological mean:
	ERA5_GP_DS['GPH'] = Z_from_GP(ERA5_GP_DS.z[:,hgt_clim_idx,:,:])*0.1		# in gpdam
	gph_c = ERA5_GP_DS.GPH.mean('time')

	for time_idx in range(len(ERA5_DS.time.values)):
		time_str = numpydatetime64_to_datetime(ERA5_DS.time.values[time_idx]).strftime("%Y%m%d_%H%MZ")

		f1 = plt.figure(figsize=(10,7.5))
		a1 = plt.axes(projection=sel_projection)
		a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
		# a1.add_image(cimgt.Stamen('terrain-background'), 4)

		# add some land marks:
		a1.coastlines(resolution="50m", zorder=9999.0)
		a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
		a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


		# plot mean geopot height as contours
		levels_2 = np.arange(476.0, 600.01, 2.0)
		contour_2 = a1.contour(gph_c.longitude.values, gph_c.latitude.values, gph_c,
								levels=levels_2, colors='black', linewidths=1.0, linestyles='solid',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_2, levels=levels_2[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# plot GP height anomalies:
		ERA5_ML_DS['GPH'] = Z_from_GP(ERA5_ML_DS.z[time_idx,hgt_idx,:,:])*0.1		# in gpdam
		gph_a = ERA5_ML_DS.GPH - gph_c
		levels_0 = np.arange(-40.0, 20.01, 2.0)
		n_levels = len(levels_0)
		cmap = mpl.cm.get_cmap('seismic', n_levels)
		norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_0[0], vmax=levels_0[-1])
		# cmap = mpl.cm.get_cmap('gist_rainbow_r', n_levels)
		contour_0 = a1.contourf(gph_a.longitude.values, gph_a.latitude.values, gph_a.values, 
								cmap=cmap, norm=norm, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())
		contour_05 = a1.contour(gph_a.longitude.values, gph_a.latitude.values, gph_a.values,
								levels=[-16.0], colors='black', linewidths=0.8, linestyles='dashed',
								transform=ccrs.PlateCarree())
		a1.clabel(contour_05, levels=[-16.0], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


		# place markers and labels:
		a1.plot(station_coords['Kiruna'][0], station_coords['Kiruna'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)

		# dummy plots for legend:
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.5, label="500$\,$hPa GP height" + "\nclimatology 1979-2022 (gpdam)")
		a1.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.2, 
				linestyle='dotted', label="GP height anomaly threshold (gpdam)")

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
		a1.text(0.02, 0.98, f"{time_str}", ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:
		lh, ll = a1.get_legend_handles_labels()
		lol = a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
						framealpha=0.8, facecolor=(0.8, 0.8, 0.8), edgecolor=(0,0,0))
		lol.set(zorder=10000.0)

		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		cb_var.set_label(label="500$\,$hPa GP height anomaly (gpdam)", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if save_figures:
			plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_GPH_{hgt_level}hPa_anomaly_{time_str}.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		plt.clf()
		gc.collect()


if wind_dir_clim_plot:
	# make sure that time grids are identical:

	height_lvl_wind = 500		# in hPa

	# load geopot height climatology:
	file_uv_clim = path_era5_ml + f"ERA5_U_V_wind_{str(height_lvl_wind)}_hPa_march_april_1979-2022.nc"
	ERA5_UV_DS = xr.open_dataset(file_uv_clim)

	# limit to certain year(s) if desired:
	# ERA5_UV_DS = ERA5_UV_DS.sel(time='2022')


	# compute climatological mean: first reduce to desired period within a year
	date_range_array = numpydatetime64_to_datetime(np.arange(date_start, date_end, dtype='datetime64[D]'))
	date_range_array_str = np.array([dra.strftime("%m-%d") for dra in date_range_array])
	ERA5_UV_DS['month_day_str'] = xr.DataArray(ERA5_UV_DS.time.dt.strftime("%m-%d").astype("str"), dims=['time'])

	# limit data array to certain periods:
	is_in_dr = np.array([month_day_str in date_range_array_str for month_day_str in ERA5_UV_DS.month_day_str.values])
	ERA5_UV_DS = ERA5_UV_DS.isel(time=is_in_dr)

	# compute temporal mean and wind direction:
	ERA5_UV_DS = ERA5_UV_DS.mean('time')
	wspeed, wdir = u_v_to_wspeed_wdir(ERA5_UV_DS.u.values, ERA5_UV_DS.v.values, convention='from')
	ERA5_UV_DS['wspeed'] = xr.DataArray(wspeed, dims=['latitude', 'longitude'])
	ERA5_UV_DS['wdir'] = xr.DataArray(wdir, dims=['latitude', 'longitude'])


	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

	# add some land marks:
	a1.coastlines(resolution="50m", zorder=9999.0)
	a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
	a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
	a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
	a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


	# plot wind direction:
	levels_0 = np.arange(0.0, 360.01, 10.0)
	n_levels = len(levels_0)
	cmap = mpl.cm.get_cmap('hsv', n_levels)
	contour_0 = a1.contourf(ERA5_UV_DS.wdir.longitude.values, ERA5_UV_DS.wdir.latitude.values, ERA5_UV_DS.wdir.values, 
							cmap=cmap, levels=levels_0, 
							transform=ccrs.PlateCarree())

	# add wind barbs:
	a1.barbs(ERA5_UV_DS.wdir.longitude.values[::6], ERA5_UV_DS.wdir.latitude.values[::3], ERA5_UV_DS.u.values[::3,::6], ERA5_UV_DS.v.values[::3,::6],
								length=4.5, pivot='middle', barbcolor=(1,1,1,0.5), rounding=True,
								zorder=9999, transform=ccrs.PlateCarree())


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
	a1.text(0.02, 0.98, f"{date_start} - {(dt.datetime.strptime(date_end, '%Y-%m-%d') + dt.timedelta(days=-1)).strftime('%Y-%m-%d')}", 
			ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
			bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0, transform=a1.transAxes)


	# colorbar(s) and legends:

	cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="Wind direction (deg)", fontsize=fs_small)
	cb_var.ax.tick_params(labelsize=fs_dwarf)

	if save_figures:
		plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_wind_dir_{str(height_lvl_wind)}hPa_climatology_WAI_1979-2022.png"
		# plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_wind_dir_{str(height_lvl_wind)}hPa_climatology_CAO_2022.png"		# for selected year
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)
	else:
		plt.show()

	plt.close()
	gc.collect()


if wind_dir_diff_cao_wai_plot:
	# make sure that time grids are identical:

	# load geopot height climatology:
	file_uv_clim = path_era5_ml + "ERA5_U_V_wind_850_hPa_march_april_1979-2022.nc"
	ERA5_UV_DS = xr.open_dataset(file_uv_clim)


	# for wind direction differences between CAO and WAI period:
	# compute climatological mean: first reduce to desired period within a year
	date_range_array_wai = numpydatetime64_to_datetime(np.arange("2022-03-11", "2022-03-21", dtype='datetime64[D]'))
	date_range_array_cao = numpydatetime64_to_datetime(np.arange("2022-03-21", "2022-04-13", dtype='datetime64[D]'))
	date_range_array_wai_str = np.array([dra.strftime("%m-%d") for dra in date_range_array_wai])
	date_range_array_cao_str = np.array([dra.strftime("%m-%d") for dra in date_range_array_cao])
	ERA5_UV_DS['month_day_str'] = xr.DataArray(ERA5_UV_DS.time.dt.strftime("%m-%d").astype("str"), dims=['time'])

	# limit data array to certain periods:
	is_in_dr_wai = np.array([month_day_str in date_range_array_wai_str for month_day_str in ERA5_UV_DS.month_day_str.values])
	is_in_dr_cao = np.array([month_day_str in date_range_array_cao_str for month_day_str in ERA5_UV_DS.month_day_str.values])
	ERA5_UV_WAI_DS = ERA5_UV_DS.isel(time=is_in_dr_wai)
	ERA5_UV_CAO_DS = ERA5_UV_DS.isel(time=is_in_dr_cao)

	# compute temporal mean and wind direction:
	ERA5_UV_WAI_DS = ERA5_UV_WAI_DS.mean('time')
	ERA5_UV_CAO_DS = ERA5_UV_CAO_DS.mean('time')
	wspeed_wai, wdir_wai = u_v_to_wspeed_wdir(ERA5_UV_WAI_DS.u.values, ERA5_UV_WAI_DS.v.values, convention='from')
	wspeed_cao, wdir_cao = u_v_to_wspeed_wdir(ERA5_UV_CAO_DS.u.values, ERA5_UV_CAO_DS.v.values, convention='from')

	ERA5_UV_WAI_DS['wspeed'] = xr.DataArray(wspeed_wai, dims=['latitude', 'longitude'])
	ERA5_UV_WAI_DS['wdir'] = xr.DataArray(wdir_wai, dims=['latitude', 'longitude'])
	ERA5_UV_CAO_DS['wspeed'] = xr.DataArray(wspeed_cao, dims=['latitude', 'longitude'])
	ERA5_UV_CAO_DS['wdir'] = xr.DataArray(wdir_cao, dims=['latitude', 'longitude'])

	# define wind direction and correct issues for wind direction differences around 0 or 360 deg:
	wdir_diff = ERA5_UV_CAO_DS['wdir'] - ERA5_UV_WAI_DS['wdir']
	wdir_diff = xr.where(wdir_diff > 180.0, x=wdir_diff-360.0, y=wdir_diff)
	wdir_diff = xr.where(wdir_diff < -180.0, x=wdir_diff+360.0, y=wdir_diff)


	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

	# add some land marks:
	a1.coastlines(resolution="50m", zorder=9999.0)
	a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
	a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
	a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
	a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)


	# plot wind direction:
	levels_0 = np.arange(-180.0, 180.01, 5.0)
	n_levels = len(levels_0)
	norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_0[0], vmax=levels_0[-1])
	cmap = mpl.cm.get_cmap('seismic', n_levels)
	contour_0 = a1.contourf(wdir_diff.longitude.values, wdir_diff.latitude.values, wdir_diff.values, 
							cmap=cmap, norm=norm, levels=levels_0, extend='both',
							transform=ccrs.PlateCarree())


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
	a1.text(0.02, 0.98, f"CAO period - WAI period", 
			ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
			bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0, transform=a1.transAxes)


	# colorbar(s) and legends:

	cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='both', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="Wind direction difference (deg)", fontsize=fs_small)
	cb_var.ax.tick_params(labelsize=fs_dwarf)

	if save_figures:
		plot_file = path_plots + f"HALO-AC3_ERA5_Polar_Low_wind_dir_850hPa_climatology_diff_CAO-WAI_1979-2022.png"
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)
	else:
		plt.show()

	plt.close()
	gc.collect()
