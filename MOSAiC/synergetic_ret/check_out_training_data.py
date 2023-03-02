import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import pdb
import glob
import matplotlib as mpl
mpl.rcParams.update({"font.family": "monospace"})

import os
import sys
sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")

from my_classes import era5
from met_tools import *


"""
	This script visualizes the atmos part of the training data with scatterplots,
	histograms, profiles, ... all for each year.
	- import training data for a year
	- modify quantities if required
	- visualize
"""


# paths:
path_data = {'era5': "/net/blanc/awalbroe/Data/synergetic_ret/training_data_00/merged/"}
path_plots = "/net/blanc/awalbroe/Plots/synergetic_ret/training_data_atmos/"

# additional settings:
set_dict = {'save_figures': True,
			'iwv_hist': False,
			'lwp_hist': False,
			'temp_profs': False,
			'q_profs': False,
			'temp_profs_bl': False,
			'q_profs_bl': False,
			'q_inv_hist': True,
			'map_plot': False}

path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

# import data:
files = sorted(glob.glob(path_data['era5'] + "ERA5_syn_ret_*.nc"))
era5_atmos = era5(files)
era5_atmos.time_npdt = era5_atmos.time.astype("datetime64[s]")


# visualize:
fs = 16
fs_small = fs - 2
fs_dwarf = fs_small - 2

years = np.unique(era5_atmos.time_npdt.astype('datetime64[Y]').astype(int) + 1970)
for year in years:

	# IWV histogram:
	if set_dict['iwv_hist']:
		# compute weights for histogram:
		data_plot = xr.DataArray(era5_atmos.iwv, dims=['time', 'x', 'y'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}").values.flatten()
		data_plot = data_plot[np.where(~np.isnan(data_plot))[0]]
		n_data = float(len(data_plot))
		weights_data = np.ones((int(n_data),)) / n_data

		f1 = plt.figure(figsize=(16,7))
		a1 = plt.axes()

		x_lim = [0, 35]		# kg m-2

		# plotting:
		le_hist = a1.hist(data_plot, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 0.5),
							weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))
		# add auxiliary info:
		a1.text(0.98, 0.98, f"Min = {np.min(data_plot):.1f}\nMax = {np.max(data_plot):.1f}\nMean = {data_plot.mean():.1f}\n" +
				f"Median = {np.median(data_plot):.1f}\nN = {len(data_plot)}", fontsize=fs_dwarf, ha='right', va='top',
				transform=a1.transAxes)

		# set axis limits:
		a1.set_xlim(x_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_ylabel("Freq. occurrence", fontsize=fs)
		a1.set_title(f"ERA5 IWV, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_iwv_histogram_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['lwp_hist']:
		# compute weights for histogram:
		data_plot = xr.DataArray(era5_atmos.lwp, dims=['time', 'x', 'y'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}").values.flatten()
		data_plot = data_plot[np.where(~np.isnan(data_plot))[0]]*1000.0
		n_data = float(len(data_plot))
		weights_data = np.ones((int(n_data),)) / n_data

		f1 = plt.figure(figsize=(16,7))
		a1 = plt.axes()

		x_lim = [0, 1100]		# g m-2

		# plotting:
		le_hist = a1.hist(data_plot, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 25),
							weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))
		# add auxiliary info: 
		a1.text(0.98, 0.98, f"Min = {np.min(data_plot):.1f}\nMax = {np.max(data_plot):.1f}\nMean = {data_plot.mean():.1f}\n" +
				f"Median = {np.median(data_plot):.1f}\nN = {len(data_plot)}", fontsize=fs_dwarf, ha='right', va='top',
				transform=a1.transAxes)

		# set axis limits:
		a1.set_xlim(x_lim)
		# a1.set_yscale("log")

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
		a1.set_ylabel("Freq. occurrence", fontsize=fs)
		a1.set_title(f"ERA5 LWP, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_lwp_histogram_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['temp_profs']:
		# reduce to required year:
		data_plot = xr.DataArray(era5_atmos.temp, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")
		data_plot_hgt = xr.DataArray(era5_atmos.height, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")

		f1 = plt.figure(figsize=(9,11))
		a1 = plt.axes()

		x_lim = [190, 290]		# K
		y_lim = [0, 13000]		# m

		# plotting:
		for k in range(len(data_plot.x)):
			for l in range(len(data_plot.y)):
				a1.plot(data_plot.values[0,k,l,:], data_plot_hgt.values[0,k,l,:], linewidth=0.5, alpha=0.2)

		# set axis limits:
		a1.set_xlim(x_lim)
		a1.set_ylim(y_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("Temperature (K)", fontsize=fs)
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_title(f"ERA5 Temperature, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_temp_profs_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['q_profs']:
		# reduce to required year:
		data_plot = xr.DataArray(era5_atmos.q*1000.0, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")
		data_plot_hgt = xr.DataArray(era5_atmos.height, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")

		f1 = plt.figure(figsize=(9,11))
		a1 = plt.axes()

		x_lim = [0, 8]		# g kg-1
		y_lim = [0, 13000]	# m

		# plotting:
		for k in range(len(data_plot.x)):
			for l in range(len(data_plot.y)):
				a1.plot(data_plot.values[0,k,l,:], data_plot_hgt.values[0,k,l,:], linewidth=0.5, alpha=0.2)

		# set axis limits:
		a1.set_xlim(x_lim)
		a1.set_ylim(y_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("$q$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_title(f"ERA5 Specific Humidity $q$, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_q_profs_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['temp_profs_bl']:
		# reduce to required year:
		data_plot = xr.DataArray(era5_atmos.temp, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")
		data_plot_hgt = xr.DataArray(era5_atmos.height, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")

		f1 = plt.figure(figsize=(12,7))
		a1 = plt.axes()

		x_lim = [230, 290]		# K
		y_lim = [0, 2000]			# m

		# plotting:
		for k in range(len(data_plot.x)):
			for l in range(len(data_plot.y)):
				a1.plot(data_plot.values[0,k,l,:], data_plot_hgt.values[0,k,l,:], linewidth=0.5, alpha=0.2)

		# set axis limits:
		a1.set_xlim(x_lim)
		a1.set_ylim(y_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("Temperature (K)", fontsize=fs)
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_title(f"ERA5 Temperature boundary layer, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_temp_profs_bl_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['q_profs_bl']:
		# reduce to required year:
		data_plot = xr.DataArray(era5_atmos.q*1000.0, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")
		data_plot_hgt = xr.DataArray(era5_atmos.height, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")

		f1 = plt.figure(figsize=(12,7))
		a1 = plt.axes()

		x_lim = [0, 8]			# g kg-1
		y_lim = [0, 2000]		# m

		# plotting:
		for k in range(len(data_plot.x)):
			for l in range(len(data_plot.y)):
				a1.plot(data_plot.values[0,k,l,:], data_plot_hgt.values[0,k,l,:], linewidth=0.5, alpha=0.2)

		# set axis limits:
		a1.set_xlim(x_lim)
		a1.set_ylim(y_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("$q$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
		a1.set_ylabel("Height (m)", fontsize=fs)
		a1.set_title(f"ERA5 Specific Humidity $q$ boundary layer, {year:04}", fontsize=fs)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_q_profs_bl_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['q_inv_hist']:

		data_plot = xr.DataArray(era5_atmos.q*1000.0, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")
		data_plot_hgt = xr.DataArray(era5_atmos.height, dims=['time', 'x', 'y', 'z'], 
								coords={'time': era5_atmos.time_npdt}).sel(time=f"{year:04}")

		# Detect humidity inversions:
		q_inv, z_inv, inv_strength, inv_height = detect_hum_inversions(data_plot.values[0,:,:,:], data_plot_hgt.values[0,:,:,:], 
																		return_inv_strength_height=True)



		# plot it:
		fs = 14
		fs_small = fs - 2
		fs_dwarf = fs_small -2


		data_plot1 = [z_inv['bot'].flatten()[np.where(~np.isnan(z_inv['bot'].flatten()))[0]],
						z_inv['top'].flatten()[np.where(~np.isnan(z_inv['top'].flatten()))[0]]]
		n_data1 = [float(len(data_plot1[0])), float(len(data_plot1[1]))]
		weights_data1 = [np.ones((int(n_data1[0]),)) / n_data1[0],
							np.ones((int(n_data1[1]),)) / n_data1[1]]


		f1 = plt.figure(figsize=(16,8))
		a1 = plt.subplot2grid((1,3), (0,0))
		a2 = plt.subplot2grid((1,3), (0,1))
		a3 = plt.subplot2grid((1,3), (0,2))

		hgt_lim = [0, 8000]

		# ploting:
		le_hist_1 = a1.hist(data_plot1, bins=np.arange(hgt_lim[0], hgt_lim[1]+0.00001, 100.0),
									weights=weights_data1, color=["#0000a6a6", "#a60000a6"], ec=(0,0,0), stacked=True,
									orientation='horizontal', label=['Inversion bottom', 'Inversion top'])

		# legends and colorbars:
		lh, ll = a1.get_legend_handles_labels()
		a1.legend(handles=lh, labels=ll, fontsize=fs_dwarf, loc='upper right', markerscale=1.5)

		# set axis limits:
		a1.set_ylim(hgt_lim)

		# set ticks and tick labels and parameters:
		a1.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a1.minorticks_on()
		a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a1.set_xlabel("Freq. occurrence", fontsize=fs_small)
		a1.set_ylabel("Height (m)", fontsize=fs_small)
		a1.set_title(f"Inversion height", fontsize=fs_small)



		# inversion strength:
		data_plot2 = inv_strength.flatten()[np.where(~np.isnan(inv_strength.flatten()))[0]]
		n_data2 = float(len(data_plot2))
		weights_data2 = np.ones((int(n_data2),)) / n_data2

		x_lim = [0.0, 2.5]

		# plotting:
		le_hist = a2.hist(data_plot2, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 0.1),
							weights=weights_data2, color=(0.8,0.8,0.8), ec=(0,0,0))
		# add auxiliary info:
		a2.text(0.98, 0.98, f"Min = {np.min(data_plot2):.1f}\nMax = {np.max(data_plot2):.1f}\nMean = {data_plot2.mean():.1f}\n" +
				f"Median = {np.median(data_plot2):.1f}\nN = {len(data_plot2)}", fontsize=fs_dwarf, ha='right', va='top',
				transform=a2.transAxes)

		# set axis limits:
		a2.set_xlim(x_lim)

		# set ticks and tick labels and parameters:
		a2.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a2.minorticks_on()
		a2.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a2.set_xlabel("$q_{\mathrm{top}} - q_{\mathrm{bot}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs_small)
		a2.set_ylabel("Freq. occurrence", fontsize=fs_small)
		a2.set_title(f"Humidity inversion strength", fontsize=fs_small)


		# inversion height:
		data_plot3 = inv_height.flatten()[np.where(~np.isnan(inv_height.flatten()))[0]]
		n_data3 = float(len(data_plot3))
		weights_data3 = np.ones((int(n_data3),)) / n_data3

		x_lim = [0.0, 4000.0]

		# plotting:
		le_hist = a3.hist(data_plot3, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 100.0),
							weights=weights_data3, color=(0.8,0.8,0.8), ec=(0,0,0))
		# add auxiliary info:
		a3.text(0.98, 0.98, f"Min = {np.min(data_plot3):.1f}\nMax = {np.max(data_plot3):.1f}\nMean = {data_plot3.mean():.1f}\n" +
				f"Median = {np.median(data_plot3):.1f}\nN = {len(data_plot3)}", fontsize=fs_dwarf, ha='right', va='top',
				transform=a3.transAxes)

		# set axis limits:
		a3.set_xlim(x_lim)

		# set ticks and tick labels and parameters:
		a3.tick_params(axis='both', labelsize=fs_small)

		# grid:
		a3.minorticks_on()
		a3.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

		# set labels:
		a3.set_xlabel("$z_{\mathrm{top}} - z_{\mathrm{bot}}$ (m)", fontsize=fs_small)
		a3.set_ylabel("Freq. occurrence", fontsize=fs_small)
		a3.set_title(f"Humidity inversion depth", fontsize=fs_small)


		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_q_inv_histogram_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()
			pdb.set_trace()


	if set_dict['map_plot']:

		# reduce datato current year:
		data_plot = xr.DataArray(era5_atmos.sfc_slf, dims=['time', 'x', 'y'], 
								coords={'time': era5_atmos.time_npdt,
										'x': era5_atmos.lon[0,:,0],
										'y': era5_atmos.lat[0,0,:]}).sel(time=f"{year:04}")


		import cartopy
		import cartopy.crs as ccrs
		import cartopy.io.img_tiles as cimgt

		marker_size = 9.0

		# map_settings:
		lon_centre = 0.0
		lat_centre = 75.0
		lon_lat_extent = [-60.0, 60.0, 60.0, 90.0]		# (zoomed in)
		sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


		# some extra info for the plot:
		station_coords = {'Kiruna': [20.223, 67.855],
							'Longyearbyen': [15.632, 78.222]}


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

		# plot data:
		levels_0 = np.arange(0.0, 1.00001, 0.01)
		n_levels = len(levels_0)
		cmap = mpl.cm.get_cmap('Blues', n_levels)
		cmap = cmap(range(n_levels))
		cmap[0,:] = np.array([1.,0.,0.,0.5])
		cmap = mpl.colors.ListedColormap(cmap)
		contour_0 = a1.contourf(data_plot.x.values, data_plot.y.values, data_plot.values[0,:,:].T, 
								cmap=cmap, levels=levels_0, extend='both',
								transform=ccrs.PlateCarree())


		# place markers and labels:
		a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
				marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
				transform=ccrs.PlateCarree(), zorder=10000.0)


		PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
		text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
													x=marker_size*0.75, y=marker_size*0.75)
		a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "LYR",
				ha='left', va='bottom',
				color=(1,0,0), fontsize=fs_dwarf, transform=text_transform, 
				bbox={'facecolor': (211.0/255.0,211.0/255.0,211.0/255.0), 'edgecolor': (0,0,0), 'boxstyle': 'square'},
				zorder=10000.0)


		# colorbar(s) and legends:
		cb_var = f1.colorbar(mappable=contour_0, ax=a1, extend='max', orientation='horizontal', 
								fraction=0.06, pad=0.04, shrink=0.8)
		# # cb_var.set_label(label="$IWV$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_small)
		# # cb_var.set_label(label="$z$ (m)", fontsize=fs_small)
		cb_var.set_label(label="$sfc\_slf$ ()", fontsize=fs_small)
		# # cb_var.set_label(label="$sfc\_sif$ ()", fontsize=fs_small)
		cb_var.ax.tick_params(labelsize=fs_dwarf)

		if set_dict['save_figures']:
			plotname = f"ERA5_syn_ret_training_map_plot_sfc_slf_{year:04}"
			f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
		else:
			plt.show()

		plt.close()
		plt.clf()