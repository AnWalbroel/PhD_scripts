import xarray as xr
import numpy as np
import datetime as dt
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

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_datetime
import pdb


def regional_averages(DS, lat_bnds, lon_bnds):

	"""
	Computes regional averages over the boxes of the HALO-AC3 synoptic overview paper.
	Latitude and longitude boundaries must be provided to split the dataset DS into its
	northern, central and southern region components.

	Parameters:
	-----------
	DS : xarray dataset
		ERA5 data with latitude and longitude coordinates. Will be split into the three
		regions.
	lat_bnds : dict of arrays
		Keys 'N', 'C', 'S' indicate the southern (northern) latitude boundaries as first 
		(last) value of the array.
	lon_bnds : dict of array
		Keys 'N', 'C', 'S' indicate the wester (eastern) longitude boundaries as first 
		(last) value of the array.
	"""

	# northern:
	DS_N = DS.where(((DS.latitude >= lat_bnds['N'][0]) & (DS.latitude < lat_bnds['N'][1]) & 
								(DS.longitude >= lon_bnds['N'][0]) & (DS.longitude <= lon_bnds['N'][1])) | 
								((DS.latitude >= lat_bnds['N_ex'][0]) & (DS.latitude < lat_bnds['N_ex'][1]) & 
								(DS.longitude >= lon_bnds['N_ex'][0]) & (DS.longitude <= lon_bnds['N_ex'][1])))

	# central:
	DS_C = DS.where(((DS.latitude >= lat_bnds['C'][0]) & (DS.latitude < lat_bnds['C'][1]) & 
								(DS.longitude >= lon_bnds['C'][0]) & (DS.longitude <= lon_bnds['C'][1])))

	# southern:
	DS_S = DS.where(((DS.latitude >= lat_bnds['S'][0]) & (DS.latitude < lat_bnds['S'][1]) & 
								(DS.longitude >= lon_bnds['S'][0]) & (DS.longitude <= lon_bnds['S'][1])))


	# compute means over regions:
	DS_N = DS_N.mean(['latitude', 'longitude'])
	DS_C = DS_C.mean(['latitude', 'longitude'])
	DS_S = DS_S.mean(['latitude', 'longitude'])

	return DS_N, DS_C, DS_S


###################################################################################################
###################################################################################################


"""
	Script to visualize single level ERA5 data of i.e., mean sea level pressure 
	(MSLP), 2 m temperature, IWV, ... 
	- import data
	- crop data to correct time and space
	- visualize
"""

# Paths:
path_era5 = {'single_level': "/mnt/d/heavy_data/ERA5_data/single_level/",
			'multi_level': "/mnt/d/heavy_data/ERA5_data/multi_level/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/HALO_AC3/ERA5_single_level_plots/"


# some other settings: Choose date range and location boundaries:
set_dict = {'save_figures': False,		# if True, saved to .png, if False, just show plot
			'date_start': "2022-03-11",
			'date_end': "2022-03-20",
			'temp_spacing': 6,			# temporal spacing of the data in hours (inquire your data)
			'MSLP_map': False,			# map plot of mean sea level pressure
			'met_time_series': False,	# time series plot of basic meteorolog. variables (can be combined with load_climatology
			'anomaly_maps': True,		# creates anomaly maps; needs load_climatology=True
			'region_avg': False,		# regional averages will be computed (True for met_time_series)
			'load_climatology': True,	# if True, climatology of 1979-2022 won't be cut
			'daily': True,				# if True, only daily values at 12 UTC are considered
			}


# find and import era 5 data:
if set_dict['anomaly_maps']:
	file_era5 = path_era5['single_level'] + "ERA5_single_levels_T2m_MSLP_IWV_march_april_24h_1979-2022.nc"
	file_era5_ml = path_era5['multi_level'] + "ERA5_pressure_levels_T850_hPa_march_april_1979-2022.nc"
	ERA5_ML_DS = xr.open_dataset(file_era5_ml)
else:
	file_era5 = path_era5['single_level'] + "ERA5_single_level_T2m_MSLP_IWV_march_april_6h_1979-2022.nc"
ERA5_DS = xr.open_dataset(file_era5)


# eventually, reduce to daily values at 12 UTC:
if set_dict['daily']: 
	ERA5_DS = ERA5_DS.isel(time=(ERA5_DS.time.dt.hour.isin(12)))	# reduces to values at 12 UTC only
	set_dict['temp_spacing'] = 24		# adjusted to the "daily" setting


# adjust settings based on input given above:
if set_dict['met_time_series']:
	set_dict['region_avg'] = True
	set_dict['date_start'] = "2022-03-07"
	set_dict['date_end'] = "2022-04-12"
elif set_dict['MSLP_map']:
	set_dict['region_avg'] = False

# for climatologies, we need np.arange, and for this, an extended date_end:
set_dict['date_end_plus'] = (dt.datetime.strptime(set_dict['date_end'], "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")


# load climatology if desired:
if set_dict['load_climatology']:

	# to compute climatological mean: first reduce to desired period within a year
	if set_dict['daily']:	# then, date_range_array needs 12 UTC only
		 date_range_array = numpydatetime64_to_datetime(np.arange(set_dict['date_start'], set_dict['date_end_plus'], 
							np.timedelta64(set_dict['temp_spacing'], "h"), dtype='datetime64[s]') + np.timedelta64(12,'h'))
	else:
		date_range_array = numpydatetime64_to_datetime(np.arange(set_dict['date_start'], set_dict['date_end_plus'], 
							np.timedelta64(set_dict['temp_spacing'], "h"), dtype='datetime64[s]'))
	date_range_array_str = np.array([dra.strftime("%m-%d %H") for dra in date_range_array])
	ERA5_DS['date_str'] = xr.DataArray(ERA5_DS.time.dt.strftime("%m-%d %H").astype("str"), dims=['time'])

	# limit data array to certain periods and areas (also: sea only!):
	is_in_dr = np.array([date_str in date_range_array_str for date_str in ERA5_DS.date_str.values])
	ERA5_clim_DS = ERA5_DS.isel(time=is_in_dr)

	# compute climatological means for each time step: group;
	ERA5_clim_DS_grouped = ERA5_clim_DS.groupby('date_str').mean(dim='time')


# filter time (and space) of ERA5 data:
ERA5_DS = ERA5_DS.sel(time=slice(set_dict['date_start'], set_dict['date_end']))


# filter location if needed:
if set_dict['region_avg']:
	# create averages for each measurement region (northern, central, southern, see Synoptic Overview)
	# northern:
	lat_bnds = {'N': np.array([81.5, 89.3]),
				'N_ex': np.array([84.5, 89.3]),		# extended part of northern region
				'C': np.array([75.0, 81.5]),
				'S': np.array([70.6, 75.0])}
	lon_bnds = {'N': np.array([-9.0, 30.0]),
				'N_ex': np.array([-54.0, -9.0]),
				'C': np.array([-9.0, 16.0]),
				'S': np.array([0.0, 23.0])}

	ERA5_DS_N, ERA5_DS_C, ERA5_DS_S = regional_averages(ERA5_DS, lat_bnds, lon_bnds)


	# repeat for climatology, if needed:
	if set_dict['load_climatology']:
		ERA5_clim_DS_grouped_N, ERA5_clim_DS_grouped_C, ERA5_clim_DS_grouped_S = regional_averages(ERA5_clim_DS_grouped, lat_bnds, lon_bnds)

		# dummy time:
		ERA5_clim_DS_grouped_N['time_d'] = xr.DataArray(ERA5_DS_N.time.values, dims=['date_str'])
		ERA5_clim_DS_grouped_C['time_d'] = xr.DataArray(ERA5_DS_C.time.values, dims=['date_str'])
		ERA5_clim_DS_grouped_S['time_d'] = xr.DataArray(ERA5_DS_S.time.values, dims=['date_str'])


# repeat processing for ML ERA5 data for anomaly maps and average over subperiod, if needed:
if set_dict['anomaly_maps']:

	ERA5_clim_DS_grouped = ERA5_clim_DS_grouped.mean('date_str')
	ERA5_DS = ERA5_DS.mean('time')


	# processing of ERA5_ML_DS:
	ERA5_ML_DS = ERA5_ML_DS.isel(time=(ERA5_ML_DS.time.dt.hour.isin(12)))	# reduces to values at 12 UTC only


	# climatological mean:
	ERA5_ML_DS['date_str'] = xr.DataArray(ERA5_ML_DS.time.dt.strftime("%m-%d %H").astype("str"), dims=['time'])
	# limit data array to certain periods and areas (also: sea only!):
	is_in_dr = np.array([date_str in date_range_array_str for date_str in ERA5_ML_DS.date_str.values])
	ERA5_ML_clim_DS = ERA5_ML_DS.isel(time=is_in_dr)

	# compute climatological means for each time step: group;
	ERA5_ML_clim_DS_grouped = ERA5_ML_clim_DS.groupby('date_str').mean(dim='time')

	# filter time (and space) of ERA5 data and location:
	ERA5_ML_DS = ERA5_ML_DS.sel(time=slice(set_dict['date_start'], set_dict['date_end']))
	ERA5_ML_DS = ERA5_ML_DS.mean('time')
	ERA5_ML_clim_DS_grouped = ERA5_ML_clim_DS_grouped.mean('date_str')



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


# some extra info for the plot:
station_coords = {'Kiruna': [20.223, 67.855],
					'Longyearbyen': [15.632, 78.222]}


if set_dict['MSLP_map']:
	for k, sel_time in enumerate(ERA5_DS.time):

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

		# some description:
		a1.text(0.02, 0.98, f"MSLP (hPa), {sel_time.dt.strftime('%Y-%m-%d %H UTC').astype('str').values}", 
				ha='left', va='top', color=(0,0,0), fontsize=fs_small, 
				bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
				zorder=10000.0, transform=a1.transAxes)


		# colorbar(s) and legends:

		if set_dict['save_figures']:
			path_plots_f = path_plots + "MSLP_map/"
			plot_file = path_plots_f + f"HALO-AC3_ERA5_MSLP_{sel_time.dt.strftime('%Y-%m-%d_%HZ').astype('str').values}.png"

			# check if folder exists:
			path_plots_dir = os.path.dirname(path_plots_f)
			if not os.path.exists(path_plots_dir):
				os.makedirs(path_plots_dir)

			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
			print("Saved " + plot_file)
		else:
			plt.show()

		plt.close()
		gc.collect()


if set_dict['met_time_series']:
	f1, a1 = plt.subplots(ncols=1, nrows=3, figsize=(9,9), constrained_layout=True)

	a1 = a1.flatten()

	# axis limits:
	labels = ['MSLP', 'T2m', 'IWV']
	units = {'MSLP': "hPa", 'T2m': "$^{\circ}\mathrm{C}$", 'IWV': "$\mathrm{kg}\,\mathrm{m}^{-2}$"}
	ax_lims = {'MSLP': [970.0, 1045.0],
				'T2m': [-30.0, 10.0],
				'IWV': [0.0, 15.0],
				'time': [ERA5_DS_N.time.values[0], ERA5_DS_N.time.values[-1]]}

	# plot MSLP (north, central, south): in hPa
	l_m = 1.2	# line width main plot
	l_c = 1.0	# line width climatology
	a1[0].plot(ERA5_DS_N.time, ERA5_DS_N.msl*0.01, color=c_N, linewidth=l_m, label='N')
	a1[0].plot(ERA5_DS_C.time, ERA5_DS_C.msl*0.01, color=c_C, linewidth=l_m, label='C')
	a1[0].plot(ERA5_DS_S.time, ERA5_DS_S.msl*0.01, color=c_S, linewidth=l_m, label='S')

	# T2m: in deg C
	a1[1].plot(ERA5_DS_N.time, ERA5_DS_N.t2m-273.15, color=c_N, linewidth=l_m)
	a1[1].plot(ERA5_DS_C.time, ERA5_DS_C.t2m-273.15, color=c_C, linewidth=l_m)
	a1[1].plot(ERA5_DS_S.time, ERA5_DS_S.t2m-273.15, color=c_S, linewidth=l_m)

	# IWV: in kg m-2
	a1[2].plot(ERA5_DS_N.time, ERA5_DS_N.tcwv, color=c_N, linewidth=l_m)
	a1[2].plot(ERA5_DS_C.time, ERA5_DS_C.tcwv, color=c_C, linewidth=l_m)
	a1[2].plot(ERA5_DS_S.time, ERA5_DS_S.tcwv, color=c_S, linewidth=l_m)

	if set_dict['load_climatology']:	# then include climat. in plot
		a1[0].plot(ERA5_clim_DS_grouped_N.time_d, ERA5_clim_DS_grouped_N.msl*0.01, color=c_N, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[0].plot(ERA5_clim_DS_grouped_C.time_d, ERA5_clim_DS_grouped_C.msl*0.01, color=c_C, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[0].plot(ERA5_clim_DS_grouped_S.time_d, ERA5_clim_DS_grouped_S.msl*0.01, color=c_S, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)

		# T2m: in deg C
		a1[1].plot(ERA5_clim_DS_grouped_N.time_d, ERA5_clim_DS_grouped_N.t2m-273.15, color=c_N, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[1].plot(ERA5_clim_DS_grouped_C.time_d, ERA5_clim_DS_grouped_C.t2m-273.15, color=c_C, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[1].plot(ERA5_clim_DS_grouped_S.time_d, ERA5_clim_DS_grouped_S.t2m-273.15, color=c_S, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)

		# IWV: in kg m-2
		a1[2].plot(ERA5_clim_DS_grouped_N.time_d, ERA5_clim_DS_grouped_N.tcwv, color=c_N, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[2].plot(ERA5_clim_DS_grouped_C.time_d, ERA5_clim_DS_grouped_C.tcwv, color=c_C, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)
		a1[2].plot(ERA5_clim_DS_grouped_S.time_d, ERA5_clim_DS_grouped_S.tcwv, color=c_S, linestyle='dashed', linewidth=l_c, alpha=0.65, zorder=-10)

		# dummy for legend:
		a1[0].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linestyle='dashed', linewidth=l_c, alpha=0.65, label='Climatology 1979-2022')


	# aux info: subplot identifiers:
	a1[0].text(0.01, 0.95, "(a)", fontsize=fs, ha='left', va='top', transform=a1[0].transAxes)
	a1[1].text(0.01, 0.95, "(b)", fontsize=fs, ha='left', va='top', transform=a1[1].transAxes)
	a1[2].text(0.01, 0.95, "(c)", fontsize=fs, ha='left', va='top', transform=a1[2].transAxes)

	# legend:
	lh, ll = a1[0].get_legend_handles_labels()
	a1[0].legend(lh, ll, loc='lower left', ncol=4, bbox_to_anchor=(0.0, 1.0), fontsize=fs)


	# further parameters:
	x_ticks = np.arange(set_dict['date_start'], set_dict['date_end_plus'], np.timedelta64(24, 'h'), dtype='datetime64[h]')
	x_tick_labels = list()
	for k, x_tick in enumerate(x_ticks):	# set every third tick a tick label
		if k % 3 == 1:
			x_tick_labels.append(numpydatetime64_to_datetime(x_tick).strftime("%d %b"))
		else:
			x_tick_labels.append("")

	for ix, ax in enumerate(a1):
		# set axis limits:
		ax.set_xlim(x_ticks[0], np.datetime64(set_dict['date_end_plus']))
		ax.set_ylim(ax_lims[labels[ix]][0], ax_lims[labels[ix]][1])

		# set ticks and tick labels and parameters:
		ax.xaxis.set_major_formatter(dt_fmt)
		ax.tick_params(axis='both', labelsize=fs_dwarf)
		ax.set_xticks(x_ticks)
		if ix < 2: 
			ax.xaxis.set_ticklabels([])
		else:
			ax.set_xticklabels(x_tick_labels)

		# grid:
		ax.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# labels:
		ax.set_ylabel(f"{labels[ix]} ({units[labels[ix]]})", fontsize=fs)
		if ix == 2: ax.set_xlabel("Date of 2022", fontsize=fs)

	if set_dict['save_figures']:
		path_plots_f = path_plots + "met_time_series/"

		# check if folder exists:
		path_plots_dir = os.path.dirname(path_plots_f)
		if not os.path.exists(path_plots_dir):
			os.makedirs(path_plots_dir)

		plot_name_add = ""
		plot_name_add2 = "_2022"
		if set_dict['daily']: plot_name_add = "_daily"
		if set_dict['load_climatology']: plot_name_add2 = "_1979-2022"
		plotname = f"HALO-AC3_ERA5_time_series{plot_name_add}_MSLP_T2m_IWV{plot_name_add2}"
		f1.savefig(path_plots_f + plotname + ".png", dpi=300, bbox_inches='tight')
	else:
		plt.show()

		plt.close()


if set_dict['anomaly_maps']:

	f1, a1 = plt.subplots(nrows=3, ncols=1, subplot_kw={'projection': sel_projection}, figsize=(16,24), constrained_layout=True)
	a1 = a1.flatten()

	for ax in a1:
		ax.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

		# add some land marks:
		ax.coastlines(resolution="50m", zorder=10000.0, linewidth=0.75)
		ax.add_feature(cartopy.feature.BORDERS)
		# ax.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		# ax.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		ax.gridlines(draw_labels=True, color=(0.8,0.8,0.8))

	# plot anomalies:
	levels_1 = np.concatenate((np.arange(-8.0, -0.9, 0.5), np.arange(1.0, 8.1, 0.5)))
	levels_2 = np.arange(-20.0, 30.1, 2.0)
	levels_hat = np.arange(8.0, 9999.1, 100.0)		### for hatches
	n_levels = len(levels_1)
	cmap = mpl.cm.get_cmap('RdBu_r', n_levels)
	cmap = cmap(range(n_levels))			# must be done to access the colormap values
	cmap[np.where(levels_1==-1.0)[0][0],:] = np.array([1.0,1.0,1.0,1.0])			# adapt colormap
	cmap = mpl.colors.ListedColormap(cmap)
	norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_1[0], vmax=levels_1[-1])
	var_plot = ERA5_DS.t2m - ERA5_clim_DS_grouped.t2m
	contourf_0 = a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							cmap=cmap, norm=norm, levels=levels_1, extend='both', 
							transform=ccrs.PlateCarree())
	contour_05 = a1[0].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							levels=levels_2, colors='black', linewidths=0.75, linestyles='dashed',
							transform=ccrs.PlateCarree())
	a1[0].clabel(contour_05, levels=levels_2, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)
	# hatched area for deviation > 10.0:
	a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, colors='none', 
					levels=levels_hat, hatches=['///'], transform=ccrs.PlateCarree())


	# T850:
	var_plot = ERA5_ML_DS.t - ERA5_ML_clim_DS_grouped.t
	contourf_1 = a1[1].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							cmap=cmap, norm=norm, levels=levels_1, extend='both', 
							transform=ccrs.PlateCarree())
	contour_06 = a1[1].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							levels=levels_2, colors='black', linewidths=0.75, linestyles='dashed',
							transform=ccrs.PlateCarree())
	a1[1].clabel(contour_06, levels=levels_2, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)
	a1[1].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, colors='none', 
					levels=levels_hat, hatches=['\\\\'], transform=ccrs.PlateCarree())


	# IWV:
	var_plot = ERA5_DS.tcwv - ERA5_clim_DS_grouped.tcwv
	levels_3 = np.arange(-3.0, 3.1, 0.05)
	levels_4 = np.arange(-10.0, 10.1, 2.0)
	cmap = mpl.cm.get_cmap('BrBG', len(levels_3))
	norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_3[0], vmax=levels_3[-1])
	contourf_2 = a1[2].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							cmap=cmap, norm=norm, levels=levels_3, extend='both', 
							transform=ccrs.PlateCarree())
	contour_07 = a1[2].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							levels=levels_4, colors='black', linewidths=0.75, linestyles='dashed',
							transform=ccrs.PlateCarree())
	a1[2].clabel(contour_07, levels=levels_4, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


	# some description:
	a1[0].text(0.02, 0.98, f"{set_dict['date_start']} - {set_dict['date_end']}", 
			ha='left', va='top', color=(0,0,0), fontsize=fs_dwarf, 
			bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0, transform=a1[0].transAxes)


	# colorbar(s) and legends:
	cb_var = f1.colorbar(mappable=contourf_0, ax=a1[0], extend='both', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="T2m anomaly (K)", fontsize=fs_small)
	cb_var.ax.tick_params(labelsize=fs_dwarf)

	cb_var = f1.colorbar(mappable=contourf_1, ax=a1[1], extend='both', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="T850hPa anomaly (K)", fontsize=fs_small)
	cb_var.ax.tick_params(labelsize=fs_dwarf)

	cb_var = f1.colorbar(mappable=contourf_2, ax=a1[2], extend='both', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="IWV anomaly (kg m-2)", fontsize=fs_small)
	cb_var.ax.tick_params(labelsize=fs_dwarf)


	if set_dict['save_figures']:
		path_plots_f = path_plots + "anomaly_maps/"
		plot_file = path_plots_f + f"HALO-AC3_ERA5_anomaly_T2m_T850_IWV_hatching.png"

		# check if folder exists:
		path_plots_dir = os.path.dirname(path_plots_f)
		if not os.path.exists(path_plots_dir):
			os.makedirs(path_plots_dir)

		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)
	else:
		plt.show()

	plt.close()
	gc.collect()

	# # # # # # # # # # # # # # # # f1, a1 = plt.subplots(nrows=3, ncols=1, subplot_kw={'projection': sel_projection}, figsize=(16,24), constrained_layout=True)
	# # # # # # # # # # # # # # # # a1 = a1.flatten()

	# # # # # # # # # # # # # # # # for ax in a1:
		# # # # # # # # # # # # # # # # ax.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

		# # # # # # # # # # # # # # # # # add some land marks:
		# # # # # # # # # # # # # # # # ax.coastlines(resolution="50m", zorder=10000.0, linewidth=0.75)
		# # # # # # # # # # # # # # # # ax.add_feature(cartopy.feature.BORDERS)
		# # # # # # # # # # # # # # # # # ax.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
		# # # # # # # # # # # # # # # # # ax.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
		# # # # # # # # # # # # # # # # ax.gridlines(draw_labels=True, color=(0.8,0.8,0.8))

	# # # # # # # # # # # # # # # # # plot anomalies:
	# # # # # # # # # # # # # # # # levels_1 = np.arange(-8.0, 8.1, 0.5)
	# # # # # # # # # # # # # # # # levels_2 = np.arange(-20.0, 30.1, 2.0)
	# # # # # # # # # # # # # # # # levels_hat = np.arange(8.0, 9999.1, 100.0)		### for hatches
	# # # # # # # # # # # # # # # # n_levels = len(levels_1)
	# # # # # # # # # # # # # # # # cmap = mpl.cm.get_cmap('RdBu_r', n_levels)
	# # # # # # # # # # # # # # # # norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_1[0], vmax=levels_1[-1])
	# # # # # # # # # # # # # # # # var_plot = ERA5_DS.t2m - ERA5_clim_DS_grouped.t2m
	# # # # # # # # # # # # # # # # contourf_0 = a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							# # # # # # # # # # # # # # # # cmap=cmap, norm=norm, levels=levels_1, extend='both', 
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # contour_05 = a1[0].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							# # # # # # # # # # # # # # # # levels=levels_2, colors='black', linewidths=0.75, linestyles='dashed',
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # a1[0].clabel(contour_05, levels=levels_2, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)
	# # # # # # # # # # # # # # # # # hatched area for deviation > 10.0:
	# # # # # # # # # # # # # # # # a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, colors='none', 
					# # # # # # # # # # # # # # # # levels=levels_hat, hatches=['///'], transform=ccrs.PlateCarree())


	# # # # # # # # # # # # # # # # # T850:
	# # # # # # # # # # # # # # # # var_plot = ERA5_ML_DS.t - ERA5_ML_clim_DS_grouped.t
	# # # # # # # # # # # # # # # # contourf_1 = a1[1].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							# # # # # # # # # # # # # # # # cmap=cmap, norm=norm, levels=levels_1, extend='both', 
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # contour_06 = a1[1].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							# # # # # # # # # # # # # # # # levels=levels_2, colors='black', linewidths=0.75, linestyles='dashed',
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # a1[1].clabel(contour_06, levels=levels_2, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)
	# # # # # # # # # # # # # # # # a1[1].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, colors='none', 
					# # # # # # # # # # # # # # # # levels=levels_hat, hatches=['\\\\'], transform=ccrs.PlateCarree())


	# # # # # # # # # # # # # # # # # IWV:
	# # # # # # # # # # # # # # # # var_plot = ERA5_DS.tcwv - ERA5_clim_DS_grouped.tcwv
	# # # # # # # # # # # # # # # # levels_3 = np.arange(-3.0, 3.1, 0.05)
	# # # # # # # # # # # # # # # # levels_4 = np.arange(-10.0, 10.1, 2.0)
	# # # # # # # # # # # # # # # # cmap = mpl.cm.get_cmap('BrBG', len(levels_3))
	# # # # # # # # # # # # # # # # norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_3[0], vmax=levels_3[-1])
	# # # # # # # # # # # # # # # # contourf_2 = a1[2].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							# # # # # # # # # # # # # # # # cmap=cmap, norm=norm, levels=levels_3, extend='both', 
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # contour_07 = a1[2].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
							# # # # # # # # # # # # # # # # levels=levels_4, colors='black', linewidths=0.75, linestyles='dashed',
							# # # # # # # # # # # # # # # # transform=ccrs.PlateCarree())
	# # # # # # # # # # # # # # # # a1[2].clabel(contour_07, levels=levels_4, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


	# # # # # # # # # # # # # # # # # some description:
	# # # # # # # # # # # # # # # # a1[0].text(0.02, 0.98, f"{set_dict['date_start']} - {set_dict['date_end']}", 
			# # # # # # # # # # # # # # # # ha='left', va='top', color=(0,0,0), fontsize=fs_dwarf, 
			# # # # # # # # # # # # # # # # bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			# # # # # # # # # # # # # # # # zorder=10000.0, transform=a1[0].transAxes)


	# # # # # # # # # # # # # # # # # colorbar(s) and legends:
	# # # # # # # # # # # # # # # # cb_var = f1.colorbar(mappable=contourf_0, ax=a1[0], extend='both', orientation='horizontal', 
							# # # # # # # # # # # # # # # # fraction=0.06, pad=0.04, shrink=0.8)
	# # # # # # # # # # # # # # # # cb_var.set_label(label="T2m anomaly (K)", fontsize=fs_small)
	# # # # # # # # # # # # # # # # cb_var.ax.tick_params(labelsize=fs_dwarf)

	# # # # # # # # # # # # # # # # cb_var = f1.colorbar(mappable=contourf_1, ax=a1[1], extend='both', orientation='horizontal', 
							# # # # # # # # # # # # # # # # fraction=0.06, pad=0.04, shrink=0.8)
	# # # # # # # # # # # # # # # # cb_var.set_label(label="T850hPa anomaly (K)", fontsize=fs_small)
	# # # # # # # # # # # # # # # # cb_var.ax.tick_params(labelsize=fs_dwarf)

	# # # # # # # # # # # # # # # # cb_var = f1.colorbar(mappable=contourf_2, ax=a1[2], extend='both', orientation='horizontal', 
							# # # # # # # # # # # # # # # # fraction=0.06, pad=0.04, shrink=0.8)
	# # # # # # # # # # # # # # # # cb_var.set_label(label="IWV anomaly (kg m-2)", fontsize=fs_small)
	# # # # # # # # # # # # # # # # cb_var.ax.tick_params(labelsize=fs_dwarf)


	# # # # # # # # # # # # # # # # if set_dict['save_figures']:
		# # # # # # # # # # # # # # # # path_plots_f = path_plots + "anomaly_maps/"
		# # # # # # # # # # # # # # # # plot_file = path_plots_f + f"HALO-AC3_ERA5_anomaly_T2m_T850_IWV_hatching.png"

		# # # # # # # # # # # # # # # # # check if folder exists:
		# # # # # # # # # # # # # # # # path_plots_dir = os.path.dirname(path_plots_f)
		# # # # # # # # # # # # # # # # if not os.path.exists(path_plots_dir):
			# # # # # # # # # # # # # # # # os.makedirs(path_plots_dir)

		# # # # # # # # # # # # # # # # f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		# # # # # # # # # # # # # # # # print("Saved " + plot_file)
	# # # # # # # # # # # # # # # # else:
		# # # # # # # # # # # # # # # # plt.show()

	# # # # # # # # # # # # # # # # plt.close()
	# # # # # # # # # # # # # # # # gc.collect()
	pdb.set_trace()