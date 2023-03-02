import xarray as xr
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")

import matplotlib.pyplot as plt
import pdb

import sys
sys.path.insert(0, "/mnt/f/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *

"""
	Dirty script to quickly create a Hovmoeller plot of heat and moisture fluxes
	for HALO-(AC)3.
"""


# Paths
path_data = "/mnt/f/heavy_data/ERA5_data/single_level/"
path_plots = "/mnt/f/Studium_NIM/work/Documents/DIY/HALO-AC3_syn_overview/contributions/Schirmacher_fluxes/"

# settings:
set_dict = {'weighted_mean': False,
			'daily': False,				# if True, only daily means will be considered, False yields full resolution (3-hourly)
			'cr_avg': True,				# computes an average over the entire central region
			'maxima': False,				# if True (False), maxima (mean) over the years for each 3h period of 03-07 - 04-12 will be computed
			'hovmoeller_plot': False,	# if True: extended central region; False: only central region
			'incl_climatology': False,	# if True, all years (1979-2022) will be considered, False: only 2022
			'plot_map': False}			# map plot of heat or moisture flux


# import data:
file = "ERA5_single_level_heat_flux_moisture_flux_6hourly_1979-2022.nc"
DS = xr.open_dataset(path_data+file)
if not set_dict['incl_climatology']: DS = DS.sel(time='2022')		# if you want to restrict to 2022 only
LSM_DS = xr.open_dataset(path_data + "ERA5_single_level_20220406-20220409_HALO-AC3.nc")	# contains land sea mask


# compute climatological mean: first reduce to desired period within a year
if set_dict['daily']:
	date_range_array = numpydatetime64_to_datetime(np.arange("2022-03-07", "2022-04-13", dtype='datetime64[D]'))
	date_range_array_str = np.array([dra.strftime("%m-%d") for dra in date_range_array])
	DS['month_day_str'] = xr.DataArray(DS.time.dt.strftime("%m-%d").astype("str"), dims=['time'])
else:
	date_range_array = numpydatetime64_to_datetime(np.arange("2022-03-07", "2022-04-13", np.timedelta64(3, "h"), dtype='datetime64[s]'))
	date_range_array_str = np.array([dra.strftime("%m-%d %H") for dra in date_range_array])
	DS['month_day_str'] = xr.DataArray(DS.time.dt.strftime("%m-%d %H").astype("str"), dims=['time'])

is_in_dr = np.array([month_day_str in date_range_array_str for month_day_str in DS.month_day_str.values])
DS = DS.isel(time=is_in_dr)


# limit data array to certain periods and areas (also: sea only!):
lat_lims = np.array([75.0, 81.5])
if set_dict['hovmoeller_plot']: 
	lon_lims = np.array([-20.0, 25.0])
else:	# then, central region is chosen automatically
	lon_lims = np.array([-9.0, 16.0])

LSM_DA = LSM_DS.sel(latitude=slice(lat_lims[1], lat_lims[0]), 
					longitude=slice(lon_lims[0], lon_lims[1])).lsm[0,:,:]	# LSM=1 == land
DS = DS.sel(time=slice("1979", "2022"), latitude=slice(lat_lims[1], lat_lims[0]), 
					longitude=slice(lon_lims[0], lon_lims[1]))				# here, you can limit to certain years

# create Land Sea Mask:
LSM_mask = LSM_DA.values.flatten()
LSM_mask = np.reshape(LSM_mask == 0.0, (len(LSM_DA.latitude), len(LSM_DA.longitude)))	# True: sea, False: land
LSM_mask_flat = LSM_mask.flatten()


# group by month-day-str: computes # mean/max/min over all times for each group 
# --> meanmax/min over all years for each month-day (month-day-hour)
if set_dict['maxima']:
	DS_grouped_mean = DS.groupby('month_day_str').max(dim='time') 
else:
	DS_grouped_mean = DS.groupby('month_day_str').mean(dim='time')


# apply land sea mask:
len_time_grouped = len(DS_grouped_mean.month_day_str)
for k in range(len_time_grouped):
	DS_grouped_mean_heat_flux_flat = DS_grouped_mean['p70.162'].values[k,:,:].flatten()
	DS_grouped_mean_mois_flux_flat = DS_grouped_mean['p72.162'].values[k,:,:].flatten()

	# apply mask:
	DS_grouped_mean_heat_flux_flat[~LSM_mask_flat] = np.nan
	DS_grouped_mean_mois_flux_flat[~LSM_mask_flat] = np.nan

	DS_grouped_mean['p70.162'][k,:,:] = xr.DataArray(np.reshape(DS_grouped_mean_heat_flux_flat, (len(DS_grouped_mean.latitude), len(DS_grouped_mean.longitude))))
	DS_grouped_mean['p72.162'][k,:,:] = xr.DataArray(np.reshape(DS_grouped_mean_mois_flux_flat, (len(DS_grouped_mean.latitude), len(DS_grouped_mean.longitude))))


# compute average over latitudes:
# simple mean, not weighted:
DS_grouped_mean['heat_flux'] = DS_grouped_mean['p70.162'].mean('latitude')
DS_grouped_mean['mois_flux'] = DS_grouped_mean['p72.162'].mean('latitude')

# weighted mean:
if set_dict['weighted_mean']:
	weights = np.cos(np.deg2rad(DS_grouped_mean.latitude.values))
	# weights = np.ones_like(DS_grouped_mean.latitude.values)		# yields the same as DS_grouped_mean[...].mean('latitude')
	heat_flux_hov = np.zeros(DS_grouped_mean['p70.162'].shape)
	mois_flux_hov = np.zeros(DS_grouped_mean['p72.162'].shape)
	for k, lat in enumerate(DS_grouped_mean.latitude.values):
		heat_flux_hov[:,k,:] = weights[k]*DS_grouped_mean['p70.162'].values[:,k,:]
		mois_flux_hov[:,k,:] = weights[k]*DS_grouped_mean['p72.162'].values[:,k,:]

	# also need to respect that some missing values exist: need to exclude some weights for certain longitudes and latitudes:
	heat_flux_hov_mean = np.zeros(DS_grouped_mean['heat_flux'].shape)
	mois_flux_hov_mean = np.zeros(DS_grouped_mean['mois_flux'].shape)
	for l, lon in enumerate(DS_grouped_mean.longitude.values):
		# identify non nan indices via land sea mask:
		heat_flux_hov_mean[:,l] = np.nansum(heat_flux_hov[:,LSM_mask[:,l],l], axis=1) / np.sum(weights[LSM_mask[:,l]])
		mois_flux_hov_mean[:,l] = np.nansum(mois_flux_hov[:,LSM_mask[:,l],l], axis=1) / np.sum(weights[LSM_mask[:,l]])


	# heat_flux_hov_mean = np.nansum(heat_flux_hov, axis=1) / np.sum(weights)		# this uses also those weights where actually nan values are set to zero and should be excluded --> false, i think
	# mois_flux_hov_mean = np.nansum(mois_flux_hov, axis=1) / np.sum(weights)
else:
	weights = None
	heat_flux_hov_mean = None
	mois_flux_hov_mean = None


# climatological or non-climatological maxima for latitude-averaged fluxes can be found when computing i.e.,
# np.nanmax(heat_flux_hov_mean)


# average over entire central region (non-weighted):
if set_dict['cr_avg']:
	# convert to float64?:
	DS_grouped_mean['p70.162'] = DS_grouped_mean['p70.162'].astype(np.float64)
	DS_grouped_mean['p72.162'] = DS_grouped_mean['p72.162'].astype(np.float64)

	# weighted mean: weights == cos of latitudes:
	weights = np.cos(np.deg2rad(DS_grouped_mean.latitude.values))
	# weights = np.ones_like(DS_grouped_mean.latitude.values)		# uniform weights for testing

	# regional avg: flatten weights and values: first transform weights to a 2D array, then flatten it:
	weights = np.reshape(np.repeat(weights, len(DS_grouped_mean.longitude.values)), DS_grouped_mean['p70.162'].shape[1:]).flatten()

	# compute weighted regional averages: (yields the same as DS_grouped_mean['p70.162'].mean(['latitude', 'longitude']) if weights == np.ones(...)
	heat_flux_hov_mean_cr = np.zeros(DS_grouped_mean['month_day_str'].shape)
	mois_flux_hov_mean_cr = np.zeros(DS_grouped_mean['month_day_str'].shape)
	for tt in range(len_time_grouped):
		heat_flux_hov_mean_cr[tt] = np.nansum((DS_grouped_mean['p70.162'].values[tt,:,:].flatten()*weights)[LSM_mask_flat]) / np.sum(weights[LSM_mask_flat])
		mois_flux_hov_mean_cr[tt] = np.nansum((DS_grouped_mean['p72.162'].values[tt,:,:].flatten()*weights)[LSM_mask_flat]) / np.sum(weights[LSM_mask_flat])


	# simple, unweighted mean that doesn't respect the increasing density of grid points with increasing
	# latitudes:
	DS_grouped_mean['heat_flux_mean_cr'] = DS_grouped_mean['p70.162'].mean(['latitude', 'longitude'])
	DS_grouped_mean['mois_flux_mean_cr'] = DS_grouped_mean['p72.162'].mean(['latitude', 'longitude'])

	# average over entire HALO-AC3 period (time wise) can be found by np.nanmean(DS_grouped_mean['p70.162']) 
	# or DS_grouped_mean['heat_flux_mean_cr'].mean('month_day_str')

pdb.set_trace()


# visualize:
if set_dict['plot_map']:
	import cartopy
	import cartopy.crs as ccrs
	import cartopy.io.img_tiles as cimgt

	# map_settings:
	lon_centre = 5.0
	lat_centre = 75.0
	lon_lat_extent = [-40.0, 40.0, 65.0, 90.0]
	# lon_lat_extent = [-30.0, 30.0, 70.0, 85.0]		# (zoomed in)
	sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)

	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

	# add some land marks:
	a1.coastlines(resolution="50m", zorder=9999.0)
	a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
	a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

	# plot var:
	levels = np.arange(-35.e+9, 1.76e+10, 1.0e+9)
	levels = np.arange(-35.e+9, 1.76e+10, 1.0e+9)
	norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels[0], vmax=levels[-1])
	cmap = mpl.cm.get_cmap('seismic', len(levels))
	var_plot = DS_grouped_mean['p70.162'][15,:,:]

	contourf_plot = a1.contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
							cmap=cmap, levels=levels, norm=norm,
							transform=ccrs.PlateCarree())

	# colorbar(s):
	cb_var = f1.colorbar(mappable=contourf_plot, ax=a1, orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_var.set_label(label="p70.162", fontsize=12)
	cb_var.ax.tick_params(labelsize=10)

	plt.show()
	plt.close()



f1, (a1, a2) = plt.subplots(1,2)
f1.set_size_inches((14,9))

if set_dict['incl_climatology']:
	bounds = np.arange(-35.e+9, 1.76e+10, 0.5e+9)
	bounds_pos = np.arange(0., 1.76e+10, 1.0e+9)
	bounds_neg = np.arange(-35.e+9, 0., 1.0e+9)
else:
	bounds = np.arange(-54.e+9, 100e+9, 4e+9)			# FOR 2022
	bounds_pos = np.arange(0., 10.e+10, 8.0e+9)
	bounds_neg = np.arange(-54.e+9, 0., 8.0e+9)
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=bounds[0], vmax=bounds[-1])

if set_dict['weighted_mean']:
	heat_flux_plot = a1.contourf(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, heat_flux_hov_mean, 
								cmap='seismic', norm=norm, levels=bounds)
	contours_pos = a1.contour(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, heat_flux_hov_mean,
								levels=bounds_pos[::2], linewidths=0.75, colors=np.full(bounds_pos.shape, "#000000"))
	contours_neg = a1.contour(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, heat_flux_hov_mean,
								levels=bounds_neg[::2], linewidths=0.75, colors=np.full(bounds_neg.shape, "#000000"), linestyles='dashed')

else:
	heat_flux_plot = a1.contourf(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, DS_grouped_mean['heat_flux'].values, 
								cmap='seismic', norm=norm, levels=bounds)
	contours_pos = a1.contour(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, DS_grouped_mean['heat_flux'].values,
								levels=bounds_pos[::2], linewidths=0.75, colors=np.full(bounds_pos.shape, "#000000"))
	contours_neg = a1.contour(DS_grouped_mean['heat_flux'].longitude.values, DS_grouped_mean['heat_flux'].month_day_str.values, DS_grouped_mean['heat_flux'].values,
								levels=bounds_neg[::2], linewidths=0.75, colors=np.full(bounds_neg.shape, "#000000"), linestyles='dashed')
cb1 = f1.colorbar(heat_flux_plot, ax=a1, orientation='vertical')
cb1.set_label(f"{DS_grouped_mean['heat_flux'].name} ({DS['p70.162'].units})")


if set_dict['incl_climatology']:
	bounds = np.arange(-120, 106, 5)			# FOR CLIMATOLOGY
	bounds_pos = np.arange(0, 106, 5)
	bounds_neg = np.arange(-120, 0, 5)
else:
	bounds = np.arange(-100, 400, 25)			# FOR 2022
	bounds_pos = np.arange(0, 400, 25)
	bounds_neg = np.arange(-100, 0, 25)
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=bounds[0], vmax=bounds[-1])
if set_dict['weighted_mean']:
	wv_flux_plot = a2.contourf(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, mois_flux_hov_mean, 
								cmap='seismic', norm=norm, levels=bounds)
	contours_pos = a2.contour(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, mois_flux_hov_mean,
								levels=bounds_pos[::4], linewidths=0.75, colors=np.full(bounds_pos.shape, "#000000"))
	contours_neg = a2.contour(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, mois_flux_hov_mean,
								levels=bounds_neg[::4], linewidths=0.75, colors=np.full(bounds_neg.shape, "#000000"), linestyles='dashed')

else:
	wv_flux_plot = a2.contourf(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, DS_grouped_mean['mois_flux'].values, 
								cmap='seismic', norm=norm, levels=bounds)
	contours_pos = a2.contour(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, DS_grouped_mean['mois_flux'].values,
								levels=bounds_pos[::4], linewidths=0.75, colors=np.full(bounds_pos.shape, "#000000"))
	contours_neg = a2.contour(DS_grouped_mean['mois_flux'].longitude.values, DS_grouped_mean['mois_flux'].month_day_str.values, DS_grouped_mean['mois_flux'].values,
								levels=bounds_neg[::4], linewidths=0.75, colors=np.full(bounds_neg.shape, "#000000"), linestyles='dashed')
cb2 = f1.colorbar(wv_flux_plot, ax=a2, orientation='vertical')
cb2.set_label(f"{DS_grouped_mean['mois_flux'].name} ({DS['p72.162'].units})")

y_ticks = a1.get_yticks()
y_ticks = y_ticks[::16]
y_tick_labels = [mds for mds in DS_grouped_mean['mois_flux'].month_day_str.values[::16]]
a1.set_yticks(y_ticks)
a1.set_yticklabels(y_tick_labels)
a2.set_yticks(y_ticks)
a2.set_yticklabels([])

x_ticks = np.arange(-20.0, 25.1, 5.0)
x_tick_labels = list()
for x_tick in x_ticks:
	if x_tick < 0:
		x_tick_labels.append(f"{int(-1*x_tick)}" + "$^{\circ}$W")
	else:
		x_tick_labels.append(f"{int(x_tick)}" + "$^{\circ}$E")
a1.set_xticks(x_ticks)
a1.set_xticklabels(x_tick_labels)
a2.set_xticks(x_ticks)
a2.set_xticklabels(x_tick_labels)

plt.show()
# f1.savefig(path_plots + "HALO-AC3_ERA5_hovmoeller_heat_moisture_flux_3hourly_resolution_2022.png", 
			# dpi=300, bbox_inches='tight')
pdb.set_trace()