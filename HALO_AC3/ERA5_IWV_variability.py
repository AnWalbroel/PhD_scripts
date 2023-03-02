import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import datetime as dt
import glob
from geopy import distance

import pdb


def get_samples(
	DS,
	chosen_points,
	lat_tol,
	lon_tol,
	sample_var):

	"""
	Finds and returns a certain sample of a quantity at given locations. Latitudinal and 
	longitudinal tolerance values speed up the search by limiting distance computation to
	chosen_points +/- tolerance grid points.

	Parameters:
	-----------
	DS : xarray data set
		Dataset containing latitude, longitude and data variables from which the 
		sample shall be created from.
	chosen_points : list
		List of coordinate pairs (also as list) of the locations of interest. Latitude (in deg N) must be
		mentioned before longitude (in deg E). 
		Example: chosen_points = [[75.0, 130.0], [86.2, -25.0]]
	lat_tol : float
		To speed up the search, latitude tolerance will be used to limit distance computation to 
		chosen point latitude +/- tolerance. In deg.
	lon_tol : float
		To speed up the search, longitude tolerance will be used to limit distance computation to 
		chosen point longitude +/- tolerance. In deg.
	sample_var : str
		Precise string description of the sample variable. Must be in DS.
	"""

	if not sample_var in DS.data_vars: 
		raise ValueError("sample_var must be a variable in DS.")

	samples = np.full((len(chosen_points), len(DS.time.values)), np.nan)
	for idx, sel_point in enumerate(chosen_points):
		# first, reduce to grid points with a certain radius around the wanted location
		# (given by lat and lon tolerance):
		idx_lat = np.where(np.abs(DS.latitude.values - sel_point[0]) <= lat_tol)[0]
		idx_lon = np.where(np.abs(DS.longitude.values - sel_point[1]) <= lon_tol)[0]

		assert (len(idx_lat) > 0 and len(idx_lon) > 0)

		geod_dist = np.full((len(idx_lat), len(idx_lon)), -9999.0)
		for ii, i_lat in enumerate(idx_lat):
			for jj, i_lon in enumerate(idx_lon):
				geod_dist[ii,jj] = distance.distance((sel_point[0], sel_point[1]), 
													(DS.latitude.values[idx_lat[ii]], DS.longitude.values[idx_lon[jj]])).km

		# extract correct indices:
		idx_lat_lon = [np.argmin(geod_dist, axis=0)[0], np.argmin(geod_dist, axis=1)[0]]
		print((DS.latitude.values[idx_lat[idx_lat_lon[0]]], DS.longitude.values[idx_lon[idx_lat_lon[0]]]), sel_point)
		samples[idx,:] = DS[sample_var].values[:,idx_lat[idx_lat_lon[0]],idx_lon[idx_lat_lon[0]]]


	return samples


###################################################################################################
###################################################################################################


"""
	Script to visualize IWV variability (boxplot for all years, histogram, map plot of
	IWV variability (stddev)). Certain period and certain grid points can be selected.
	- find and import ERA5 data
	- filter for time and space
	- visualize
"""

# Paths:
path_era5 = "/net/blanc/awalbroe/Data/ERA5_HALO_AC3/"
path_plots = "/net/blanc/awalbroe/Plots/HALO_AC3_quicklooks/ERA5/"

# some other settings:
save_figures = True		# if True, saved to .png, if False, just show plot
map_plot = False		# creates map plot of IWV variability		
IWV_hist_plot = False		# histogram showing IWV variability
IWV_box_plot = True		# box plot


# choose date range and location boundaries:
date_start = "1979-01-01"
date_end = "2022-01-01"

lat_bnds = [[70.6, 75.0], [75.0, 81.5], [81.5, 89.3], [84.5, 89.3]]
lon_bnds = [[0.0, 23.0], [-9.0, 16.0], [-9.0, 30.0], [-54.0, -9.0]]

# choose samples (certain coordinate points) for IWV histogram and boxplot:
chosen_points = [[75.0, 0.0], [80.0, 0.0], [85.0, 0.0], [85.0, 60.0], [85.0, 120.0], 
					[85.0, -180.0], [85.0, -120.0], [85.0, -60.0], [87.5, 0.0], [87.5, 90.0], 
					[87.5, -180.0], [87.5, -90.0]]


# find and import era 5 data:
files = sorted(glob.glob(path_era5 + "*.nc"))
file_era5 = files[0]
ERA5_DS = xr.open_dataset(file_era5)


# grid resolution based tolerance of latitude and longitude
lat_tol = np.abs(np.ceil(np.diff(ERA5_DS.latitude.values).mean()*4))
lon_tol = np.abs(np.ceil(np.diff(ERA5_DS.longitude.values).mean()*4))


# filter time and space of ERA5 data: eventually manually select important longitudes to further 
# reduce memory usage
ERA5_DS = ERA5_DS.sel(time=slice(date_start, date_end))
lon_interesting = np.unique(np.asarray([np.where(ERA5_DS.longitude.values == POI[1])[0] for POI in chosen_points]))
lat_interesting = np.unique(np.asarray([np.where(ERA5_DS.latitude.values == POI[0])[0] for POI in chosen_points]))
ERA5_DS = ERA5_DS.isel(longitude=lon_interesting, latitude=lat_interesting)

"""
# filtering for non-rectangular box by first splitting that box into pieces
# before merging together again.
DS_list = list()
for lon_bnd, lat_bnd in zip(lon_bnds, lat_bnds):
	DS_list.append(ERA5_DS.sel(longitude=slice(lon_bnd[0], lon_bnd[1]), latitude=slice(lat_bnd[1], lat_bnd[0])))

ERA5_DS = xr.merge(DS_list, join='outer')
"""


# Visualize:

fs = 16
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


# MAP PLOT:
if map_plot:
	# map_settings:
	lon_centre = 5.0
	lat_centre = 75.0
	lon_lat_extent = [-40.0, 40.0, 65.0, 90.0]
	sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)

	# some preprocessing of plottable data: 
	# ERA5_DS['tcwv'] = xr.where(ERA5_DS['lsm'] > 0.33, np.nan, ERA5_DS['tcwv'])
	ERA5_DS['tcwv_std'] = ERA5_DS.tcwv.std(dim='time')
	ERA5_DS['tcwv_rel_std'] = ERA5_DS['tcwv_std'] / ERA5_DS.tcwv.mean(dim='time')

	# some extra info for the plot:
	station_coords = {'Kiruna': [20.223, 67.855],
						'Longyearbyen': [15.632, 78.222]}

	f1 = plt.figure(figsize=(10,7.5))
	a1 = plt.axes(projection=sel_projection)
	a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
	a1.add_image(cimgt.Stamen('terrain-background'), 4)

	# add some land marks:
	a1.coastlines(resolution="50m", zorder=9999.0)
	a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
	a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

	# plot tcwv:
	iwv_levels = np.arange(0.0, 100.001, 5.0)
	cmap = mpl.cm.get_cmap('OrRd', len(iwv_levels))
	IWV_plot = a1.contourf(ERA5_DS.longitude.values, ERA5_DS.latitude.values, 100*ERA5_DS.tcwv_rel_std[:,:].values, 
							cmap=cmap, levels=iwv_levels, extend='max',
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
			color=(0,0,0), fontsize=fs, transform=text_transform, 
			bbox={'facecolor': (0.75,0.75, 0.75,0.5), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0)
	a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "Longyearbyen",
			ha='left', va='bottom',
			color=(0,0,0), fontsize=fs, transform=text_transform, 
			bbox={'facecolor': (0.75,0.75, 0.75,0.5), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
			zorder=10000.0)

	# also mark the selected grid points:
	for sel_point in chosen_points:
		a1.plot(sel_point[1], sel_point[0], color=(1,0,0), marker='v', markersize=marker_size,
				markeredgecolor=(0,0,0), transform=ccrs.PlateCarree(), zorder=10000.0)

	# colorbar(s):
	cb_iwv = f1.colorbar(mappable=IWV_plot, ax=a1, extend='max', orientation='horizontal', 
							fraction=0.06, pad=0.04, shrink=0.8)
	cb_iwv.set_label(label="Relative standard deviation ($\%$)", fontsize=fs_small)
	cb_iwv.ax.tick_params(labelsize=fs_dwarf)


	if save_figures:
		f1.savefig(path_plots + "ERA5.png", dpi=300)
	else:
		plt.show()


if IWV_hist_plot:

	def hist_plot(data):
		# count non nan data
		where_nonnan = np.where(~np.isnan(data))[0]
		data = data[where_nonnan]
		n_data = float(len(data))
		weights_data = np.ones((int(n_data),)) / n_data


		fig1 = plt.figure(figsize=(16,10))
		ax1 = plt.axes()

		x_lim = [0.0, 35.0]			# IWV in kg m-2

		le_hist = ax1.hist(data, bins=np.arange(x_lim[0], x_lim[1]+0.01, 1), 
					weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))

		ax1.text(0.98, 0.96, "Min = %.2f\n Max = %.2f\n Mean = %.2f\n Median = %.2f\n N = %i"%(np.nanmin(data),
					np.nanmax(data), np.nanmean(data), np.nanmedian(data), len(data)), ha='right', va='top', 
					transform=ax1.transAxes, fontsize=fs-2)

		ax1.set_xlim(left=x_lim[0], right=x_lim[1])

		ax1.minorticks_on()
		ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1.set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
		ax1.set_ylabel("Frequency occurrence", fontsize = fs)

		ax1.tick_params(axis='both', labelsize=fs-2)


		return fig1, ax1


	# extract samples from given coordinates: find min distance and save into new array
	IWV_samples = get_samples(ERA5_DS, chosen_points, lat_tol, lon_tol, sample_var='tcwv')
	# then visualize: plot histogram for each sel point and total
	for idx, sel_point in enumerate(chosen_points):
		fig1, ax1 = hist_plot(IWV_samples[idx,:])
		ax1.text(0.02, 1.02, f"{sel_point[0]:02} deg N, {sel_point[1]:02} deg E", ha='left', va='bottom', 
					transform=ax1.transAxes, fontsize=fs)
		
		if save_figures:
			name_base = f"IWV_histogram_idx{idx:02}"
			fig1.savefig(path_plots + name_base + ".png", dpi=400)
		else:
			plt.show()
			plt.close()


	# all points in one histogram:
	fig1, ax1 = hist_plot(IWV_samples.flatten())
	ax1.text(0.02, 1.02, "Total", ha='left', va='bottom', 
					transform=ax1.transAxes, fontsize=fs)
		
	if save_figures:
		name_base = "IWV_histogram_total"
		fig1.savefig(path_plots + name_base + ".png", dpi=400)
	else:
		plt.show()
		plt.close()


if IWV_box_plot:

	def make_boxplot_great_again(bp, col):	# change color and set linewidth to 1.5
		plt.setp(bp['boxes'], color=col, linewidth=1.5)
		plt.setp(bp['whiskers'], color=col, linewidth=1.5)
		plt.setp(bp['caps'], color=col, linewidth=1.5)
		plt.setp(bp['medians'], color=col, linewidth=1.5)

	# extract samples from given coordinates: find min distance and save into new array
	IWV_samples = get_samples(ERA5_DS, chosen_points, lat_tol, lon_tol, sample_var='tcwv')


	# box plot showing IWV variability for each chosen point:
	fig1 = plt.figure(figsize=(10,10))
	ax1 = plt.axes()


	ylim = [0, 25]		# axis limits for IWV plot in kg m^-2
	labels_bp = [f"{sel_point[0]:.1f} N\n{sel_point[1]:.1f} E" for sel_point in chosen_points]

	bp_iwv = ax1.boxplot(IWV_samples.T, sym='ko', widths=0.5)
	make_boxplot_great_again(bp_iwv, col=(0,0,0))

	# set axis limits:
	ax1.set_ylim(bottom=ylim[0], top=ylim[1])

	# set x ticks and tick labels:
	ax1.xaxis.set_ticks(range(1, len(chosen_points)+1))
	ax1.xaxis.set_ticklabels(labels_bp)

	# further settings:
	ax1.tick_params(axis='y', labelsize=fs_small)
	ax1.tick_params(axis='x', labelsize=fs_dwarf-2)
	ax1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	ax1.set_axisbelow(True)

	# set axis labels:
	ax1.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	ax1.set_xlabel("Selected grid points", fontsize=fs)


	if save_figures:
		name_base = "IWV_variability_boxplot_selected_points"
		fig1.savefig(path_plots + name_base + ".png", dpi=400)
	else:
		plt.show()