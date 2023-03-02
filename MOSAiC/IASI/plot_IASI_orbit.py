import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'monospace'})
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopy
import geopy.distance

import glob
import gc
import os

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import numpydatetime64_to_datetime
from import_data import import_iasi_nc, import_PS_mastertrack, import_radiosonde_daterange
import pdb


"""
	Visualise IASI orbit and non-profile data as map plot for a certain time step. Additionally, 
	Polarstern IWV and position are shown. 
	- import IASI, Polarstern track and radiosonde data
	- find correct time stamps
	- visualise orbit
"""


# paths:
path_data = {	'iasi': "/mnt/d/heavy_data/IASI/",		# subfolders may exist
				'ps_track': "/mnt/d/heavy_data/polarstern_track/",
				'radiosonde': "/mnt/d/heavy_data/MOSAiC_radiosondes/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/mosaic_iasi/"

# additional settings:
time_stamp_array = np.arange(np.datetime64("2020-04-19T00:00:00"), np.datetime64("2020-04-21T01:00:00"),
								np.timedelta64(6, "h"))
for ts in time_stamp_array:
	set_dict = {'time_stamp': ts,		# need IASI orbit around +/- 12 h around this time
				'save_figures': True}
	set_dict['time0'] = set_dict['time_stamp'] - np.timedelta64(3, "h")
	set_dict['time1'] = set_dict['time_stamp'] + np.timedelta64(3, "h")

	path_plots_dir = os.path.dirname(path_plots)
	if not os.path.exists(path_plots_dir):
		os.makedirs(path_plots_dir)


	# import polarstern track:
	files = sorted(glob.glob(path_data['ps_track'] + "PS122_3_link-to-mastertrack_V2.nc"))
	PS_DS = import_PS_mastertrack(files, return_DS=True)
	PS_DS = PS_DS.sel(time=slice(set_dict['time0'], set_dict['time1']))


	# Polarstern radiosonde IWV:
	rs_date_range = [f"{str(set_dict['time_stamp'].astype('datetime64[D]') - np.timedelta64(1, 'D'))}",
						f"{str(set_dict['time_stamp'].astype('datetime64[D]') + np.timedelta64(1, 'D'))}"]
	sonde_dict = import_radiosonde_daterange(path_data['radiosonde'], rs_date_range[0], rs_date_range[1],
											s_version='level_2', remove_failed=True, verbose=0)
	n_sondes = len(sonde_dict['launch_time'])
	sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype("datetime64[s]")


	# find closest radiosonde launch to the selected time stamp:
	closest_sonde_idx = np.argmin(np.abs(sonde_dict['launch_time_npdt'] - set_dict['time_stamp']))


	# find right IASI orbit file:
	subfolders = sorted(glob.glob(path_data['iasi'] + "*"))
	kk = 0		# counts the plots
	for subfolder in subfolders:
		# loop through subfolders
		files = sorted(glob.glob(subfolder + "/*.nc"))
		for file in files:
			IASI_DS = import_iasi_nc([file])

			if not np.any((IASI_DS.record_stop_time.values >= set_dict['time0']) & (IASI_DS.record_start_time.values <= set_dict['time1'])):
				del IASI_DS
				continue
			else:
				print(f"IASI orbit start time: {IASI_DS.attrs['start_sensing_data_time']}")
				IASI_DS = IASI_DS.isel(along_track=(np.where((IASI_DS.record_stop_time.values >= set_dict['time0']) & (IASI_DS.record_start_time.values <= set_dict['time1']))[0]))


				# visualise:
				fs = 20
				fs_small = fs - 2
				fs_dwarf = fs - 4
				marker_size = 15
				c_orbit = (0.71,0.596,0.388)


				# map_settings:
				# lon_centre = 0.0
				# lat_centre = 80.0
				# lon_lat_extent = [-30.0, 30.0, 70.0, 90.0]
				# sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)
				# lon_centre = 0.0
				# lat_centre = 72.0
				# lon_lat_extent = [-120.0, 120.0, 70.0, 90.0]
				# sel_projection = ccrs.NearsidePerspective(central_longitude=lon_centre, central_latitude=lat_centre, satellite_height=3000000)
				lon_centre = 8.5
				lat_centre = 73.5
				lon_lat_extent = [-110.0, 110.0, 71.5, 90.0]
				sel_projection = ccrs.NearsidePerspective(central_longitude=lon_centre, central_latitude=lat_centre, satellite_height=1800000)

				station_coords = {'Ny-Alesund': [11.928611, 78.924444]}


				f1 = plt.figure(figsize=(11,9))
				a1 = plt.axes(projection=sel_projection)
				a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

				# add some land marks:
				a1.coastlines(resolution="50m", zorder=2)
				a1.add_feature(cartopy.feature.BORDERS, zorder=2)
				a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
				a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))

				# text transformer
				PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
				text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
															x=0, y=marker_size*3.5)


				# # attempting cartopy's transformations:
				# swath = ccrs.epsg(3413).transform_points(ccrs.PlateCarree(), IASI_DS['lon'].values.ravel(), IASI_DS['lat'].values.ravel())


				# # attempts to regrid data:
				# levels_0 = np.arange(0.0, 16.1, 0.5)
				# n_levels = len(levels_0)
				# cmap = mpl.cm.get_cmap('turbo', n_levels)
				# from scipy.interpolate import griddata
				# lon_iasi = IASI_DS.lon.values.ravel()
				# lat_iasi = IASI_DS.lat.values.ravel()
				# var_iasi = IASI_DS.iwv.values.ravel()
				# var_iasi[IASI_DS.flag_fgcheck.values.ravel() > 511] = np.nan		# exclude flagged data
				# # new grid:
				# lon_grid = np.arange(-120.0, 120.01, 0.1)
				# lat_grid = np.arange(65.0, 89.91, 0.1)
				# X, Y = np.meshgrid(lon_grid, lat_grid)
				# # grid data:
				# var_grid = griddata((lon_iasi, lat_iasi), var_iasi, (X, Y), method='linear')
				# contourf_0 = a1.contourf(X, Y, var_grid, cmap=cmap, levels=levels_0, extend='max', 
										# transform=ccrs.PlateCarree())

				# trying to plot with scatter (works, but doesn't look nice)
				levels_0 = np.arange(0.0, 16.1, 0.5)
				n_levels = len(levels_0)
				cmap = mpl.cm.get_cmap('turbo', n_levels-1)
				var_plot = IASI_DS.iwv.values
				var_plot[var_plot > 600] = np.nan
				var_plot_lon = IASI_DS.lon.values
				var_plot_lat = IASI_DS.lat.values
				contourf_0 = a1.scatter(x=var_plot_lon.ravel(), y=var_plot_lat.ravel(), c=var_plot.ravel(),
										s=35, vmin=0.0, vmax=16.0, cmap=cmap, transform=ccrs.PlateCarree())


				# plot the IASI orbit track:
				a1.plot(IASI_DS.lon.values[:,60], IASI_DS.lat.values[:,60], color=c_orbit, linewidth=2.0, 
						transform=ccrs.PlateCarree(), label="IASI orbit")

				some_acrosses = np.arange(0, len(IASI_DS.along_track), 20)
				for k in some_acrosses:
					a1.plot(IASI_DS.lon.values[k,::4], IASI_DS.lat.values[k,::4], color=c_orbit, linewidth=1.25,
						transform=ccrs.PlateCarree())
					# iasi time:
					iasi_time_str = dt.datetime.utcfromtimestamp(IASI_DS.record_stop_time.values[k].astype("datetime64[s]").astype(np.int32)).strftime("%H:%M:%S")
					if (IASI_DS.lat.values[k,-1] > 86) or ((IASI_DS.lat.values[k,-1] < 85) & (IASI_DS.lat.values[k,-1] > 75) & (IASI_DS.lon.values[k,-1] < 120) & (IASI_DS.lon.values[k,-1] > -120)):
						a1.text(IASI_DS.lon.values[k,-1], IASI_DS.lat.values[k,-1], iasi_time_str, 
							ha='left', va='bottom',
							color=c_orbit, fontsize=fs_dwarf-2, transform=PlateCarree_mpl_transformer, 
							zorder=10000.0)


				# plotting the search circle around Polarstern:
				PS_DS_sel = PS_DS.sel(time=(set_dict['time_stamp']), method='nearest')
				circ_centre = [float(PS_DS_sel.Latitude.values), float(PS_DS_sel.Longitude.values)] # lat lon of circle's centre

				# define and plot circle:
				circ_rad = 50.0		# radius in km
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
						zorder=10000.0, label='Search radius')


				# station and Polarstern markers:
				a1.plot(station_coords['Ny-Alesund'][0], station_coords['Ny-Alesund'][1], color=(1,1,1),
						marker='.', markersize=marker_size*1.25, markeredgecolor=(0,0,0), 
						transform=ccrs.PlateCarree(), zorder=10000.0, label="Ny-$\mathrm{\AA}$lesund")

				# Polarstern position:
				a1.plot(PS_DS_sel.Longitude.values, PS_DS_sel.Latitude.values, color=(1,1,1), marker='v', markersize=marker_size/1.5, 
						markeredgecolor=(0,0,0),transform=ccrs.PlateCarree(), zorder=10000.0, label='Polarstern')


				# some description:
				a1.text(0.02, 0.98, f"{str(set_dict['time_stamp'].astype('datetime64[h]')).replace('T',' ')} UTC", 
						ha='left', va='top', color=(0,0,0), fontsize=fs_dwarf, 
						bbox={'facecolor': (1,1,1,0.8), 'edgecolor': (0,0,0), 'boxstyle': 'round'},
						zorder=10000.0, transform=a1.transAxes)

				# IWV at Polarstern from radiosonde:
				iwv_ps_text = a1.text(PS_DS_sel.Longitude.values, PS_DS_sel.Latitude.values, 
										f"{sonde_dict['iwv'][closest_sonde_idx]:.1f}" + "$\,\mathrm{kg}\,\mathrm{m}^{-2}$", ha='center', va='bottom',
										color=(0,0,0), fontsize=fs_dwarf-2, transform=text_transform, 
										zorder=10000.0)


				# colorbar(s) and legends:
				lh, ll = a1.get_legend_handles_labels()
				lel = a1.legend(lh, ll, loc='center left', fontsize=fs_dwarf, markerscale=1.5, 
								frameon=True, facecolor=(1,1,1,0.8), edgecolor=(0,0,0))

				cb_var = f1.colorbar(mappable=contourf_0, ax=a1, extend='max', orientation='vertical', 
										fraction=0.06, pad=0.06, shrink=0.70)
				cb_var.set_label(label="IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_small)
				cb_var.ax.tick_params(labelsize=fs_dwarf)



				if set_dict['save_figures']:
					time_stamp_str = str(set_dict['time_stamp'].astype('datetime64[h]')).replace('-','').replace('T','')
					path_plots_f = path_plots + f"{time_stamp_str}Z/"
					plot_file = path_plots_f + f"MOSAiC_IASI_Polarstern_track_orbit_{IASI_DS.attrs['start_sensing_data_time']}_IWV_{time_stamp_str}Z.png"

					path_plots_dir = os.path.dirname(path_plots_f)
					if not os.path.exists(path_plots_dir):
						os.makedirs(path_plots_dir)

					f1.savefig(plot_file, dpi=300, bbox_inches='tight')
					print("Saved " + plot_file)
				else:
					plt.show()
					pdb.set_trace()

				kk += 1
				plt.close()
				gc.collect()

pdb.set_trace()