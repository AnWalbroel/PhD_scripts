import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import glob
import gc

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
import pdb


"""
	Small script to visualize HATPRO's position for a given date range.
	- import HKD data
	- cut away unneeded stuff
	- visualize
"""


# paths:
path_data = "/mnt/e/HATPRO/Y2022/"
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/regression/"


# settings:
set_dict = {'save_figures': True,
			'date0': "2022-07-15",			# start date of data
			'date1': "2022-07-19",			# end date of data
			}


# load TB data: concat daily files: run through each day and concat all files to
# a list:
nowdate = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
startdate = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
enddate = dt.datetime.strptime(set_dict['date1'], "%Y-%m-%d")
n_sec = (enddate - nowdate).days*86400
n_freq = 14
files = list()
t_idx = 0
while nowdate <= enddate:
	# run through folder structure:
	nd_month = nowdate.month
	nd_day = nowdate.day
	folder_date = path_data + f"M{nd_month:02}/D{nd_day:02}/"

	# look for netCDF files... several file conventions are of interest:
	nc_files = sorted(glob.glob(folder_date + f"{str(nowdate.year)[-2:]}{nd_month:02}{nd_day:02}.HKD.NC"))
	ele_90_files = sorted(glob.glob(folder_date + f"ELE90_{str(nowdate.year)[-2:]}{nd_month:02}{nd_day:02}.HKD.NC"))
	if len(ele_90_files) > 0:
		nc_files.append(ele_90_files[0])

	# loop through files append files:
	for file in nc_files: files.append(file)

	nowdate = nowdate + dt.timedelta(days=1)
HKD_DS = xr.open_mfdataset(files, combine='nested', concat_dim='time')


# truncate unneeded space:
startdate_wai = "2022-07-15T12:00:00"
enddate_wai = "2022-07-19T00:00:00"
HKD_DS = HKD_DS.sel(time=slice(startdate_wai, enddate_wai))


# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# map_settings:
lon_centre = 5.0
lat_centre = 80.0
lon_lat_extent = [0.0, 20.0, 78.0, 82.5]
sel_proj = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)


# some extra info for the plot:
station_coords = {'NYA': [11.929, 78.924, "Ny-Alesund"],
					'LYR': [15.632, 78.222, "Longyearbyen"]}

f1 = plt.figure(figsize=(10,7.5))
a1 = plt.axes(projection=sel_proj)
a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())

# add terrain background and some land marks:
# a1.add_image(cimgt.Stamen("terrain-background"), 4)
a1.coastlines(resolution="50m")
# a1.add_feature(cartopy.feature.BORDERS)
# a1.add_feature(cartopy.feature.OCEAN)
# a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85))
a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8))


# plot track:
a1.plot(HKD_DS.longitude.values, HKD_DS.latitude.values, linewidth=1.5,
		color=(0.65,0,0), label="HATPRO-GPS", transform=ccrs.PlateCarree())

# plot markers and labels:
PC_mpl_transf = ccrs.PlateCarree()._as_mpl_transform(a1)
text_transf = mpl.transforms.offset_copy(PC_mpl_transf, units='dots', x=marker_size*0.75,
										y=marker_size*0.75)
for key in station_coords.keys():
	a1.plot(station_coords[key][0], station_coords[key][1], color=(1,0,0),
			marker='.', markersize=marker_size, markeredgecolor=(0,0,0), 
			transform=ccrs.PlateCarree())

	a1.text(station_coords[key][0], station_coords[key][1], station_coords[key][2],
			ha='left', va='bottom', color=(0,0,0), fontsize=fs_small, transform=text_transf,
			bbox={'facecolor': (1.0,1.0,1.0,0.75), 'edgecolor': (0,0,0), 'boxstyle':'round'})

a1.text(0.02, 0.98, f"{set_dict['date0']} - {set_dict['date1']}", ha='left', va='top', color=(0,0,0),
			fontsize=fs_small, bbox={'facecolor': (1.0, 1.0, 1.0, 0.8), 'edgecolor': (0,0,0), 
			'boxstyle': 'round'}, transform=a1.transAxes)

# legend and colorbar(s):
lh, ll = a1.get_legend_handles_labels()
a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.00, 0.94), fontsize=fs_small,
			framealpha=0.0, facecolor=(1.0,1.0,1.0,0.75), edgecolor=(0,0,0))


if set_dict['save_figures']:
	plot_name = "WALSEMA_HATPRO_GPScoords_WAI"
	f1.savefig(path_plots + plot_name + ".png", dpi=300, bbox_inches='tight')
else:
	plt.show()