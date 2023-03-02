import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import pdb
import glob
import os

import sys
import matplotlib as mpl
sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')
from data_tools import numpydatetime64_to_epochtime
from my_classes_eurec4a import *


"""
	This script is used to create simple cloud radar reflectivity profile plots.
	- Import data
	- Find wanted profile(s)
	- Plot
"""


# Settings:
save_figures = True					# if True, figures will be saved
find_what = "light_drizzle"			# indicates which profiles to look for; 
									# options: "heavy_drizzle", "no_drizzle", "light_drizzle"


dates = {'all': ['20200119',
				# '20200122', # no radar
				'20200124',
				'20200126',
				'20200128',
				'20200130',
				'20200131',
				'20200202',
				'20200205',
				'20200207',
				'20200209',
				'20200211',
				'20200213',
				#'20200215',
				'20200218'],
		'ice': 	[
				'20200119', 
				'20200209'
				],
		'liq':	[
				'20200124',
				'20200202'
				]}

dates = dates['all']		# default selection


# Paths:
unified_version = 'v0.9'

path_unified = f"/data/obs/campaigns/eurec4a/HALO/unified/{unified_version}/"
path_plots = "/net/blanc/awalbroe/Plots/EUREC4A/dBZ_profile/"


# check if plot path exists:
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)


for hh, which_date in enumerate(['20200202']):
	print(which_date)


	# Import HALO HAMP, lidar, and cloud mask data:
	HAMP_radar = radar(path_unified, version=unified_version, which_date=which_date, cut_low_altitude=False)


	# Select profiles:
	if find_what == 'heavy_drizzle':
		# >10 indices with dBZ > 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask = HAMP_radar.dBZ > 15
		time_found = np.where(np.count_nonzero(search_mask[:,height_idx], axis=1) > 10)[0]

	elif find_what == 'light_drizzle':
		# >10 indices with dBZ >= -15 but <= 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask = ((HAMP_radar.dBZ >= -15) & (HAMP_radar.dBZ <= 15))
		time_found = np.where(np.count_nonzero(search_mask[:, height_idx], axis=1) > 10)[0]

	elif find_what == 'no_drizzle':
		# no drizzle threshold of -15 was chosen because of Fig. 1 and 6 in Khain et al. 2008,
		# Fig. 9 of Fox and Illingworth 1997, an because of Liu et al. 2008.
		# And dBZ must be < -38 in the lowest 300 m
		height_idx = np.where(HAMP_radar.height < 300)[0]
		search_mask1 = HAMP_radar.dBZ >= -15
		search_mask2 = HAMP_radar.dBZ[:,height_idx] >= -38
		time_found = np.where((np.count_nonzero(search_mask1, axis=1) == 0) &
								(np.count_nonzero(search_mask2, axis=1) == 0))[0]


	# Plot:
	fs = 15
	fs_small = fs-2

	y_lim_height = [0, 4000]
	if find_what in ['heavy_drizzle', 'light_drizzle']:
		x_lim = [-40, 45]
	elif find_what == 'no_drizzle':
		x_lim = [-40, -10]

	mpl.rcParams.update({'xtick.labelsize': fs_small, 'ytick.labelsize': fs_small, 'axes.grid.which': 'major', 
							'axes.grid': True, 'axes.grid.axis': 'both', 'grid.color': (0.5,0.5,0.5),
							'axes.labelsize': fs})

	fig1 = plt.figure(figsize=(8,14))

	ax_cr = plt.axes()			# cloud radar dBZ

	
	# Radar refl plot: 
	for k, tt in enumerate(time_found):
		ax_cr.plot(HAMP_radar.dBZ[tt,:], HAMP_radar.height, linewidth=1.2)
	
	# ax_cr.plot(np.nanmean(HAMP_radar.dBZ[time_found,:], axis=0), HAMP_radar.height, linewidth=1.2, color=(0,0,0))


	# Set axis labels and titles:
	ax_cr.set_ylabel("Height (m)", fontsize=fs, labelpad=1.0)
	ax_cr.set_xlabel("Radar refl. factor (dBZ)", fontsize=fs, labelpad=1.0)
	ax_cr.set_title("HAMP cloud radar reflectivity - %s-%s-%s\n %s"%(which_date[:4], which_date[4:6], which_date[6:],
						find_what), fontsize=fs, pad=0.1)

	# Set axis limits:
	ax_cr.set_ylim(bottom=y_lim_height[0], top=y_lim_height[1])
	ax_cr.set_xlim(left=x_lim[0], right=x_lim[1])

	# handle some ticks and tick labels:
	ax_cr.minorticks_on()


	if save_figures:
		fig1.savefig(path_plots + f"HAMP_cloud_radar_dBZ_profile_{find_what}_EUREC4A_{which_date}", dpi=400)

	else:
		plt.show()

	plt.close()


print("Done....")