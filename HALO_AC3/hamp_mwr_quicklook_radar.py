import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import sys
import pdb
import os
import datetime as dt
from halo_classes import BAHAMAS, radar

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')
from data_tools import numpydatetime64_to_datetime, compute_retrieval_statistics


# from pyLARDA.Transformations
def set_xticks_and_xlabels(ax, time_extend):
	"""
	From https://github.com/jroettenbacher/phd_base/blob/main/src/pylim/helpers.py :
	This function sets the ticks and labels of the x-axis (only when the x-axis is time in UTC).
	Options:
		- 	1 days > time_extend > 12 hours:	major ticks every 2 hours, minor ticks every 1 hour
		- 	12 hours > time_extend:				major ticks every 1 hour, minor ticks every 30 minutes
	Args:
		ax: axis in which the x-ticks and labels have to be set
		time_extend: time difference of t_end - t_start (format datetime.timedelta)
	Returns:
		ax - axis with new ticks and labels
    """

	if time_extend > dt.timedelta(days=1):
		pass
	elif dt.timedelta(days=1) >= time_extend > dt.timedelta(hours=12):
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
		ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
	elif dt.timedelta(hours=12) >= time_extend > dt.timedelta(hours=6):
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
		ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
	elif dt.timedelta(hours=6) >= time_extend > dt.timedelta(hours=3):
		ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
		ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,30)))
		ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,15)))
	elif dt.timedelta(hours=3) >= time_extend > dt.timedelta(hours=1):
		ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
		ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,15)))
		ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
	elif dt.timedelta(hours=1) >= time_extend:
		ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
		ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,10)))
		ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))

	return ax


###################################################################################################
###################################################################################################


"""
	Create quicklooks from raw HALO HAMP microwave radiometer measurements.
	TBs for each available frequency will be plotted.
	Time axis boundaries will be according to BAHAMAS data.
	- Load radar data
	- Load BAHAMAS data
	- Plot quicklook overview
"""


path_data = {'radar': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/",				# base path of HALO HAMP data
			'bahamas': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/"}	# BAHAMAS data
path_output = "/net/blanc/awalbroe/Plots/HALO_AC3_quicklooks/"
save_figures = True		# if True, saved to PDF at path_output
time_series = True		# simple TB time series


dates = [	# must be in yyyymmdd
			# "20220225",
			# "20220311",
			"20220312",
			"20220313",
			"20220314",
			"20220315",
			"20220316",
			"20220320",
			"20220321",
			"20220328",
			"20220329",
			"20220330",
			"20220401",
			"20220404",
			"20220407",
			"20220408",
			"20220410",
			"20220411",
			"20220412",
			]

# Dictionary translating Research Flight numbers and dates:
RF_dict = {
			'20220225': "RF00",
			"20220311": "RF01",
			"20220312": "RF02",
			"20220313": "RF03",
			"20220314": "RF04",
			"20220315": "RF05",
			"20220316": "RF06",
			"20220320": "RF07",
			"20220321": "RF08",
			"20220328": "RF09",
			"20220329": "RF10",
			"20220330": "RF11",
			"20220401": "RF12",
			"20220404": "RF13",
			"20220407": "RF14",
			"20220408": "RF15",
			"20220410": "RF16",
			"20220411": "RF17",
			"20220412": "RF18",
			}

for which_date in dates:

	if which_date in RF_dict.keys():
		RF_now = RF_dict[which_date]


	# import MWR data:
	HAMP_radar = radar(path_data['radar'], which_date)


	# import BAHAMAS data:
	bah = BAHAMAS(path_data['bahamas'], which_date, return_DS=True)


	# Interpolate BAHAMAS altitude on radar time axis:
	HAMP_radar.n_time = len(HAMP_radar.time)
	HAMP_radar.alt = np.interp(HAMP_radar.time, bah.time, bah.DS.IRS_ALT.values, left=np.nan, right=np.nan)
	HAMP_radar.height = np.full((HAMP_radar.n_time, len(HAMP_radar.range)), np.nan)
	for kk in range(HAMP_radar.n_time):
		HAMP_radar.height[kk,:] = HAMP_radar.alt[kk] - HAMP_radar.range

	# time extent:
	time_extent = numpydatetime64_to_datetime(bah.time[-1]) - numpydatetime64_to_datetime(bah.time[0])
	# time_0 = dt.datetime(2022,3,21,15,35,0)
	# time_1 = dt.datetime(2022,3,21,16,5,0)
	# time_extent = time_1 - time_0


	# Plotting: time x height plot

	fs = 14
	fs_small = fs - 2
	fs_dwarf = fs - 4
	marker_size = 15


	import locale
	dt_fmt = mdates.DateFormatter("%H:%M") # (e.g. "12:00")
	datetick_auto = True


	if time_series:

		# Plot:
		fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

		# ax lims:
		time_lims = [bah.time_npdt[0], bah.time_npdt[-1]]
		# # # time_lims = [np.datetime64("2022-03-21T15:35"), np.datetime64("2022-03-21T16:05")]
		y_lims = [0, 12000]


		# plotting:
		xx = HAMP_radar.height
		yy = np.repeat(np.array([HAMP_radar.time_npdt]), 509, axis=0).transpose()
		bounds = np.arange(-30,31,5)
		n_levels = len(bounds)
		cmap = mpl.cm.get_cmap('viridis', n_levels)
		# pdb.set_trace()
		radar_plot = ax.contourf(yy, xx, HAMP_radar.dBZ, cmap=cmap, levels=bounds, extend='both')

		# add figure identifier of subplots: a), b), ...
		ax.text(0.02, 0.95, f"HALO-AC3_HALO_hamp_mira_{which_date}_{RF_now}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax.transAxes)


		# legends or colorbars:
		cb1 = fig.colorbar(mappable=radar_plot, ax=ax, boundaries=bounds, use_gridspec=False, spacing='proportional',
							extend='both', orientation='vertical', pad=0)
		cb1.set_label(label="Reflectivity (dBZ)", fontsize=fs_small)
		cb1.ax.tick_params(labelsize=fs_small)


		# set axis limits:
		ax.set_xlim(left=time_lims[0], right=time_lims[1])
		ax.set_ylim(bottom=y_lims[0], top=y_lims[1])


		# set x ticks and tick labels:
		ax = set_xticks_and_xlabels(ax=ax, time_extend=time_extent)


		# x tick parameters:
		ax.tick_params(axis='x', labelsize=fs_small)

		# ytick parameters and grid and y labels:
		ax.tick_params(axis='y', labelsize=fs_small)
		ax.grid(which='major', axis='both', alpha=0.4)
		ax.set_ylabel("Height (m)", fontsize=fs)

		# set labels:
		ax.set_xlabel(f"Time (HH:MM) of {str(bah.time_npdt[round(len(bah.time)/2)])[:10]}", fontsize=fs)

		if not os.path.exists(path_output):
			os.makedirs(path_output)

		if save_figures:
			plot_name = f"HALO-AC3_HALO_hamp_radar_{which_date}_{RF_now}.png"
			fig.savefig(path_output + plot_name, dpi=400, bbox_inches='tight')
			print(f"Plot saved to {path_output + plot_name}.")
		else:
			plt.show()


print("Done....")