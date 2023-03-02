import numpy as np
import datetime as dt
# import pandas as pd
import copy
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import warnings
from import_data import *
from met_tools import *
from data_tools import *
import xarray as xr
# import sys
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


################################################################


# Paths:
path_radiosondes_old = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/"
path_radiosondes_new = "/data/testbed/datasets/MOSAiC/rs41/"
path_radiosondes_level_2 = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"
path_plots = "/net/blanc/awalbroe/Plots/MOSAiC_mwr_sondes_IWV/"
all_iwv_sondes_dahlke = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/all_iwv.txt"


# Select one of the following plot_options:		###################################################
# 0: Unfiltered: Each data point is plotted, outliers are not filtered!
# 1: Omit flagged values:  Each data point is plotted, outliers are left out.
# 2: Running mean and omit flagged values: Running mean over rm_window width, outliers filtered.
# 3: Master time axis: Data points are on a master time axis instead of each instrument 
#		having its own time axis -> must be used for IWV difference, outliers are always 
#		filtered on master time axis.
# 4: Master time axis with running mean: Data, to which a running mean is applied, are on master 
#		time axis.
plot_option = 0 		# for plot_option in range(0,5):			# default: plot_option = 0
considered_period = 'mosaic'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# 'user': user defined
plot_IWV_diff_sondeIWV_abshum_spechum = False		# IWV difference: sonde_IWV_abshum - sonde_IWV_spechum
plot_IWV_diff_sondeIWV_abshum_spechum_rel = False		# IWV relative difference: sonde_IWV_abshum - sonde_IWV_spechum
plot_IWV_diff_sonde_new_vs_old = True
plot_IWV_time_series_sonde_ps_vs_dahlke = False
plot_IWV_diff_sonde_dahlke_vs_new = False
plot_IWV_diff_sonde_dahlke_vs_new_rel = False
save_figures = False
which_retrievals = 'iwv'		# which data is to be imported: 'both' contains both integrated quantities like IWV
								# and LWP. Other options: 'prw' or 'clwvi' ('iwv' or 'lwp').


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2019-09-20", "2020-06-01"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"
if not date_start and not date_end: raise ValueError("Please specify a date range in yyyy-mm-dd format.")


if which_retrievals in ['iwv', 'prw', 'both']:
	# Load radiosonde data and compute IWV:
	sonde_dict_old = import_radiosonde_daterange(path_radiosondes_old, date_start, date_end, s_version='mossonde', verbose=1)
	sonde_dict_new = import_radiosonde_daterange(path_radiosondes_new, date_start, date_end, s_version='psYYMMDDwHH', verbose=1)
	sonde_dict_level_2 = import_radiosonde_daterange(path_radiosondes_level_2, date_start, date_end, s_version='level_2', verbose=1)
	# iwv_dahlke = import_IWV_sonde_txt(all_iwv_sondes_dahlke)
	n_sondes = len(sonde_dict_old['launch_time'])
	n_sondes_new = len(sonde_dict_new['launch_time'])
	n_sondes_level_2 = len(sonde_dict_level_2['launch_time'])
	sonde_dict_old['iwv'] = np.full((n_sondes,), np.nan)
	sonde_dict_new['iwv'] = np.full((n_sondes_new,), np.nan)
	# for k in range(n_sondes): sonde_dict_old['iwv_a'][k] = compute_IWV(sonde_dict_old['rho_v'][k,:], sonde_dict_old['geopheight'][k,:])
	for k in range(n_sondes): sonde_dict_old['iwv'][k] = compute_IWV_q(sonde_dict_old['q'][k,:], sonde_dict_old['pres'][k,:])
	for k in range(n_sondes_new): sonde_dict_new['iwv'][k] = compute_IWV_q(sonde_dict_new['q'][k,:], sonde_dict_new['pres'][k,:])
	sonde_dict_old['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict_old['launch_time']])
	sonde_dict_new['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict_new['launch_time']])
	sonde_dict_level_2['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict_level_2['launch_time']])

# # compute difference of sondes:
# # old sonde launch time must be within 30 minutes of new sonde launch time:
# sonde_iwv_diff_new_old = np.full((len(sonde_dict_new['launch_time']),), np.nan)
# for k, lt_new in enumerate(sonde_dict_new['launch_time']):
	# thisisit = np.argwhere(np.abs(lt_new - sonde_dict_old['launch_time']) <= 1800).flatten()
	# if thisisit.size > 0:
		# sonde_iwv_diff_new_old[k] = sonde_dict_new['iwv'][k] - sonde_dict_old['iwv'][thisisit]

# compute difference of sondes:
# psYYMMDD.wHH sonde launch time must be within 30 minutes of level 2 sonde launch time:
sonde_iwv_diff_level_2_old = np.full((len(sonde_dict_level_2['launch_time']),), np.nan)
for k, lt_l2 in enumerate(sonde_dict_level_2['launch_time']):
	thisisit = np.argwhere(np.abs(lt_l2 - sonde_dict_old['launch_time']) <= 1800).flatten()
	if thisisit.size > 0:
		sonde_iwv_diff_level_2_old[k] = sonde_dict_level_2['iwv'][k] - sonde_dict_old['iwv'][thisisit]

# # remove Dahlke's sondes that haven't reached 10 km altitude or above:
# not_failed_sondes = np.argwhere(iwv_dahlke['balloon_burst_alt'] > 10000).flatten()
# failed_sondes = np.argwhere(iwv_dahlke['balloon_burst_alt'] <= 10000).flatten()
# iwv_dahlke['iwv'][failed_sondes] = np.nan

# # compute difference of sondes:
# # new sonde launch time must be within 30 minutes of iwv_dahlke launch time:
# sonde_iwv_diff_new_dahlke = np.full((len(iwv_dahlke['time']),), np.nan)
# psyymmddwhh_iwv = np.full((len(iwv_dahlke['time']),), np.nan)
# for k, lt_new in enumerate(iwv_dahlke['time']):
	# thisisit = np.argwhere(np.abs(lt_new - sonde_dict_new['launch_time']) <= 900).flatten()
	# if (thisisit.size > 0) and (k in not_failed_sondes):
		# sonde_iwv_diff_new_dahlke[k] = iwv_dahlke['iwv'][k] - sonde_dict_new['iwv'][thisisit]
		# psyymmddwhh_iwv[k] = sonde_dict_new['iwv'][thisisit]

# convert date_end and date_start to datetime:
date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")


# MOSAiC Legs:
MOSAiC_legs = {'leg1': [dt.datetime(2019,9,20), dt.datetime(2019,12,13)],
				'leg2': [dt.datetime(2019,12,13), dt.datetime(2020,2,24)],
				'leg3': [dt.datetime(2020,2,24), dt.datetime(2020,6,4)],
				'leg4': [dt.datetime(2020,6,4), dt.datetime(2020,8,12)],
				'leg5': [dt.datetime(2020,8,12), dt.datetime(2020,10,12)]}


dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
datetick_auto = False
fs = 19		# fontsize

# create x_ticks depending on the date range: roughly 20 x_ticks are planned
# round off to days if number of days > 15:
date_range_delta = (date_range_end - date_range_start)
if date_range_delta < dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=6)
	dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
elif date_range_delta < dt.timedelta(days=11) and date_range_delta >= dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=12)
	dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
elif date_range_delta >= dt.timedelta(days=11) and date_range_delta < dt.timedelta(21):
	x_tick_delta = dt.timedelta(days=1)
else:
	x_tick_delta = dt.timedelta(days=(date_range_delta/20).days)

x_ticks_dt = mdates.drange(date_range_start, date_range_end, x_tick_delta)	# alternative if the xticklabel is centered
# x_ticks_dt = mdates.drange(date_range_start + x_tick_delta, date_range_end + x_tick_delta, x_tick_delta)

# MOSAiC Legs text locations in plot
MOSAiC_Legs_Text = ["Leg 1", "Leg 2", "Leg 3", "Leg 4", "Leg 5"]
MLT_x_pos = []
for key in MOSAiC_legs.keys():
	if MOSAiC_legs[key][0] < date_range_start and MOSAiC_legs[key][1] <= date_range_end:
		MLT_x_pos.append(date_range_start + 0.5*(MOSAiC_legs[key][1] - date_range_start))
	elif MOSAiC_legs[key][0] >= date_range_start and MOSAiC_legs[key][1] > date_range_end:
		MLT_x_pos.append(MOSAiC_legs[key][0] + 0.5*(date_range_end - MOSAiC_legs[key][0]))
	else:
		MLT_x_pos.append(MOSAiC_legs[key][0] + 0.5*(MOSAiC_legs[key][1] - MOSAiC_legs[key][0]))



########## Plotting ##########

if plot_IWV_diff_sondeIWV_abshum_spechum:
	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig2, ax1 = plt.subplots(1,2)
	fig2.set_size_inches(22,10)

	iwv_axlim = [-0.02, 0.02]		# axis limits for IWV diff plot in kg m^-2
		

	SONDE_IWV_plot = ax1[0].plot(sonde_dict_old['launch_time_dt'], sonde_dict_old['iwv_a'] - sonde_dict_old['iwv_q'], linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='IWV(abshum - spechum)')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig2.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_diff_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_abshum_minus_spechum_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_abshum_minus_spechum_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig2.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_diff_sondeIWV_abshum_spechum_rel:
	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig3, ax1 = plt.subplots(1,2)
	fig3.set_size_inches(22,10)

	iwv_axlim = [-0.2, 0.2]		# axis limits for IWV diff plot in kg m^-2
		

	SONDE_IWV_plot = ax1[0].plot(sonde_dict_old['launch_time_dt'], 100*(sonde_dict_old['iwv_a'] - sonde_dict_old['iwv_q'])/(0.5*(sonde_dict_old['iwv_a'] + sonde_dict_old['iwv_q'])),
									linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='relative IWV(abshum - spechum)')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV} \, / \, \overline{\mathrm{IWV}}$ (%)", fontsize=fs)

	if datetick_auto:
		fig3.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_rel_diff_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_abshum_minus_spechum_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_abshum_minus_spechum_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		# plt.show()
		fig3.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_diff_sonde_new_vs_old:
	fig4, ax1 = plt.subplots(1,2)
	fig4.set_size_inches(22,10)

	iwv_axlim = [-5, 5]		# axis limits for IWV diff plot in kg m^-2
	# SONDE_IWV_plot = ax1[0].plot(sonde_dict_new['launch_time_dt'], sonde_iwv_diff_new_old, linestyle='none', marker='.', linewidth=0.5,
									# markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									# markeredgewidth=0.5, label='IWV(new - old)')

	SONDE_IWV_plot = ax1[0].plot(sonde_dict_level_2['launch_time_dt'], sonde_iwv_diff_level_2_old, linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='IWV(level_2 - mossonde)')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig4.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_diff_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_new_vs_old_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_new_vs_old_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		plt.show()
		fig4.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_time_series_sonde_ps_vs_dahlke:
	fig5, ax1 = plt.subplots(1,2)
	fig5.set_size_inches(22,10)

	iwv_axlim = [0, 32]		# axis limits for IWV diff plot in kg m^-2
	SONDE_IWV_plot = ax1[0].plot(sonde_dict_new['launch_time_dt'], sonde_dict_new['iwv'], linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='IWV psYYMMDD.wHH')
	SONDE_IWV_plot = ax1[0].plot(iwv_dahlke['datetime'][not_failed_sondes], iwv_dahlke['iwv'][not_failed_sondes], linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0.75,0,0), markeredgecolor=(0.75,0,0), 
									markeredgewidth=0.5, label='IWV Dahlke')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$ \mathrm{IWV}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig5.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_new_vs_dahlke_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_new_vs_dahlke_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		plt.show()
		fig5.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_diff_sonde_dahlke_vs_new:
	fig6, ax1 = plt.subplots(1,2)
	fig6.set_size_inches(22,10)

	iwv_axlim = [-3, 3]		# axis limits for IWV diff plot in kg m^-2
	SONDE_IWV_plot = ax1[0].plot(iwv_dahlke['datetime'], sonde_iwv_diff_new_dahlke, linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='IWV(Dahlke - psYYMMDD.wHH)')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if datetick_auto:
		fig6.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_diff_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_dahlke_vs_new_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_dahlke_vs_new_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		plt.show()
		fig6.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()


if plot_IWV_diff_sonde_dahlke_vs_new_rel:
	fig7, ax1 = plt.subplots(1,2)
	fig7.set_size_inches(22,10)

	iwv_axlim = [-6, 6]		# axis limits for IWV diff plot in kg m^-2
	SONDE_IWV_plot = ax1[0].plot(iwv_dahlke['datetime'], 100*sonde_iwv_diff_new_dahlke / (0.5*(psyymmddwhh_iwv + iwv_dahlke['iwv'])),
									linestyle='none', marker='.', linewidth=0.5,
									markersize=5.5, markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), 
									markeredgewidth=0.5, label='IWV(Dahlke - psYYMMDD.wHH)')

	auxplot = ax1[0].plot([date_range_start, date_range_end], [0, 0], color=(0,0,0), linewidth=0.75)

	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("$\Delta \mathrm{IWV} \, / \, \overline{\mathrm{IWV}}$ (%)", fontsize=fs)

	if datetick_auto:
		fig7.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-2, labelrotation=55)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')

	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])


	if save_figures:
		iwv_name_base = "IWV_rel_diff_time_series"
		if considered_period != 'user':
			iwv_name_suffix_def = "_sonde_dahlke_vs_new_" + considered_period
		else:
			iwv_name_suffix_def = "_sonde_dahlke_vs_new_" + date_start.replace("-","") + "-" + date_end.replace("-","")
		plt.show()
		fig7.savefig(path_plots + iwv_name_base + iwv_name_suffix_def + ".png", dpi=400)
	else:
		plt.show()

pdb.set_trace()
