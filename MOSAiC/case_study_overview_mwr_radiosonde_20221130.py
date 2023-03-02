import xarray as xr
import numpy as np
import glob
import os
import datetime as dt
import pdb
from import_data import import_radiosonde_daterange
from data_tools import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl


"""
	This script will be used to generate an overview plot of a case study
	(i.e., moist air intrusion in April 2020 during MOSAiC expedition) visualized
	with MWR and radiosonde data. IWV, humidity profiles and temperature profiles
	will be plotted.
	- import data in the selected time window
	- eventually interpolate radiosonde data to a temporal resolution similar to MWR 
		(for the humidity and temperature profile plots)
	- plot
"""

def remove_vars_prw(DS):

	"""
	Preprocessing HATPRO data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of HATPRO data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'time_bnds', 
					'azi', 'ele', 'ele_ret', 'prw_offset', 
					'prw_off_zenith', 'prw_off_zenith_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	# reduce redundant and non-relevant values of prw_err:
	DS['prw_err'] = DS.prw_err[-1]

	return DS


def remove_vars_clwvi(DS):

	"""
	Preprocessing HATPRO data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of HATPRO data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'time_bnds', 
					'azi', 'ele', 'ele_ret', 'clwvi_offset', 
					'clwvi_off_zenith', 'clwvi_off_zenith_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	# reduce redundant and non-relevant values of prw_err:
	DS['clwvi_err'] = DS.clwvi_err[-1]

	return DS


def remove_vars_hua(DS):

	"""
	Preprocessing HATPRO data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of HATPRO data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'time_bnds', 
					'azi', 'ele', 'ele_ret', 'hua_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	# reduce redundant and non-relevant values of hua_err:
	DS['hua_err'] = DS.hua_err[:,-1]

	return DS


def remove_vars_ta(DS):

	"""
	Preprocessing HATPRO data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of HATPRO data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'time_bnds', 
					'azi', 'ele', 'ele_ret', 'ta_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	# reduce redundant and non-relevant values of ta_err:
	DS['ta_err'] = DS.ta_err[:,-1]

	return DS


def remove_vars_ta_bl(DS):

	"""
	Preprocessing HATPRO data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of HATPRO data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'time_bnds', 
					'azi', 'ele', 'ta_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	return DS


def remove_vars_mir(DS):

	"""
	Preprocessing MiRAC-P data: Removing undesired dimensions and variables.

	Parameters:
	-----------
	DS : xarray dataset
		Dataset of MiRAC-P data.
	"""

	useless_vars = ['lat', 'lon', 'zsl', 'azi', 'ele', 'prw_offset']

	# check if the variable names exist in DS:
	for idx, var in enumerate(useless_vars):
		if not var in DS:
			useless_vars.pop(idx)

	DS = DS.drop_vars(useless_vars)

	return DS


###################################################################################################
###################################################################################################


path_radiosondes = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"	# path of radiosonde nc files
path_hatpro = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/HATPRO_l2_v01/"		# path of hatpro derived products
path_mirac = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/outside_eez/MiRAC-P_l2_v01/"		# path of mirac-p derived products

path_plots = "/net/blanc/awalbroe/Plots/figures_data_paper/"			# output of plots


save_figures = True				# if true, figures will be saved to file
save_figures_eps = False		# if true, figures will be saved to a vector graphics file
with_titles = False		# if True, plots will have titles (False for publication plots)
considered_period = 'moist_air_intrusion'
radiosonde_version = 'level_2'


# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'moist_air_intrusion': ["2020-04-13", "2020-04-22"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-04-13", "2020-04-23"]}
date_start = daterange_options[considered_period][0]
date_end = daterange_options[considered_period][1]
date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

# extended range to display sonde values before first and after last launch within
# the selected period: For example, When date_start = '2020-04-13', the first sonde
# might have been launched around 2020-04-13 04:50:00 UTC. Therefore values before that
# would not be plotted. Therefore I extend the date range:
date_start_sonde = dt.datetime.strftime(dt.datetime.strptime(date_start, "%Y-%m-%d") - dt.timedelta(days=1), "%Y-%m-%d")
date_end_sonde = dt.datetime.strftime(dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1), "%Y-%m-%d")


# import data:

# Load radiosonde data in the date range:
sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start_sonde, date_end_sonde, s_version=radiosonde_version, 
										remove_failed=True, verbose=1)
n_sondes = len(sonde_dict['launch_time'])
sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype("datetime64[s]")

# HATPRO: find the right files, load IWV, humidity and temperature profiles
all_files_prw = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_prw_v01_*.nc"))
all_files_clwvi = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_clwvi_v01_*.nc"))
all_files_hua = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_hua_v01_*.nc"))
all_files_ta = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_ta_v01_*.nc"))
all_files_ta_bl = sorted(glob.glob(path_hatpro + "ioppol_tro_mwrBL00_l2_ta_v01_*.nc"))

# IWV:
files = find_files_daterange(all_files_prw, date_start_dt, date_end_dt, [-17,-9])
HATPRO_DS_prw = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_prw)
HATPRO_DS_prw['prw_err'] = float(HATPRO_DS_prw.prw_err[0])

# LWP:
files = find_files_daterange(all_files_clwvi, date_start_dt, date_end_dt, [-17,-9])
HATPRO_DS_clwvi = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_clwvi)
HATPRO_DS_clwvi['clwvi_err'] = float(HATPRO_DS_clwvi.clwvi_err[0])

# Abs hum profiles:
files = find_files_daterange(all_files_hua, date_start_dt, date_end_dt, [-17,-9])
HATPRO_DS_hua = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_hua)
HATPRO_DS_hua['hua_err'] = HATPRO_DS_hua.hua_err[0,:]

# Temperature profiles:
files = find_files_daterange(all_files_ta, date_start_dt, date_end_dt, [-17,-9])
HATPRO_DS_ta = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_ta)
HATPRO_DS_ta['ta_err'] = HATPRO_DS_ta.ta_err[0,:]

# Temperature profiles:
files = find_files_daterange(all_files_ta_bl, date_start_dt, date_end_dt, [-17,-9])
HATPRO_DS_ta_bl = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_ta_bl)
HATPRO_DS_ta_bl['ta_err'] = HATPRO_DS_ta_bl.ta_err[0,:]

# MiRAC-P:
all_files_mir = sorted(glob.glob(path_mirac + "MOSAiC_uoc_lhumpro-243-340_l2_prw_v01*.nc"))
files = find_files_daterange(all_files_mir, date_start_dt, date_end_dt, [-17,-9])
MIRAC_DS_prw = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=remove_vars_mir)


# Filter flagged values (flag > 0):
ok_idx = np.where((MIRAC_DS_prw.flag.values == 0) | (np.isnan(MIRAC_DS_prw.flag.values)))[0]
MIRAC_DS_prw = MIRAC_DS_prw.isel(time=ok_idx)


# Create merged retrieved product dataset:
RET_DS = xr.Dataset({	'ta': (['time', 'height'], HATPRO_DS_ta.ta.values),
						'hua': (['time', 'height'], HATPRO_DS_hua.hua.values),
						'prw': (['time'], HATPRO_DS_prw.prw.values),
						'clwvi': (['time'], HATPRO_DS_clwvi.clwvi.values),
						'ta_bl': (['time_bl', 'height'], HATPRO_DS_ta_bl.ta.values),
						'time_sec': (['time'], HATPRO_DS_prw.time.values.astype(np.float64)),
						'time_bl_sec': (['time_bl'], HATPRO_DS_ta_bl.time.values.astype(np.float64))},
					coords=	{'time': (['time'], HATPRO_DS_prw.time),
							'time_bl': (['time_bl'], HATPRO_DS_ta_bl.time),
							'height': (['height'], HATPRO_DS_ta.height)})


# Plot:
# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P
c_RS = (1,0.435,0)			# radiosondes
c_LWP = (0,0,0,0.55)
rho_v_cmap = mpl.cm.gist_earth
temp_cmap = mpl.cm.nipy_spectral

import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
dt_fmt = mdates.DateFormatter("%d %b") # (e.g. "Feb 23")
datetick_auto = False


# merge BL and zenith temperature profile:
# lowest 2000 m: BL only; 2000-2500m: linear transition from BL to zenith; >2500: zenith only:
# Leads to loss in temporal resolution of HATPRO: interpolated to BL scan time grid:
th_bl, th_tr = 2000, 2500		# tr: transition zone
idx_bl = np.where(RET_DS.height.values <= th_bl)[0][-1]
idx_tr = np.where((RET_DS.height.values > th_bl) & (RET_DS.height.values <= th_tr))[0]
pc_bl = (-1/(th_tr-th_bl))*RET_DS.height.values[idx_tr] + 1/(th_tr-th_bl)*th_tr 	# percentage of BL mode
pc_ze = 1 - pc_bl												# respective percentage of zenith mode during transition
ta_bl = RET_DS.ta_bl.interp(coords={'time_bl': RET_DS.time})
ta_combined = RET_DS.ta
ta_combined[:,:idx_bl+1] = ta_bl[:,:idx_bl+1]
ta_combined[:,idx_tr] = pc_bl*ta_bl[:,idx_tr] + pc_ze*RET_DS.ta[:,idx_tr]
RET_DS['ta_combined'] = xr.DataArray(ta_combined, coords={'time': RET_DS.time, 'height': RET_DS.height},
										dims=['time', 'height'])

##############
# reduce the arrays for plotting:
set_dict = {'date0': date_start_dt.strftime("%Y-%m-%d"),
			'date1': date_end_dt.strftime("%Y-%m-%d"),
			'save_figures': True}
set_dict['with_ip'] = True
startdate_wai = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
enddate_wai = dt.datetime.strptime(set_dict['date1'], "%Y-%m-%d")
startdate_wai_sec = datetime_to_epochtime(startdate_wai)
enddate_wai_sec = datetime_to_epochtime(enddate_wai)

RET_DS['ta_rm'] = (['time', 'height'], running_mean_time_2D(RET_DS.ta_combined.values, 300.0, RET_DS.time_sec.values, axis=0))

wai_idx = np.where((sonde_dict['launch_time'] >= startdate_wai_sec) & (sonde_dict['launch_time'] <= enddate_wai_sec))[0]
wai_idx = np.insert(wai_idx, 0, wai_idx[0]-1)		# add sonde before startdate; otherwise blank in figure
wai_idx = np.append(wai_idx, wai_idx[-1]+1)			# add sonde after enddate; otherwise blank in figure
sonde_dict['launch_time_npdt'] = sonde_dict['launch_time_npdt'][wai_idx]
sonde_dict['IWV'] = sonde_dict['iwv'][wai_idx]
sonde_dict['rho_v'] = sonde_dict['rho_v'][wai_idx,:]
sonde_dict['height'] = sonde_dict['height'][wai_idx,:]
sonde_dict['temp'] = sonde_dict['temp'][wai_idx,:]


dt_fmt = mdates.DateFormatter("%m-%d")
datetick_auto = False

# create x_ticks depending on the date range:
date_range_delta = (enddate_wai - startdate_wai)
if (date_range_delta < dt.timedelta(days=10)) & (date_range_delta >= dt.timedelta(days=3)):
	x_tick_delta = dt.timedelta(hours=12)
	dt_fmt = mdates.DateFormatter("%m-%d %HZ")
elif (date_range_delta < dt.timedelta(days=3)) & (date_range_delta >= dt.timedelta(days=2)):
	x_tick_delta = dt.timedelta(hours=6)
	dt_fmt = mdates.DateFormatter("%m-%d %HZ")
elif date_range_delta < dt.timedelta(days=2):
	x_tick_delta = dt.timedelta(hours=3)
	dt_fmt = mdates.DateFormatter("%m-%d %HZ")
else:
	x_tick_delta = dt.timedelta(days=3)
	dt_fmt = mdates.DateFormatter("%m-%d %HZ")


x_ticks_dt = mdates.drange(startdate_wai, enddate_wai + dt.timedelta(hours=1), x_tick_delta)

fig1 = plt.figure(figsize=(10,15))
ax_iwv = plt.subplot2grid((5,1), (0,0))			# IWV
ax_hua_rs = plt.subplot2grid((5,1), (1,0))		# radiosonde abs. hum. profiles
ax_hua_hat = plt.subplot2grid((5,1), (2,0))		# hatpro abs. hum. profiles
ax_ta_rs = plt.subplot2grid((5,1), (3,0))		# radiosonde temperature profiles
ax_ta_hat = plt.subplot2grid((5,1), (4,0))		# hatpro temperature profiles (zenith)


# ax lims:
iwv_lims = [0.0, 20.0]		# kg m-2
lwp_lims = [0.0, 800.0]		# g m-2
height_lims = [0, 8000]		# m
time_lims = [startdate_wai, enddate_wai]

rho_v_levels = np.arange(0.0, 5.51, 0.2)		# in g m-3
temp_levels = np.arange(200.0, 285.001, 2)		# in K
temp_contour_levels = np.arange(-70.0, 50.1, 10.0)		# in deg C

# colors:
rho_v_cmap = mpl.cm.get_cmap('gist_earth', len(rho_v_levels))
temp_cmap = mpl.cm.get_cmap('nipy_spectral', len(temp_levels))
temp_contour_cmap = np.full(temp_contour_levels.shape, "#000000")


# plot LWP but on IWV axis. Need to translate LWP values to IWV axis:
LI = (iwv_lims[1] - iwv_lims[0]) / (lwp_lims[1] - lwp_lims[0])
L0 = iwv_lims[1] - LI*lwp_lims[1]
LWP_on_IWV_axis = LI*RET_DS.clwvi.values*1000.0 + L0
ax_iwv.plot(RET_DS.time.values, LWP_on_IWV_axis, color=c_LWP, linewidth=1.0)


# plot IWV:
ax_iwv.plot(sonde_dict['launch_time_npdt'], sonde_dict['IWV'], linestyle='none', 
			marker='.', linewidth=0.5,
			markersize=marker_size, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
			markeredgewidth=0.5, label='Radiosonde')
filter_flag = np.where((HATPRO_DS_prw.flag.values == 0) | (np.isnan(HATPRO_DS_prw.flag.values)))[0]
ax_iwv.plot(MIRAC_DS_prw.time.values, MIRAC_DS_prw.prw.values, color=c_M, linewidth=1.2)
ax_iwv.plot(RET_DS.time.values[filter_flag], RET_DS.prw.values[filter_flag], color=c_H, linewidth=1.2)

# dummy lines for the legend (thicker lines)
ax_iwv.plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, label='HATPRO IWV')
ax_iwv.plot([np.nan, np.nan], [np.nan, np.nan], color=c_M, linewidth=2.0, label='MiRAC-P IWV')
ax_iwv.plot([np.nan, np.nan], [np.nan, np.nan], color=c_LWP, linewidth=2.0, label='LWP')


# LWP axis to read values:
ax_lwp = ax_iwv.twinx()
filter_flag = np.where((HATPRO_DS_clwvi.flag.values == 0) | (np.isnan(HATPRO_DS_clwvi.flag.values)))[0]
ax_lwp.plot(RET_DS.time.values[filter_flag], RET_DS.clwvi.values[filter_flag]*1000.0, color=(0,0,0,0.4), linewidth=1.0, zorder=-100.5)


if set_dict['with_ip']:
	# plot radiosonde humidity profile:
	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	rho_v_rs_curtain = ax_hua_rs.contourf(yv, xv, 1000*sonde_dict['rho_v'], levels=rho_v_levels,
										cmap=rho_v_cmap, extend='max')


	print("Plotting HATPRO humidity profile....")
	# plot hatpro hum profile:
	filter_flag = np.where((HATPRO_DS_hua.flag.values == 0) | (np.isnan(HATPRO_DS_hua.flag.values)))[0]
	xv, yv = np.meshgrid(RET_DS.height.values, RET_DS.time.values[filter_flag])
	rho_v_hat_curtain = ax_hua_hat.contourf(yv, xv, 1000*RET_DS.hua.values[filter_flag,:], levels=rho_v_levels,
										cmap=rho_v_cmap, extend='max')


	# plot radiosonde temperature profile:
	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	temp_rs_curtain = ax_ta_rs.contourf(yv, xv, sonde_dict['temp'], levels=temp_levels,
										cmap=temp_cmap, extend='both')

	# add black contour lines and contour labels:
	temp_rs_contour = ax_ta_rs.contour(yv, xv, sonde_dict['temp'] - 273.15, levels=temp_contour_levels,
											colors='black', linewidths=0.9, alpha=0.5)
	ax_ta_rs.clabel(temp_rs_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
					colors='black', inline_spacing=10, fontsize=fs_dwarf)


	print("Plotting HATPRO temperature profiles....")
	# plot hatpro zenith temperature profile:
	filter_flag = np.where((HATPRO_DS_ta.flag.values == 0) | (np.isnan(HATPRO_DS_ta.flag.values)))[0]
	xv, yv = np.meshgrid(RET_DS['height'], RET_DS['time'][filter_flag])
	temp_hat_curtain = ax_ta_hat.contourf(yv, xv, RET_DS['ta_rm'][filter_flag,:], levels=temp_levels,
											cmap=temp_cmap, extend='both')

	# add black contour lines of some temperatures: (only every 500th value to avoid clabel overlap)
	temp_hat_contour = ax_ta_hat.contour(yv[::500,:], xv[::500,:], RET_DS['ta_rm'].values[filter_flag[::500],:] - 273.15, levels=temp_contour_levels,
											colors='black', linewidths=0.9, alpha=0.5)
	ax_ta_hat.clabel(temp_hat_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
					colors='black', inline_spacing=12, fontsize=fs_dwarf)




# add figure identifier of subplots: a), b), ...
ax_iwv.text(0.02, 0.95, "a)", fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_iwv.transAxes)
ax_hua_rs.text(0.02, 0.95, "b) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax_hua_rs.transAxes)
ax_hua_hat.text(0.02, 0.95, "c) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax_hua_hat.transAxes)
ax_ta_rs.text(0.02, 0.95, "d) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_rs.transAxes)
ax_ta_hat.text(0.02, 0.95, "e) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_hat.transAxes)


# legends and colorbars:
lh, ll = ax_iwv.get_legend_handles_labels()
ax_iwv.legend(handles=lh, labels=ll, loc='upper right', fontsize=fs_small, ncol=4,
				framealpha=0.65, markerscale=1.5)

cb_hua_rs = fig1.colorbar(mappable=rho_v_rs_curtain, ax=ax_hua_rs, use_gridspec=True,
							orientation='vertical', extend='max', fraction=0.09, pad=0.01, shrink=0.9)
cb_hua_rs.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
cb_hua_rs.ax.tick_params(labelsize=fs_dwarf)

cb_hua_hat = fig1.colorbar(mappable=rho_v_hat_curtain, ax=ax_hua_hat, use_gridspec=True,
							orientation='vertical', extend='max', fraction=0.09, pad=0.01, shrink=0.9)
cb_hua_hat.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
cb_hua_hat.ax.tick_params(labelsize=fs_dwarf)

cb_ta_rs = fig1.colorbar(mappable=temp_rs_curtain, ax=ax_ta_rs, use_gridspec=True,
							orientation='vertical', extend='both', fraction=0.09, pad=0.01, shrink=0.9)
cb_ta_rs.set_label(label="T (K)", fontsize=fs_small)
cb_ta_rs.ax.tick_params(labelsize=fs_dwarf)

cb_ta_hat = fig1.colorbar(mappable=temp_hat_curtain, ax=ax_ta_hat, use_gridspec=True,
							orientation='vertical', extend='both', fraction=0.09, pad=0.01, shrink=0.9)
cb_ta_hat.set_label(label="T (K)", fontsize=fs_small)
cb_ta_hat.ax.tick_params(labelsize=fs_dwarf)


# set axis limits:
ax_iwv.set_xlim(left=time_lims[0], right=time_lims[1])
ax_hua_rs.set_xlim(left=time_lims[0], right=time_lims[1])
ax_hua_hat.set_xlim(left=time_lims[0], right=time_lims[1])
ax_ta_rs.set_xlim(left=time_lims[0], right=time_lims[1])
ax_ta_hat.set_xlim(left=time_lims[0], right=time_lims[1])

ax_iwv.set_ylim(bottom=iwv_lims[0], top=iwv_lims[1])
ax_lwp.set_ylim(bottom=lwp_lims[0], top=lwp_lims[1])
ax_hua_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_hua_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_ta_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_ta_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])


# set x ticks and tick labels:
ax_iwv.xaxis.set_ticks(x_ticks_dt)
ax_iwv.xaxis.set_ticklabels([])
ax_hua_rs.xaxis.set_ticks(x_ticks_dt)
ax_hua_rs.xaxis.set_ticklabels([])
ax_hua_hat.xaxis.set_ticks(x_ticks_dt)
ax_hua_hat.xaxis.set_ticklabels([])
ax_ta_rs.xaxis.set_ticks(x_ticks_dt)
ax_ta_rs.xaxis.set_ticklabels([])
# ax_ta_rs.xaxis.set_major_formatter(dt_fmt)			#################
ax_ta_hat.xaxis.set_ticks(x_ticks_dt)
ax_ta_hat.xaxis.set_major_formatter(dt_fmt)


# set y ticks and tick labels:
if ax_hua_rs.get_yticks()[-1] == height_lims[1]:
	ax_hua_rs.yaxis.set_ticks(ax_hua_rs.get_yticks()[:-1])			# remove top tick
if ax_hua_hat.get_yticks()[-1] == height_lims[1]:
	ax_hua_hat.yaxis.set_ticks(ax_hua_hat.get_yticks()[:-1])			# remove top tick
if ax_ta_rs.get_yticks()[-1] == height_lims[1]:
	ax_ta_rs.yaxis.set_ticks(ax_ta_rs.get_yticks()[:-1])			# remove top tick
if ax_ta_hat.get_yticks()[-1] == height_lims[1]:
	ax_ta_hat.yaxis.set_ticks(ax_ta_hat.get_yticks()[:-1])			# remove top tick


# x tick parameters:
ax_ta_hat.tick_params(axis='x', labelsize=fs_small, labelrotation=90)


# y tick parameters:
ax_iwv.tick_params(axis='y', labelsize=fs_small)
ax_lwp.tick_params(axis='y', labelsize=fs_small, labelcolor=c_LWP)
ax_hua_rs.tick_params(axis='y', labelsize=fs_small)
ax_hua_hat.tick_params(axis='y', labelsize=fs_small)
ax_ta_rs.tick_params(axis='y', labelsize=fs_small)
ax_ta_hat.tick_params(axis='y', labelsize=fs_small)


# grid:
ax_iwv.grid(which='major', axis='both', alpha=0.4)
ax_hua_rs.grid(which='major', axis='both', alpha=0.4)
ax_hua_hat.grid(which='major', axis='both', alpha=0.4)
ax_ta_rs.grid(which='major', axis='both', alpha=0.4)
ax_ta_hat.grid(which='major', axis='both', alpha=0.4)


# set labels:
ax_iwv.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
ax_lwp.set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs, color=c_LWP)
ax_hua_rs.set_ylabel("Height (m)", fontsize=fs)
ax_hua_hat.set_ylabel("Height (m)", fontsize=fs)
ax_ta_rs.set_ylabel("Height (m)", fontsize=fs)
ax_ta_hat.set_ylabel("Height (m)", fontsize=fs)

ax_ta_hat.set_xlabel(f"{startdate_wai.year}", fontsize=fs)

# if with_titles:
	# ax_iwv.set_title("IWV (a) and profiles of humidity (b,c) and temperature (d-e) from\nHATPRO and radiosondes", fontsize=fs)


# Limit axis spacing:
plt.subplots_adjust(hspace=0.0)			# removes space between subplots

# Adjust axis width and position:
ax_iwv_pos = ax_iwv.get_position().bounds
ax_iwv.set_position([ax_iwv_pos[0], ax_iwv_pos[1], ax_iwv_pos[2]*0.9, ax_iwv_pos[3]])


plot_name = f"MOSAiC_HATPRO_MiRAC-P_sonde_overview_{startdate_wai:%Y%m%d}-{enddate_wai:%Y%m%d}"
if set_dict['save_figures']:
	fig1.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
else:
	plt.show()


print(f"Plot saved to {path_plots + plot_name}.")
HATPRO_DS_prw.close()
HATPRO_DS_hua.close()
HATPRO_DS_ta.close()
HATPRO_DS_ta_bl.close()
MIRAC_DS_prw.close()