import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams.update({'font.family': 'monospace'})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import glob
import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from import_data import *
from data_tools import *

import pdb


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


"""
	Script to comopare IASI data (i.e., IWV) with MWR products on Polarstern and radiosonde measurements.
	IASI data must be processed with manage_iasi.py beforehand to reduce the size.
	- import MWR, sonde and IASI data
	- cut data to desired time window
	- visualise
"""


# paths
path_data = {	'iasi': "/mnt/d/heavy_data/IASI_mosaic_processed/",
				'hatpro': "/mnt/d/heavy_data/MOSAiC_radiometers/HATPRO_l2_v01/",
				'mirac-p': "/mnt/d/heavy_data/MOSAiC_radiometers/MiRAC-P_l2_v01/",
				'radiosonde': "/mnt/d/heavy_data/MOSAiC_radiosondes/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/mosaic_iasi/"

# additional settings:
set_dict = {'save_figures': True,				# if True, figure is saved to file
			'which_retrievals': 'iwv',			# specify which variable of MWR data is to be imported; here, only 'iwv'
			'sonde_version': 'level_2',			# specify radiosonde version, here, only 'level_2'
			'date_start': "2020-04-10",
			'date_end': "2020-04-21",			# including end date
			'time_series': False,				# plots an IWV time series of HATPRO, MiRAC-P, radiosonde and IASI
			'time_series_iasi_error': False,		# time series with exclusively IASI data
			'time_height_plot': True,			# plots time x height cross section
			'with_ip': True					# for 'time_height_plot': if True: contourf, which interpolates data;
												# if False: pcolormesh, not interpolating
			}

# for radiosonde importing, take one day before date_start to avoid a data gap in time_height_plot:
set_dict['date_start_pre'] = str(np.datetime64(set_dict['date_start']) - np.timedelta64(1, "D"))

path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)


# import data:
HATPRO_DS = import_hatpro_level2a_daterange_xarray(path_data['hatpro'], set_dict['date_start'], set_dict['date_end'], 
														which_retrieval=set_dict['which_retrievals'])
MIRAC_DS = import_mirac_level2a_daterange_xarray(path_data['mirac-p'], set_dict['date_start'], set_dict['date_end'], 
														which_retrieval=set_dict['which_retrievals'])
m_dict = import_mirac_level2a_daterange_pangaea(path_data['mirac-p'], set_dict['date_start'], set_dict['date_end'], 
														which_retrieval=set_dict['which_retrievals'])

if set_dict['time_height_plot']:	# then also temperature and hum profiles are needed
	date_start_dt = dt.datetime.strptime(set_dict['date_start'], "%Y-%m-%d")
	date_end_dt = dt.datetime.strptime(set_dict['date_end'], "%Y-%m-%d")

	# HATPRO: find the right files, load IWV, humidity and temperature profiles
	all_files_hua = sorted(glob.glob(path_data['hatpro'] + "ioppol_tro_mwr00_l2_hua_v01_*.nc"))
	all_files_ta = sorted(glob.glob(path_data['hatpro'] + "ioppol_tro_mwr00_l2_ta_v01_*.nc"))
	all_files_ta_bl = sorted(glob.glob(path_data['hatpro'] + "ioppol_tro_mwrBL00_l2_ta_v01_*.nc"))

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


	# Filter flagged values (flag > 0):
	HATPRO_DS_hua = HATPRO_DS_hua.isel(time=np.where((HATPRO_DS_hua.flag.values == 0) | (np.isnan(HATPRO_DS_hua.flag.values)))[0])
	HATPRO_DS_ta = HATPRO_DS_ta.isel(time=np.where((HATPRO_DS_ta.flag.values == 0) | (np.isnan(HATPRO_DS_ta.flag.values)))[0])
	HATPRO_DS_ta_bl = HATPRO_DS_ta_bl.isel(time=np.where((HATPRO_DS_ta_bl.flag.values == 0) | (np.isnan(HATPRO_DS_ta_bl.flag.values)))[0])


	# DOWNSAMPLING: For example: to minute averages: load datasets for computation:
	# also remove some time steps where data is missing
	HATPRO_DS_ta.load()
	HATPRO_DS_ta_bl.load()
	HATPRO_DS_hua.load()
	HATPRO_DS_ta = HATPRO_DS_ta.resample(time='60s').mean()
	HATPRO_DS_ta = HATPRO_DS_ta.isel(time=np.where(~np.isnan(HATPRO_DS_ta.ta.values[:,0]))[0])
	HATPRO_DS_hua = HATPRO_DS_hua.resample(time='60s').mean()
	HATPRO_DS_hua = HATPRO_DS_hua.isel(time=np.where(~np.isnan(HATPRO_DS_hua.hua.values[:,0]))[0])


# radiosondes:
sonde_dict = import_radiosonde_daterange(path_data['radiosonde'], set_dict['date_start_pre'], set_dict['date_end'], 
										s_version=set_dict['sonde_version'], remove_failed=True, verbose=0)
n_sondes = len(sonde_dict['launch_time'])
sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype("datetime64[s]")

# IASI data (processed with manage_iasi.py): limit IASI to 
IASI_DS = import_iasi_processed(path_data['iasi'], "2020-04-09", set_dict['date_end'])

# limit HATPRO and MiRAC-P data to values where flag indicates good data: (no need for downsampling because
# data size is sufficiently small
HATPRO_DS = HATPRO_DS.isel(time=(HATPRO_DS.flag.values == 0))
MIRAC_DS = MIRAC_DS.isel(time=(MIRAC_DS.flag.values ==0))

# limit IASI data to indices where flag indicates good data:
if not set_dict['time_series_iasi_error']: # flags to be plotted without filtering
	IASI_DS = IASI_DS.isel(time=(np.where(~np.any(IASI_DS.flag_fgcheck > 511, axis=1))[0]))



# visualise:
# colours:
fs = 24
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15

dt_fmt = mdates.DateFormatter("%b %d")

c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.779,0.615)		# MiRAC-P
c_RS = (1,0.435,0)			# radiosondes
c_I = (1.0,0.83,0.31)		# IASI


if set_dict['time_series']:
	f1 = plt.figure(figsize=(15,7.5))
	a1 = plt.axes()

	# axis limits:
	y_lims = [0, 15]		# kg m-2
	x_lims = [np.datetime64(set_dict['date_start']).astype('datetime64[s]'), 
				np.datetime64(set_dict['date_end']) + np.timedelta64(86400, 's')]
	time_line_daily = np.arange(x_lims[0], x_lims[1], np.timedelta64(86400, 's'))

	# plot: 5-min running mean for hatpro and mirac-p:
	hatpro_iwv = running_mean_datetime(HATPRO_DS.prw.values, 300, HATPRO_DS.time.values.astype("datetime64[s]").astype(np.float32))
	mirac_iwv = running_mean_datetime(MIRAC_DS.prw.values, 300, MIRAC_DS.time.values.astype("datetime64[s]").astype(np.float32))

	# # and bring IASI to a higher temporal resolution to avoid automatic line fillings between the larger gaps (surely, markers could
	# # be used, but they're too bulky).
	# IASI_DS_iwv_plot = IASI_DS.iwv_mean.resample(time='60s').asfreq()	# asfreq ensures that new time steps are NANs!


	a1.plot(HATPRO_DS.time.values, hatpro_iwv, linewidth=1.5, color=c_H, label='HATPRO')
	a1.plot(MIRAC_DS.time.values, mirac_iwv, linewidth=1.5, color=c_M, label='MiRAC-P')
	a1.plot(sonde_dict['launch_time_npdt'], sonde_dict['iwv'], linestyle='none', linewidth=0.5,
			marker='.', markersize=marker_size, markerfacecolor=c_RS, markeredgecolor=(0,0,0),
			markeredgewidth=0.5, label='Radiosonde')
	# a1.plot(IASI_DS.time.values, IASI_DS.iwv_mean.values, linewidth=1.5, color=c_I, label='IASI')
	# a1.plot(IASI_DS.time.values, IASI_DS.iwv_mean.values, linestyle='none', linewidth=0.5, 
			# marker='s', markersize=marker_size/4, markerfacecolor=c_I, markeredgecolor=(1,1,1,0), 
			# markeredgewidth=0.5, label='IASI')
	a1.errorbar(IASI_DS.time.values, IASI_DS.iwv_mean.values, yerr=IASI_DS.iwv_std.values, ecolor=c_I,
				elinewidth=1.0, capsize=marker_size/8, markerfacecolor=c_I, markeredgecolor=(0,0,0,0.25), markeredgewidth=0.5,
				marker='s', markersize=marker_size/4, linestyle='none', capthick=1.0, label='IASI')
						

	# aux info:

	# legends and coluorbars:
	lh, ll = a1.get_legend_handles_labels()
	a1.legend(lh, ll, loc='upper left', fontsize=fs, markerscale=1.5)

	# set axis limits:
	a1.set_xlim(x_lims[0], x_lims[1])
	a1.set_ylim(y_lims[0], y_lims[1])

	# set ticks and tick labels and parameters:
	a1.set_xticks(time_line_daily[1::2])
	a1.xaxis.set_major_formatter(dt_fmt)
	a1.tick_params(axis='both', labelsize=fs_small)

	# grid:
	a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# set labels:
	a1.set_xlabel(f"Date in {HATPRO_DS.time[0].dt.year.values}", fontsize=fs)
	a1.set_ylabel("Integrated Water Vapour ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)


	if set_dict['save_figures']:

		plotname = ("MOSAiC_HATPRO_MiRAC-P_radiosonde_IASI_iwv_time_series_w_errorbars" +
					f"{set_dict['date_start'].replace('-','')}-{set_dict['date_end'].replace('-','')}.png")
		plot_file = path_plots + plotname
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)					

	else:
		plt.show()
		pdb.set_trace()


if set_dict['time_series_iasi_error']:

	flag_name = 'flag_fgcheck'

	f1, a1 = plt.subplots(ncols=1, nrows=2, figsize=(12,12), constrained_layout=True)
	a1 = a1.flatten()

	# axis limits:
	x_lims = [np.datetime64(set_dict['date_start']).astype('datetime64[s]'), 
				np.datetime64(set_dict['date_end']) + np.timedelta64(86400, 's')]
	time_line_daily = np.arange(x_lims[0], x_lims[1], np.timedelta64(86400, 's'))

	# plot iasi IWV:
	a1[0].plot(IASI_DS.time.values, IASI_DS.iwv_mean.values, linewidth=1.5, color=c_I, label='IASI')

	# plot IASI flag:
	for kk in range(len(IASI_DS.n_hits)):
		a1[1].plot(IASI_DS.time.values, IASI_DS[flag_name].values[:,kk], linewidth=1.25)

	# aux info:

	# legends and coluorbars:
	lh, ll = a1[0].get_legend_handles_labels()
	a1[0].legend(lh, ll, loc='upper left', fontsize=fs_small, markerscale=1.5)

	# set axis limits:
	a1[0].set_xlim(x_lims[0], x_lims[1])
	a1[1].set_xlim(x_lims[0], x_lims[1])
	a1[1].set_yscale('log')

	# set ticks and tick labels and parameters:
	a1[0].tick_params(axis='both', labelsize=fs_dwarf)
	a1[1].tick_params(axis='both', labelsize=fs_dwarf)

	# grid:
	a1[0].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
	a1[1].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# set labels:
	a1[1].set_xlabel(f"Date in {HATPRO_DS.time[0].dt.year.values}", fontsize=fs_small)
	a1[0].set_ylabel("Integrated Water Vapour ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_small)
	a1[1].set_ylabel(flag_name, fontsize=fs_small)


	plt.show()
	pdb.set_trace()


if set_dict['time_height_plot']:
	fs = 14
	fs_small = fs - 2
	fs_dwarf = fs - 4
	marker_size = 15


	# merge BL and zenith temperature profile:
	# lowest 2000 m: BL only; 2000-2500m: linear transition from BL to zenith; >2500: zenith only:
	# Leads to loss in temporal resolution of HATPRO: interpolated BL scan time to zenith time grid:
	th_bl, th_tr = 2000, 2500		# tr: transition zone
	idx_bl = np.where(HATPRO_DS_ta.height.values <= th_bl)[0][-1]	# final index in BL
	idx_tr = np.where((HATPRO_DS_ta.height.values > th_bl) & (HATPRO_DS_ta.height.values <= th_tr))[0]
	pc_bl = (-1.0/(th_tr-th_bl))*HATPRO_DS_ta.height.values[idx_tr] + 1.0/(th_tr-th_bl)*th_tr 	# percentage of BL mode
	pc_ze = 1.0 - pc_bl												# respective percentage of zenith mode during transition
	ta_bl = HATPRO_DS_ta_bl.ta.interp(coords={'time': HATPRO_DS_ta.time})
	ta_combined = HATPRO_DS_ta.ta
	ta_combined[:,:idx_bl+1] = ta_bl[:,:idx_bl+1]
	ta_combined[:,idx_tr] = pc_bl*ta_bl[:,idx_tr] + pc_ze*HATPRO_DS_ta.ta[:,idx_tr]
	HATPRO_DS_ta['ta_combined'] = xr.DataArray(ta_combined, coords={'time': HATPRO_DS_ta.time, 'height': HATPRO_DS_ta.height},
											dims=['time', 'height'])


	# Compute absolute humidity profile from spec hum. in IASI data, then compute the air density
	# profile to compute a height grid:
	iasi_rho_v_mean = convert_spechum_to_abshum(IASI_DS.atmospheric_temperature_mean.values, 
												IASI_DS.pressure_levels_humidity.values,
												IASI_DS.atmospheric_water_vapor_mean.values)
	IASI_DS['rho_v_mean'] = xr.DataArray(iasi_rho_v_mean, dims=['time', 'nlq'], 
											attrs={	'long_name': "Absolute humidity profile",
													'units': "kg m-3"})
	# iasi_rho_mean = rho_air(IASI_DS.pressure_levels_humidity.values, IASI_DS.atmospheric_temperature_mean.values, 
							# IASI_DS.rho_v_mean.values)
	# iasi_hum_height = Z_from_pres(IASI_DS.pressure_levels_humidity.values, iasi_rho_mean, IASI_DS.surface_pressure_mean.values)


	# set IASI values exceeding common sense to nan:
	IASI_DS['atmospheric_temperature_mean'] = xr.where(IASI_DS.atmospheric_temperature_mean < 335.0, x=IASI_DS.atmospheric_temperature_mean, y=np.nan)
	IASI_DS['rho_v_mean'] = xr.where(IASI_DS.atmospheric_water_vapor_mean < 0.1, x=IASI_DS.rho_v_mean, y=np.nan)


	dt_fmt = mdates.DateFormatter("%b %d")
	datetick_auto = False

	# create x_ticks depending on the date range:
	date_range_delta = (date_end_dt - date_start_dt)
	if (date_range_delta < dt.timedelta(days=10)) & (date_range_delta >= dt.timedelta(days=3)):
		x_tick_delta = dt.timedelta(hours=12)
		dt_fmt = mdates.DateFormatter("%b %d %HZ")
	elif (date_range_delta < dt.timedelta(days=3)) & (date_range_delta >= dt.timedelta(days=2)):
		x_tick_delta = dt.timedelta(hours=6)
		dt_fmt = mdates.DateFormatter("%b %d %HZ")
	elif date_range_delta < dt.timedelta(days=2):
		x_tick_delta = dt.timedelta(hours=3)
		dt_fmt = mdates.DateFormatter("%b %d %HZ")
	elif (date_range_delta < dt.timedelta(days=20)) & (date_range_delta >= dt.timedelta(days=10)):
		x_tick_delta = dt.timedelta(hours=24)
		dt_fmt = mdates.DateFormatter("%b %d")
	else:
		x_tick_delta = dt.timedelta(days=3)
		dt_fmt = mdates.DateFormatter("%b %d")


	x_ticks_dt = mdates.drange(date_start_dt, date_end_dt + dt.timedelta(hours=1), x_tick_delta)

	f1 = plt.figure(figsize=(10,15))
	ax_hua_rs = plt.subplot2grid((6,1), (0,0))		# radiosonde abs. hum. profiles
	ax_hua_hat = plt.subplot2grid((6,1), (1,0))		# hatpro abs. hum. profiles
	ax_hua_ias = plt.subplot2grid((6,1), (2,0))		# IASI abs. hum. profiles
	ax_ta_rs = plt.subplot2grid((6,1), (3,0))		# radiosonde temperature profiles
	ax_ta_hat = plt.subplot2grid((6,1), (4,0))		# hatpro temperature profiles (zenith)
	ax_ta_ias = plt.subplot2grid((6,1), (5,0))		# IASI temperature profiles (zenith)


	# ax lims:
	height_lims = [0, 8000]		# m
	time_lims = [date_start_dt, date_end_dt]

	rho_v_levels = np.arange(0.0, 5.51, 0.25)		# in g m-3
	temp_levels = np.arange(200.0, 285.001, 2)		# in K
	temp_contour_levels = np.arange(-70.0, 50.1, 10.0)		# in deg C

	# colors:
	rho_v_cmap = mpl.cm.get_cmap('gist_earth', len(rho_v_levels))
	temp_cmap = mpl.cm.get_cmap('nipy_spectral', len(temp_levels))
	temp_contour_cmap = np.full(temp_contour_levels.shape, "#000000")


	if set_dict['with_ip']:
		# plot radiosonde humidity profile:
		xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
		rho_v_rs_curtain = ax_hua_rs.contourf(yv, xv, 1000*sonde_dict['rho_v'], levels=rho_v_levels,
											cmap=rho_v_cmap, extend='max')


		print("Plotting HATPRO humidity profile....")
		# plot hatpro hum profile:
		xv, yv = np.meshgrid(HATPRO_DS_hua.height.values, HATPRO_DS_hua.time.values)
		rho_v_hat_curtain = ax_hua_hat.contourf(yv, xv, 1000*HATPRO_DS_hua.hua.values, levels=rho_v_levels,
											cmap=rho_v_cmap, extend='max')

		# plot IASI hum profile: (and surface pressure for orientation)
		xv, yv = np.meshgrid(IASI_DS.pressure_levels_humidity.values*0.01, IASI_DS.time.values)
		rho_v_ias_curtain = ax_hua_ias.contourf(yv, xv, 1000*IASI_DS.rho_v_mean.values, levels=rho_v_levels,
											cmap=rho_v_cmap, extend='max')
		ax_hua_ias.plot(IASI_DS.time.values, 0.01*IASI_DS.surface_pressure_mean.values, color=(0,0,0), linewidth=1.25)

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
		xv, yv = np.meshgrid(HATPRO_DS_ta['height'], HATPRO_DS_ta['time'])
		temp_hat_curtain = ax_ta_hat.contourf(yv, xv, HATPRO_DS_ta['ta_combined'], levels=temp_levels,
												cmap=temp_cmap, extend='both')

		# add black contour lines of some temperatures: (only every 500th value to avoid clabel overlap)
		temp_hat_contour = ax_ta_hat.contour(yv[::500,:], xv[::500,:], HATPRO_DS_ta['ta_combined'].values[::500,:] - 273.15, levels=temp_contour_levels,
												colors='black', linewidths=0.9, alpha=0.5)
		ax_ta_hat.clabel(temp_hat_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
						colors='black', inline_spacing=12, fontsize=fs_dwarf)


		# plot IASI temperature profile:
		xv, yv = np.meshgrid(IASI_DS.pressure_levels_temp.values*0.01, IASI_DS.time.values)
		temp_ias_curtain = ax_ta_ias.contourf(yv, xv, IASI_DS.atmospheric_temperature_mean.values, levels=temp_levels,
												cmap=temp_cmap, extend='both')
		ax_ta_ias.plot(IASI_DS.time.values, 0.01*IASI_DS.surface_pressure_mean.values, color=(0,0,0), linewidth=1.25)

	else:
		norm_rho_v = mpl.colors.BoundaryNorm(rho_v_levels, rho_v_cmap.N)
		norm_temp = mpl.colors.BoundaryNorm(temp_levels, temp_cmap.N)

		# radiosonde humidity profile:
		xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
		rho_v_rs_curtain = ax_hua_rs.pcolormesh(yv, xv, 1000*sonde_dict['rho_v'], shading='nearest',
												norm=norm_rho_v, cmap=rho_v_cmap)


		print("Plotting HATPRO humidity profile....")
		# plot hatpro hum profile:
		xv, yv = np.meshgrid(RET_DS.height.values, RET_DS.time.values)
		rho_v_hat_curtain = ax_hua_hat.pcolormesh(yv, xv, 1000*RET_DS.hua.values, shading='nearest',
												norm=norm_rho_v, cmap=rho_v_cmap)


		# plot radiosonde temperature profile:
		xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
		temp_rs_curtain = ax_ta_rs.pcolormesh(yv, xv, sonde_dict['temp'], norm=norm_temp,
											cmap=temp_cmap)

		# add black contour lines and contour labels:
		temp_rs_contour = ax_ta_rs.contour(yv, xv, sonde_dict['temp'] - 273.15, levels=temp_contour_levels,
												colors='black', linewidths=0.9, alpha=0.5)
		ax_ta_rs.clabel(temp_rs_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
						colors='black', inline_spacing=10, fontsize=fs_dwarf)


		print("Plotting HATPRO temperature profiles....")
		# plot hatpro zenith temperature profile:
		xv, yv = np.meshgrid(RET_DS['height'], RET_DS['time'])
		temp_hat_curtain = ax_ta_hat.pcolormesh(yv, xv, RET_DS['ta_rm'], norm=norm_temp,
											cmap=temp_cmap)

		# add black contour lines of some temperatures:
		temp_hat_contour = ax_ta_hat.contour(yv, xv, RET_DS['ta_rm'] - 273.15, levels=temp_contour_levels,
												colors='black', linewidths=0.9, alpha=0.5)
		ax_ta_hat.clabel(temp_hat_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
						colors='black', inline_spacing=12, fontsize=fs_dwarf)



	# add figure identifier of subplots: a), b), ...
	ax_hua_rs.text(0.02, 0.95, "a) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
					transform=ax_hua_rs.transAxes)
	ax_hua_hat.text(0.02, 0.95, "b) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
					transform=ax_hua_hat.transAxes)
	ax_hua_ias.text(0.02, 0.95, "c) IASI ", color=(0.5,0.5,0.5), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_hua_ias.transAxes)
	ax_ta_rs.text(0.02, 0.95, "d) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_rs.transAxes)
	ax_ta_hat.text(0.02, 0.95, "e) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_hat.transAxes)
	ax_ta_ias.text(0.02, 0.95, "f) IASI ", color=(0.5,0.5,0.5), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_ias.transAxes)


	# legends and colorbars:
	cb_hua_rs = f1.colorbar(mappable=rho_v_rs_curtain, ax=ax_hua_rs, use_gridspec=True,
								orientation='vertical', extend='max', fraction=0.09, pad=0.01, shrink=0.9)
	cb_hua_rs.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
	cb_hua_rs.ax.tick_params(labelsize=fs_dwarf)

	cb_hua_hat = f1.colorbar(mappable=rho_v_hat_curtain, ax=ax_hua_hat, use_gridspec=True,
								orientation='vertical', extend='max', fraction=0.09, pad=0.01, shrink=0.9)
	cb_hua_hat.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
	cb_hua_hat.ax.tick_params(labelsize=fs_dwarf)

	cb_hua_ias = f1.colorbar(mappable=rho_v_ias_curtain, ax=ax_hua_ias, use_gridspec=True,
								orientation='vertical', extend='max', fraction=0.09, pad=0.01, shrink=0.9)
	cb_hua_ias.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
	cb_hua_ias.ax.tick_params(labelsize=fs_dwarf)

	cb_ta_rs = f1.colorbar(mappable=temp_rs_curtain, ax=ax_ta_rs, use_gridspec=True,
								orientation='vertical', extend='both', fraction=0.09, pad=0.01, shrink=0.9)
	cb_ta_rs.set_label(label="T (K)", fontsize=fs_small)
	cb_ta_rs.ax.tick_params(labelsize=fs_dwarf)

	cb_ta_hat = f1.colorbar(mappable=temp_hat_curtain, ax=ax_ta_hat, use_gridspec=True,
								orientation='vertical', extend='both', fraction=0.09, pad=0.01, shrink=0.9)
	cb_ta_hat.set_label(label="T (K)", fontsize=fs_small)
	cb_ta_hat.ax.tick_params(labelsize=fs_dwarf)

	cb_ta_ias = f1.colorbar(mappable=temp_ias_curtain, ax=ax_ta_ias, use_gridspec=True,
								orientation='vertical', extend='both', fraction=0.09, pad=0.01, shrink=0.9)
	cb_ta_ias.set_label(label="T (K)", fontsize=fs_small)
	cb_ta_ias.ax.tick_params(labelsize=fs_dwarf)


	# set axis limits:
	ax_hua_rs.set_xlim(left=time_lims[0], right=time_lims[1])
	ax_hua_hat.set_xlim(left=time_lims[0], right=time_lims[1])
	ax_hua_ias.set_xlim(left=time_lims[0], right=time_lims[1])
	ax_ta_rs.set_xlim(left=time_lims[0], right=time_lims[1])
	ax_ta_hat.set_xlim(left=time_lims[0], right=time_lims[1])
	ax_ta_ias.set_xlim(left=time_lims[0], right=time_lims[1])

	ax_hua_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
	ax_hua_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])
	ax_hua_ias.set_ylim(bottom=1050.0, top=330.0)
	ax_ta_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
	ax_ta_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])
	ax_ta_ias.set_ylim(bottom=1050.0, top=330.0)


	# set x ticks and tick labels:
	ax_hua_rs.xaxis.set_ticks(x_ticks_dt)
	ax_hua_rs.xaxis.set_ticklabels([])
	ax_hua_hat.xaxis.set_ticks(x_ticks_dt)
	ax_hua_hat.xaxis.set_ticklabels([])
	ax_hua_ias.xaxis.set_ticks(x_ticks_dt)
	ax_hua_ias.xaxis.set_ticklabels([])
	ax_ta_rs.xaxis.set_ticks(x_ticks_dt)
	ax_ta_rs.xaxis.set_ticklabels([])
	# ax_ta_rs.xaxis.set_major_formatter(dt_fmt)			#################
	ax_ta_hat.xaxis.set_ticks(x_ticks_dt)
	ax_ta_hat.xaxis.set_ticklabels([])
	ax_ta_ias.xaxis.set_ticks(x_ticks_dt)
	ax_ta_ias.xaxis.set_major_formatter(dt_fmt)


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
	ax_ta_ias.tick_params(axis='x', labelsize=fs_small, labelrotation=90)


	# y tick parameters:
	ax_hua_rs.tick_params(axis='y', labelsize=fs_small)
	ax_hua_hat.tick_params(axis='y', labelsize=fs_small)
	ax_hua_ias.tick_params(axis='y', labelsize=fs_small)
	ax_ta_rs.tick_params(axis='y', labelsize=fs_small)
	ax_ta_hat.tick_params(axis='y', labelsize=fs_small)
	ax_ta_ias.tick_params(axis='y', labelsize=fs_small)


	# grid:
	ax_hua_rs.grid(which='major', axis='both', alpha=0.4)
	ax_hua_hat.grid(which='major', axis='both', alpha=0.4)
	ax_hua_ias.grid(which='major', axis='both', alpha=0.4)
	ax_ta_rs.grid(which='major', axis='both', alpha=0.4)
	ax_ta_hat.grid(which='major', axis='both', alpha=0.4)
	ax_ta_ias.grid(which='major', axis='both', alpha=0.4)


	# set labels:
	ax_hua_rs.set_ylabel("Height (m)", fontsize=fs)
	ax_hua_hat.set_ylabel("Height (m)", fontsize=fs)
	ax_hua_ias.set_ylabel("Pressure (hPa)", fontsize=fs)
	ax_ta_rs.set_ylabel("Height (m)", fontsize=fs)
	ax_ta_hat.set_ylabel("Height (m)", fontsize=fs)
	ax_ta_ias.set_ylabel("Pressure (hPa)", fontsize=fs)

	ax_ta_ias.set_xlabel(f"{date_start_dt.year}", fontsize=fs)

	# if with_titles:
		# ax_iwv.set_title("IWV (a) and profiles of humidity (b,c) and temperature (d-e) from\nHATPRO and radiosondes", fontsize=fs)


	# Limit axis spacing:
	plt.subplots_adjust(hspace=0.0)			# removes space between subplots



	if set_dict['save_figures']:
		plot_name = ("MOSAiC_HATPRO_radiosonde_IASI_time_height_profiles_" +
					f"{set_dict['date_start'].replace('-','')}-{set_dict['date_end'].replace('-','')}")
		if not set_dict['with_ip']: plot_name += "_no_ip"

		plot_file = path_plots + plot_name + ".png"
		f1.savefig(plot_file, dpi=400, bbox_inches='tight')
		print("Saved " + plot_file)
	else:
		plt.show()
		pdb.set_trace()