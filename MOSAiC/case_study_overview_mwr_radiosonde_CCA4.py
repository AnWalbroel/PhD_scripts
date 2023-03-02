import xarray as xr
import numpy as np
import glob
import os
import datetime as dt
from copy import deepcopy
import pdb
from import_data import import_radiosonde_daterange
from data_tools import find_files_daterange, numpydatetime64_to_epochtime, datetime_to_epochtime
from met_tools import wspeed_wdir_to_u_v
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

path_plots = "/net/blanc/awalbroe/Plots/figures_data_paper/temp/"			# output of plots

with_ip = True			# if True: contourf will be used (lin. interpolation between data points for smooth img); 
						# if False: no interp (pcolormesh)
wind_barb_plot = True	# if True: wind barb plot overlayed over radiosonde temperature plot
						# if False: wdir and wspeed will be shown in a subplot
save_figures = True		# if true, figures will be saved to file
save_figures_eps = False		# if true, figures will be saved to a vector graphics file
with_titles = False		# if True, plots will have titles (False for publication plots)
considered_period = 'moist_air_intrusion'
radiosonde_version = 'level_2'


# wind barb plot only for non-interpolated version:
if with_ip and not wind_barb_plot:
	print("Currently, non-wind-barb-plot is only implemented for non-interpolated version.")
	with_ip = False


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
										remove_failed=True, verbose=1, with_wind=True)
n_sondes = len(sonde_dict['launch_time'])
sonde_dict['launch_time_npdt'] = sonde_dict['launch_time'].astype("datetime64[s]")
sonde_dict['U'], sonde_dict['V'] = wspeed_wdir_to_u_v(sonde_dict['wspeed']*1.94384, sonde_dict['wdir'], 'from')		# from m s-1 to knots

# regrid to 12 hourly time and 1000 m height grid:
n_days = (date_end_dt-date_start_dt).days + 1
n_steps = n_days*2 + 1
n_height = len(sonde_dict['height'][0,:])

# time:
time_ip = np.array([np.datetime64(date_start_dt.strftime("%Y-%m-%dT%H:%M:%S")) + np.timedelta64(k*12, "h") for k in range(n_steps)])
sonde_dict['U_ip'] = np.zeros((n_steps, n_height))
sonde_dict['V_ip'] = np.zeros((n_steps, n_height))
time_ip_epoch = numpydatetime64_to_epochtime(time_ip)
for k in range(n_height): 
	sonde_dict['U_ip'][:,k] = np.interp(time_ip_epoch, sonde_dict['launch_time'], sonde_dict['U'][:,k])
	sonde_dict['V_ip'][:,k] = np.interp(time_ip_epoch, sonde_dict['launch_time'], sonde_dict['V'][:,k])

# height:
# Z_ip = np.arange(500, 7501, 1000)
Z_ip = np.arange(0, 8001, 1000)
sonde_dict['U_ip'] = np.array([np.interp(Z_ip, sonde_dict['height'][0,:], sonde_dict['U_ip'][k,:],
								left=np.nan, right=np.nan) for k in range(n_steps)])
sonde_dict['V_ip'] = np.array([np.interp(Z_ip, sonde_dict['height'][0,:], sonde_dict['V_ip'][k,:],
								left=np.nan, right=np.nan) for k in range(n_steps)])
time_ip_repeat = np.repeat(np.reshape(time_ip, (n_steps,1)), len(Z_ip), axis=1)
Z_ip_repeat = np.repeat(np.reshape(Z_ip, (1,len(Z_ip))), n_steps, axis=0)


# HATPRO: find the right files, load IWV, humidity and temperature profiles
all_files_hua = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_hua_v01_*.nc"))
all_files_ta = sorted(glob.glob(path_hatpro + "ioppol_tro_mwr00_l2_ta_v01_*.nc"))
all_files_ta_bl = sorted(glob.glob(path_hatpro + "ioppol_tro_mwrBL00_l2_ta_v01_*.nc"))

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
ok_idx = np.where((HATPRO_DS_hua.flag.values == 0) | (np.isnan(HATPRO_DS_hua.flag.values)))[0]
HATPRO_DS_hua = HATPRO_DS_hua.isel(time=ok_idx)
ok_idx = np.where((HATPRO_DS_ta.flag.values == 0) | (np.isnan(HATPRO_DS_ta.flag.values)))[0]
HATPRO_DS_ta = HATPRO_DS_ta.isel(time=ok_idx)
ok_idx = np.where((HATPRO_DS_ta_bl.flag.values == 0) | (np.isnan(HATPRO_DS_ta_bl.flag.values)))[0]
HATPRO_DS_ta_bl = HATPRO_DS_ta_bl.isel(time=ok_idx)


# Merge BL and zenith temperature profiles from HATPRO:
# lowest 2000 m: BL only; 2000-2500m: linear transition from BL to zenith; >2500: zenith only:
# Leads to loss in temporal resolution of HATPRO: interpolated to BL scan time grid:
th_bl, th_tr = 2000, 2500		# tr: transition zone
idx_bl = np.where(HATPRO_DS_ta.height.values <= th_bl)[0][-1]
idx_tr = np.where((HATPRO_DS_ta.height.values > th_bl) & (HATPRO_DS_ta.height.values <= th_tr))[0]
pc_bl = (-1/(th_tr-th_bl))*HATPRO_DS_ta.height.values[idx_tr] + 1/(th_tr-th_bl)*th_tr 	# percentage of BL mode
pc_ze = 1 - pc_bl												# respective percentage of zenith mode during transition
HATPRO_DS_ta = HATPRO_DS_ta.interp(coords={'time': HATPRO_DS_ta_bl.time})
HATPRO_ta_combined = HATPRO_DS_ta.ta.values
HATPRO_ta_combined[:,:idx_bl+1] = HATPRO_DS_ta_bl.ta.values[:,:idx_bl+1]
HATPRO_ta_combined[:,idx_tr] = pc_bl*HATPRO_DS_ta_bl.ta.values[:,idx_tr] + pc_ze*HATPRO_DS_ta.ta.values[:,idx_tr]
HATPRO_DS_ta['ta_combined'] = xr.DataArray(HATPRO_ta_combined, coords={'time': HATPRO_DS_ta.time, 'height': HATPRO_DS_ta.height},
											dims=['time', 'height'])


# Plot:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_RS = (1,0.435,0)			# radiosondes
rho_v_cmap = mpl.cm.gist_earth
temp_cmap = mpl.cm.nipy_spectral

import locale
locale.setlocale(locale.LC_ALL, "en_GB.utf8")
dt_fmt = mdates.DateFormatter("%b %d") # (e.g. "Feb 23")
datetick_auto = False

# create x_ticks depending on the date range:
date_range_delta = (date_end_dt - date_start_dt)
if date_range_delta < dt.timedelta(days=21):
	x_tick_delta = dt.timedelta(days=1)
	dt_fmt = mdates.DateFormatter("%b %d %HZ")
else:
	x_tick_delta = dt.timedelta(days=20)

x_ticks_dt = mdates.drange(date_start_dt, date_end_dt + dt.timedelta(days=1,hours=1), x_tick_delta)

if wind_barb_plot:
	fig1 = plt.figure(figsize=(10,10))
	ax_hua_rs = plt.subplot2grid((4,1), (0,0))		# radiosonde abs. hum. profiles
	ax_hua_hat = plt.subplot2grid((4,1), (1,0))		# hatpro abs. hum. profiles
	ax_ta_rs = plt.subplot2grid((4,1), (2,0))		# radiosonde temperature profiles
	ax_ta_hat = plt.subplot2grid((4,1), (3,0))		# hatpro temperature profiles (zenith and BL)
else:
	fig1 = plt.figure(figsize=(10,12.5))
	ax_hua_rs = plt.subplot2grid((5,1), (0,0))		# radiosonde abs. hum. profiles
	ax_hua_hat = plt.subplot2grid((5,1), (1,0))		# hatpro abs. hum. profiles
	ax_ta_bl_hat = plt.subplot2grid((5,1), (2,0))	# radiosonde wind dir and speed profiles
	ax_ta_rs = plt.subplot2grid((5,1), (3,0))		# radiosonde temperature profiles
	ax_ta_hat = plt.subplot2grid((5,1), (4,0))		# hatpro temperature profiles (zenith and BL)


# ax lims:
height_lims = [0, 8000]		# m
time_lims = [date_start_dt, date_end_dt + dt.timedelta(days=1)]

rho_v_levels = np.arange(0, 5.51, 0.2)		# in g m-3
temp_levels = np.arange(200, 285.001, 2)	# in K
temp_contour_levels = np.arange(-70.0, 50.1, 10.0)		# in deg C
temp_contour_cmap = np.full(temp_contour_levels.shape, "#000000")


if with_ip:
	# plot radiosonde humidity profile:
	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	rho_v_rs_curtain = ax_hua_rs.contourf(yv, xv, 1000*sonde_dict['rho_v'], levels=rho_v_levels,
										cmap=rho_v_cmap)

	if wind_barb_plot:
		ax_hua_rs.barbs(time_ip_repeat, Z_ip_repeat, sonde_dict['U_ip'], sonde_dict['V_ip'],
								length=4.5, pivot='middle', barbcolor=(1,1,1,0.5), rounding=True,
								zorder=9999) # zorder=100 ensures that the barbs are on top of everything else


	print("Plotting HATPRO humidity profile....")
	# plot hatpro hum profile:
	xv, yv = np.meshgrid(HATPRO_DS_hua.height.values, HATPRO_DS_hua.time.values)
	rho_v_hat_curtain = ax_hua_hat.contourf(yv, xv, 1000*HATPRO_DS_hua.hua.values, levels=rho_v_levels,
										cmap=rho_v_cmap)


	# plot radiosonde temperature profile:
	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	temp_rs_curtain = ax_ta_rs.contourf(yv, xv, sonde_dict['temp'], levels=temp_levels,
										cmap=temp_cmap)

	# add black contour lines of some temperatures:
	temp_rs_contour = ax_ta_rs.contour(yv, xv, sonde_dict['temp'] - 273.15, levels=temp_contour_levels,
											colors=temp_contour_cmap, alpha=0.5)
	ax_ta_rs.clabel(temp_rs_contour, inline=True, fmt="%i", inline_spacing=18, fontsize=fs_small)


	print("Plotting HATPRO temperature profiles....")
	# plot combined hatpro temperature profile:
	xv, yv = np.meshgrid(HATPRO_DS_ta.height.values, HATPRO_DS_ta.time.values)
	temp_hat_curtain = ax_ta_hat.contourf(yv, xv, HATPRO_DS_ta.ta_combined.values, levels=temp_levels,
											cmap=temp_cmap)

	# add black contour lines of some temperatures:
	temp_hat_contour = ax_ta_hat.contour(yv, xv, HATPRO_DS_ta.ta_combined.values - 273.15, levels=temp_contour_levels,
											colors=temp_contour_cmap, alpha=0.5)
	ax_ta_hat.clabel(temp_hat_contour, inline=True, fmt="%i", inline_spacing=18, fontsize=fs_small)

else:
	# Norms for colourmap
	norm_rho_v = mpl.colors.BoundaryNorm(rho_v_levels, rho_v_cmap.N)
	norm_temp = mpl.colors.BoundaryNorm(temp_levels, temp_cmap.N)

	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	rho_v_rs_curtain = ax_hua_rs.pcolormesh(yv, xv, 1000*sonde_dict['rho_v'], shading='nearest', norm=norm_rho_v,
										cmap=rho_v_cmap)

	if wind_barb_plot:
		ax_hua_rs.barbs(time_ip_repeat, Z_ip_repeat, sonde_dict['U_ip'], sonde_dict['V_ip'],
								length=4.5, pivot='middle', barbcolor=(1,1,1,0.5), rounding=True,
								zorder=9999) # zorder=100 ensures that the barbs are on top of everything else


	print("Plotting HATPRO humidity profile....")
	# plot hatpro hum profile:
	xv, yv = np.meshgrid(HATPRO_DS_hua.height.values, HATPRO_DS_hua.time.values)
	rho_v_hat_curtain = ax_hua_hat.pcolormesh(yv, xv, 1000*HATPRO_DS_hua.hua.values, shading='nearest', norm=norm_rho_v,
										cmap=rho_v_cmap)


	# plot radiosonde temperature profile:
	xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
	temp_rs_curtain = ax_ta_rs.pcolormesh(yv, xv, sonde_dict['temp'], shading='nearest', norm=norm_temp,
										cmap=temp_cmap)

	# add black contour lines of some temperatures:
	temp_rs_contour = ax_ta_rs.contour(yv, xv, sonde_dict['temp'] - 273.15, levels=temp_contour_levels,
											colors=temp_contour_cmap, alpha=0.5)
	ax_ta_rs.clabel(temp_rs_contour, inline=True, fmt="%i", inline_spacing=18, fontsize=fs_small)


	print("Plotting HATPRO temperature profiles....")
	# plot combined hatpro temperature profile:
	xv, yv = np.meshgrid(HATPRO_DS_ta.height.values, HATPRO_DS_ta.time.values)
	temp_hat_curtain = ax_ta_hat.pcolormesh(yv, xv, HATPRO_DS_ta.ta_combined.values, shading='nearest', norm=norm_temp,
										cmap=temp_cmap)

	# add black contour lines of some temperatures:
	temp_hat_contour = ax_ta_hat.contour(yv, xv, HATPRO_DS_ta.ta_combined.values - 273.15, levels=temp_contour_levels,
											colors=temp_contour_cmap, alpha=0.5)
	ax_ta_hat.clabel(temp_hat_contour, inline=True, fmt="%i", inline_spacing=18, fontsize=fs_small)


	if not wind_barb_plot:

		# plot radiosonde wind direction (colours) and speed (alpha) profiles:
		xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
		wdir_cmap = mpl.cm.hsv
		wspeed_cmap = mpl.cm.binary

		# modify colormap to start with blueish (0 deg == blue == cold)
		wdir_cmap_array = wdir_cmap(range(wdir_cmap.N))
		wdir_cmap_array_mod = deepcopy(wdir_cmap_array)
		wdir_cmap_array_mod[:256-128,:] = wdir_cmap_array[128:,:]
		wdir_cmap_array_mod[256-128:,:] = wdir_cmap_array[:128,:]
		from matplotlib.colors import ListedColormap
		wdir_cmap = ListedColormap(wdir_cmap_array_mod)
		wdir_levels = np.arange(0,361,45)
		norm_wdir = mpl.colors.BoundaryNorm(wdir_levels, wdir_cmap.N)
		norm_wspeed = mpl.colors.BoundaryNorm(np.linspace(0,25,256), wspeed_cmap.N)

		# New grid for radiosonde wind data: minutes since time_lims[0]:
		time_grid_start = sonde_dict['launch_time_npdt'][0].astype("datetime64[m]")
		time_grid_end = sonde_dict['launch_time_npdt'][-1].astype("datetime64[m]")
		time_grid = np.arange(time_grid_start, time_grid_end, np.timedelta64(1, "m"))
		len_time_grid = len(time_grid)
		sonde_dict['wdir_gr'] = np.zeros((len_time_grid, n_height))
		sonde_dict['wspeed_gr'] = np.zeros((len_time_grid, n_height))

		# find indices of the launch times on the new grid: the radiosonde launch is centered on a grid point
		# -> that means: i.e., 2nd radiosonde launch will go from half 1->2 launch time to half of 2->3 launch time.
		for k in range(n_sondes):

			LT = sonde_dict['launch_time_npdt'][k]		# current sonde launch time
			if k < n_sondes - 1 and k > 0:
				LTm = sonde_dict['launch_time_npdt'][k-1]	# prev LT
				LTp = sonde_dict['launch_time_npdt'][k+1]	# next LT
				idx0 = (LT - (LT - LTm)/2 - time_grid_start).astype("timedelta64[m]").astype("int")
				idx1 = ((LTp - LT)/2 + LT - time_grid_start).astype("timedelta64[m]").astype("int")

			elif k == 0: 
				LTp = sonde_dict['launch_time_npdt'][k+1]
				idx0 = 0			# idx0: index of grid where radiosonde obs from kth launch will start
				idx1 = ((LTp - LT)/2 + LT - time_grid_start).astype("timedelta64[m]").astype("int")

			else:
				LTm = sonde_dict['launch_time_npdt'][k-1]
				idx0 = (LT - (LT - LTm)/2 - time_grid_start).astype("timedelta64[m]").astype("int")
				idx1 = len_time_grid	# idx1: index of grid where radiosonde obs from kth launch will end

			sonde_dict['wdir_gr'][idx0:idx1,:] = sonde_dict['wdir'][k,:]
			sonde_dict['wspeed_gr'][idx0:idx1,:] = sonde_dict['wspeed'][k,:]

		# identify where axis limits are and limit radiosonde regridded data to those limits:
		idx0 = (np.array([datetime_to_epochtime(date_start_dt)]).astype("datetime64[s]")[0] - time_grid_start).astype("timedelta64[m]").astype("int")
		idx1 = (np.array([datetime_to_epochtime(date_end_dt + dt.timedelta(days=1))]).astype("datetime64[s]")[0] - time_grid_start).astype("timedelta64[m]").astype("int")
		sonde_dict['wdir_gr'] = sonde_dict['wdir_gr'][idx0:idx1+1]
		sonde_dict['wspeed_gr'] = sonde_dict['wspeed_gr'][idx0:idx1+1]


		# transform wspeed to 0-1: alpha = 1: speed >= 25 m s-1; alpha = 0: 0 m s-1:
		ths_max = 25.0    	# m s-1
		sonde_dict['wspeed_alpha'] = sonde_dict['wspeed_gr'] * (1.0/ths_max)
		sonde_dict['wspeed_alpha'][sonde_dict['wspeed_alpha'] >= 1.0] = 1.0
		sonde_dict['wspeed_alpha'][np.isnan(sonde_dict['wspeed_alpha'])] = 0.0
		idx_hgt_max = np.where(sonde_dict['height'][0,:] >= height_lims[1])[0][0]
		sonde_dict['wdir_gr'] = sonde_dict['wdir_gr'][:,:idx_hgt_max+1]
		sonde_dict['wspeed_alpha'] = sonde_dict['wspeed_alpha'][:,:idx_hgt_max+1]
		wind_rs_curtain = ax_ta_bl_hat.pcolormesh(yv, xv, np.full_like(sonde_dict['wdir'], np.nan), shading='nearest', norm=norm_wdir,
											cmap=wdir_cmap)
		wind_rs_curtain2 = ax_ta_bl_hat.pcolormesh(yv, xv, np.full_like(sonde_dict['wdir'], np.nan), shading='nearest', norm=norm_wspeed,
											cmap=wspeed_cmap)
		wdir_wspeed_img = ax_ta_bl_hat.imshow(sonde_dict['wdir_gr'].T, cmap=wdir_cmap, norm=norm_wdir, aspect='auto',
											origin='lower', interpolation='none', alpha=sonde_dict['wspeed_alpha'].T)


# add figure identifier of subplots: a), b), ...
ax_hua_rs.text(0.02, 0.95, "a) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax_hua_rs.transAxes)
ax_hua_hat.text(0.02, 0.95, "b) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', 
				transform=ax_hua_hat.transAxes)
if wind_barb_plot:
	ax_ta_rs.text(0.02, 0.95, "c) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_rs.transAxes, zorder=10000)
	ax_ta_hat.text(0.02, 0.95, "d) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_hat.transAxes)
else:
	ax_ta_bl_hat.text(0.02, 0.95, "c) Radiosonde", color=(0,0,0), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_bl_hat.transAxes)
	ax_ta_rs.text(0.02, 0.95, "d) Radiosonde", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_rs.transAxes)
	ax_ta_hat.text(0.02, 0.95, "e) HATPRO", color=(1,1,1), fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax_ta_hat.transAxes)


# legends and colorbars:
cb_hua_rs = fig1.colorbar(mappable=rho_v_rs_curtain, ax=ax_hua_rs, use_gridspec=True,
							extend='max', orientation='vertical', fraction=0.095, pad=0.01, shrink=0.9)
cb_hua_rs.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
cb_hua_rs.ax.tick_params(labelsize=fs_dwarf)

cb_hua_hat = fig1.colorbar(mappable=rho_v_hat_curtain, ax=ax_hua_hat, use_gridspec=True,
							extend='max', orientation='vertical', fraction=0.095, pad=0.01, shrink=0.9)
cb_hua_hat.set_label(label="$\\rho_v$ ($\mathrm{g}\,\mathrm{m}^{-3}$)", fontsize=fs_small)
cb_hua_hat.ax.tick_params(labelsize=fs_dwarf)

cb_ta_rs = fig1.colorbar(mappable=temp_rs_curtain, ax=ax_ta_rs, use_gridspec=True,
							extend='max', orientation='vertical', fraction=0.095, pad=0.01, shrink=0.9)
cb_ta_rs.set_label(label="T (K)", fontsize=fs_small)
cb_ta_rs.ax.tick_params(labelsize=fs_dwarf)

cb_ta_hat = fig1.colorbar(mappable=temp_hat_curtain, ax=ax_ta_hat, use_gridspec=True,
							extend='max', orientation='vertical', fraction=0.095, pad=0.01, shrink=0.9)
cb_ta_hat.set_label(label="T (K)", fontsize=fs_small)
cb_ta_hat.ax.tick_params(labelsize=fs_dwarf)

if not wind_barb_plot:
	cb_ta_bl_hat = fig1.colorbar(mappable=wind_rs_curtain, ax=ax_ta_bl_hat, use_gridspec=True,
								extend='neither', orientation='vertical', fraction=0.095, pad=0.01, shrink=0.9)
	cb_ta_bl_hat.set_label(label="Wind direction (deg)", fontsize=fs_small)
	cb_ta_bl_hat.ax.tick_params(labelsize=fs_dwarf)

	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	cbaxes = inset_axes(ax_ta_bl_hat, width="1%", height="90%", loc='right')
	cb_ta_bl_hat2 = fig1.colorbar(mappable=wind_rs_curtain2, cax=cbaxes, use_gridspec=True,
								extend='neither', orientation='vertical')
	cb_ta_bl_hat2.set_label(label="Wind speed ($\mathrm{m}\,\mathrm{s}^{-1}$)", labelpad=-40, fontsize=fs_small)
	cbaxes.yaxis.set_ticks(np.arange(0,26,5))
	cbaxes.yaxis.set_ticklabels([f"{int(lab)}" for lab in np.arange(0,26,5)])
	cbaxes.yaxis.set_ticks_position("left")
	cb_ta_bl_hat2.ax.tick_params(labelsize=fs_dwarf)


# set axis limits:
ax_hua_rs.set_xlim(left=time_lims[0], right=time_lims[1])
ax_hua_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_hua_hat.set_xlim(left=time_lims[0], right=time_lims[1])
ax_hua_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_ta_rs.set_xlim(left=time_lims[0], right=time_lims[1])
ax_ta_rs.set_ylim(bottom=height_lims[0], top=height_lims[1])
ax_ta_hat.set_xlim(left=time_lims[0], right=time_lims[1])
ax_ta_hat.set_ylim(bottom=height_lims[0], top=height_lims[1])


# set x ticks and tick labels:
ax_hua_rs.xaxis.set_ticks(x_ticks_dt)
ax_hua_rs.xaxis.set_ticklabels([])
ax_hua_hat.xaxis.set_ticks(x_ticks_dt)
ax_hua_hat.xaxis.set_ticklabels([])
ax_ta_rs.xaxis.set_ticks(x_ticks_dt)
ax_ta_rs.xaxis.set_ticklabels([])
# ax_ta_rs.xaxis.set_major_formatter(dt_fmt)			#################
ax_ta_hat.xaxis.set_ticks(x_ticks_dt)
ax_ta_hat.xaxis.set_major_formatter(dt_fmt)
if not wind_barb_plot:

	# WDIR PLOT: set xticks and yticks manually:
	xtick_wdir = []
	t0 = time_grid[idx0]
	while t0 <= time_grid[idx1]:
		xtick_wdir.append((t0 - time_grid[idx0]).astype("timedelta64[m]").astype("int"))
		t0 += np.timedelta64(1, "D")
	ax_ta_bl_hat.xaxis.set_ticks(xtick_wdir)
	ax_ta_bl_hat.xaxis.set_ticklabels([])
	ax_ta_bl_hat.yaxis.set_ticks([0, 400, 800, 1200])
	ax_ta_bl_hat.yaxis.set_ticklabels([f"{int(sonde_dict['height'][0,sh])}" for sh in ax_ta_bl_hat.get_yticks()])


# set y ticks and tick labels:
if ax_hua_rs.get_yticks()[-1] == height_lims[1]:
	ax_hua_rs.yaxis.set_ticks(ax_hua_rs.get_yticks()[:-1])			# remove top tick
if ax_hua_hat.get_yticks()[-1] == height_lims[1]:
	ax_hua_hat.yaxis.set_ticks(ax_hua_hat.get_yticks()[:-1])		# remove top tick
if ax_ta_rs.get_yticks()[-1] == height_lims[1]:
	ax_ta_rs.yaxis.set_ticks(ax_ta_rs.get_yticks()[:-1])			# remove top tick
if ax_ta_hat.get_yticks()[-1] == height_lims[1]:
	ax_ta_hat.yaxis.set_ticks(ax_ta_hat.get_yticks()[:-1])			# remove top tick


# x tick parameters; also align x tick labels correctly:
ax_ta_hat.tick_params(axis='x', labelsize=fs_small)
fig1.canvas.draw()
xlabels = ax_ta_hat.get_xticklabels()
ax_ta_hat.set_xticklabels(xlabels, rotation=45, ha='right', rotation_mode='anchor')


# y tick parameters:
ax_hua_rs.tick_params(axis='y', labelsize=fs_small)
ax_hua_hat.tick_params(axis='y', labelsize=fs_small)
ax_ta_rs.tick_params(axis='y', labelsize=fs_small)
ax_ta_hat.tick_params(axis='y', labelsize=fs_small)
if not wind_barb_plot: 
	ax_ta_bl_hat.tick_params(axis='y', labelsize=fs_small)
	ax_ta_bl_hat.grid(which='major', axis='both', alpha=0.4)
	ax_ta_bl_hat.set_ylabel("Height (m)", fontsize=fs)

# grid:
ax_hua_rs.grid(which='major', axis='both', alpha=0.4)
ax_hua_hat.grid(which='major', axis='both', alpha=0.4)
ax_ta_rs.grid(which='major', axis='both', alpha=0.4)
ax_ta_hat.grid(which='major', axis='both', alpha=0.4)


# set labels:
ax_hua_rs.set_ylabel("Height (m)", fontsize=fs)
ax_hua_hat.set_ylabel("Height (m)", fontsize=fs)
ax_ta_rs.set_ylabel("Height (m)", fontsize=fs)
ax_ta_hat.set_ylabel("Height (m)", fontsize=fs)

ax_ta_hat.set_xlabel(f"{date_start_dt.year}", fontsize=fs)


if with_titles:
	ax_hua_rs.set_title("Profiles of humidity (a,b) and temperature (c-e) from\nHATPRO and radiosondes", fontsize=fs)


# Limit axis spacing:
plt.subplots_adjust(hspace=0.0)			# removes space between subplots


plot_name = "MOSAiC_moist_air_intrusion_overview_hatpro_sonde"
if with_ip: plot_name += "_interp"
if not wind_barb_plot: plot_name += "_no_barbs"
if save_figures:
	fig1.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
	print(f"Plot saved to '{path_plots + plot_name}.png'.")
elif save_figures_eps:
	fig1.savefig(path_plots + plot_name + ".pdf", bbox_inches='tight')
	print(f"Plot saved to '{path_plots + plot_name}.pdf'.")
else:
	plt.show()

HATPRO_DS_hua.close()
HATPRO_DS_ta.close()
HATPRO_DS_ta_bl.close()