import numpy as np
import xarray as xr
import sys
import pdb
import datetime as dt
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from import_data import import_radiosondes_PS131_txt
from data_tools import *
from met_tools import convert_rh_to_abshum


"""
	Visualizing and comparing radiosonde data with MWR (HATPRO (MWR_PRO) in this case).
	IWV, maybe LWP, temperature and humidity profiles will be covered.
	- load HATPRO retrieved and radiosonde data
	- edit data for comparison (find temporal overlap)
	- visualize
"""


# paths:
path_data = {'radiosondes': "/data/radiosondes/Polarstern/PS131_ATWAICE_upper_air_soundings/",
			'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l2/",
			}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/hatpro_lvl2/"


# settings:
if len(sys.argv) == 3:
	date0 = sys.argv[1]
	date1 = sys.argv[2]
else:
	date0 = "2022-07-01"
	date1 = "2022-08-12"
set_dict = {'save_figures': True,
			'iwv_time_series': False,	# IWV time series (sonde and HATPRO)
			'iwv_scatter_plot': False,	# visualization of radiosonde - mwr comparison as scatter plot
			'wai_overview': True,		# time series of temperature, humidity, LWP, IWV of WAI event
			'with_ip': True,			# for 'wai_overview': if True: contourf, which interpolates data;
										# if False: pcolormesh, not interpolating
			'date0': date0,				# start date of data
			'date1': date1,				# end date of data
			'launch_window': 1800.0,	# time window to consider after sonde launch in seconds
			}


# load radiosonde data: concat each sonde to generate a (sonde_launch x height) dict.
# interpolation of radiosonde data to regular grid requried to get a 2D array
files = sorted(glob.glob(path_data['radiosondes'] + "*.txt"))
sonde_dict_temp = import_radiosondes_PS131_txt(files)
new_height = np.arange(0.0, 20000.0001, 20.0)
n_height = len(new_height)
n_sondes = len(sonde_dict_temp.keys())
sonde_dict = {'temp': np.full((n_sondes, n_height), np.nan),
				'pres': np.full((n_sondes, n_height), np.nan),
				'relhum': np.full((n_sondes, n_height), np.nan),
				'height': np.full((n_sondes, n_height), np.nan),
				'wdir': np.full((n_sondes, n_height), np.nan),
				'wspeed': np.full((n_sondes, n_height), np.nan),
				'q': np.full((n_sondes, n_height), np.nan),
				'IWV': np.full((n_sondes,), np.nan),
				'launch_time': np.zeros((n_sondes,)),			# in sec since 1970-01-01 00:00:00 UTC
				'launch_time_npdt': np.full((n_sondes,), np.datetime64("1970-01-01T00:00:00"))}

# interpolate to new grid:
for idx in sonde_dict_temp.keys():
	for key in sonde_dict_temp[idx].keys():
		if key not in ["height", "IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx),:] = np.interp(new_height, sonde_dict_temp[idx]['height'], sonde_dict_temp[idx][key])
		elif key == "height":
			sonde_dict[key][int(idx),:] = new_height
		elif key in ["IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx)] = sonde_dict_temp[idx][key]
del sonde_dict_temp

# compute absolute humidity:
sonde_dict['rho_v'] = convert_rh_to_abshum(sonde_dict['temp'], sonde_dict['relhum'])


# load HATPRO data: concat daily files: run through each day and concat all files to
# a list:
nowdate = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
startdate = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
enddate = dt.datetime.strptime(set_dict['date1'], "%Y-%m-%d")
n_sec = (enddate - nowdate).days*86400
n_t_bl = ((enddate - nowdate).days + 1)*60
n_height = 43		# inquired from ncdump of level 2 MWR_PRO processed HATPRO data
mwr_dict = {'time': np.zeros((n_sec,)),
			'time_npdt': np.full((n_sec), np.datetime64("1970-01-01T00:00:00")),
			'prw': np.zeros((n_sec,)),
			'clwvi': np.zeros((n_sec,)),
			'ta': np.zeros((n_sec, n_height)),
			'hua': np.zeros((n_sec, n_height)),
			}
mwr_bl_dict = {'time': np.zeros((n_t_bl,)),
			'time_npdt': np.full((n_t_bl), np.datetime64("1970-01-01T00:00:00")),
			'ta': np.zeros((n_t_bl, n_height))}
t_idx = 0
t_idx_bl = 0
while nowdate <= enddate:
	# run through folder structure:
	nd_year = f"{nowdate.year:04}"
	nd_month = f"{nowdate.month:02}"
	nd_day = f"{nowdate.day:02}"
	folder_date = path_data['hatpro'] + f"{nd_year}/{nd_month}/{nd_day}/"

	# look for netCDF files... several files are of interest: repeat for prw, clwvi, ta, hua (zenith):
	fois = sorted(glob.glob(folder_date + f"ioppol_tro_mwr00_l2_prw_v00_{nd_year}{nd_month}*.nc"))
	if len(fois) == 1:
		DS = xr.open_dataset(fois[0])
		n_time = len(DS.time.values)

		mwr_dict['time_npdt'][t_idx:t_idx+n_time] = DS.time.values
		mwr_dict['time'][t_idx:t_idx+n_time] = mwr_dict['time_npdt'][t_idx:t_idx+n_time].astype(np.float64)
		mwr_dict['prw'][t_idx:t_idx+n_time] = DS.prw.values
		DS.close()
	else:
		raise ValueError(f"No or too many file(s) on {nowdate}.")

	fois = sorted(glob.glob(folder_date + f"ioppol_tro_mwr00_l2_clwvi_v00_{nd_year}{nd_month}*.nc"))
	if len(fois) == 1:
		DS = xr.open_dataset(fois[0])
		mwr_dict['clwvi'][t_idx:t_idx+n_time] = DS.clwvi.values
		DS.close()

	fois = sorted(glob.glob(folder_date + f"ioppol_tro_mwr00_l2_ta_v00_{nd_year}{nd_month}*.nc"))
	if len(fois) == 1:
		DS = xr.open_dataset(fois[0])
		mwr_dict['height'] = DS.height.values
		mwr_dict['ta'][t_idx:t_idx+n_time,:] = DS.ta.values
		DS.close()

	fois = sorted(glob.glob(folder_date + f"ioppol_tro_mwr00_l2_hua_v00_{nd_year}{nd_month}*.nc"))
	if len(fois) == 1:
		DS = xr.open_dataset(fois[0])
		mwr_dict['hua'][t_idx:t_idx+n_time,:] = DS.hua.values
		DS.close()


	# repeat for BL data:
	fois = sorted(glob.glob(folder_date + f"ioppol_tro_mwrBL00_l2_ta_v00_{nd_year}{nd_month}*.nc"))
	if len(fois) == 1:
		DS = xr.open_dataset(fois[0])
		n_time_bl = len(DS.time.values)
		mwr_bl_dict['time_npdt'][t_idx_bl:t_idx_bl+n_time_bl] = DS.time.values
		mwr_bl_dict['time'][t_idx_bl:t_idx_bl+n_time_bl] = mwr_bl_dict['time_npdt'][t_idx_bl:t_idx_bl+n_time_bl].astype(np.float64)
		mwr_bl_dict['ta'][t_idx_bl:t_idx_bl+n_time_bl,:] = DS.ta.values
		mwr_bl_dict['height'] = DS.height.values
		DS.close()

	# increment loop indices
	t_idx += n_time
	t_idx_bl += n_time_bl
	nowdate += dt.timedelta(days=1)
	

# truncate unneeded space:
mwr_dict['prw'] = mwr_dict['prw'][:t_idx]
mwr_dict['clwvi'] = mwr_dict['clwvi'][:t_idx]
mwr_dict['ta'] = mwr_dict['ta'][:t_idx,:]
mwr_dict['hua'] = mwr_dict['hua'][:t_idx,:]
mwr_dict['time_npdt'] = mwr_dict['time_npdt'][:t_idx]
mwr_dict['time'] = mwr_dict['time'][:t_idx]

mwr_bl_dict['ta'] = mwr_bl_dict['ta'][:t_idx_bl,:]
mwr_bl_dict['time_npdt'] = mwr_bl_dict['time_npdt'][:t_idx_bl]
mwr_bl_dict['time'] = mwr_bl_dict['time'][:t_idx_bl]

# convert to data set:
RET_DS = xr.Dataset({	'ta': (['time', 'height'], mwr_dict['ta']),
						'hua': (['time', 'height'], mwr_dict['hua']),
						'prw': (['time'], mwr_dict['prw']),
						'clwvi': (['time'], mwr_dict['clwvi']),
						'ta_bl': (['time_bl', 'height'], mwr_bl_dict['ta']),
						'time_sec': (['time'], mwr_dict['time']),
						'time_bl_sec': (['time_bl'], mwr_bl_dict['time'])},
					coords=	{'time': (['time'], mwr_dict['time_npdt']),
							'time_bl': (['time_bl'], mwr_bl_dict['time_npdt']),
							'height': (['height'], mwr_dict['height'])})


# visualization:
fs = 22
fs_small = fs - 2

c_H = (0.067,0.29,0.769)	# HATPRO
c_RS = (1,0.435,0)			# radiosondes
c_LWP = (0,0,0,0.55)

dt_fmt = mdates.DateFormatter("%m-%d")


if set_dict['iwv_time_series']:

	# Note HATPRO calibration time:
	# ct_HATPRO = [dt.datetime(2022,7,7,9,52,10)]
	ct_HATPRO = np.array([np.datetime64("2022-07-07T09:52:10")])

	f1, a1 = plt.subplots(1,2)
	f1.set_size_inches(22,10)

	axlim = [0, 40]			# in kg m-2


	# plotting: 
	a1[0].plot(sonde_dict['launch_time_npdt'], sonde_dict['IWV'], linestyle='none', marker='.', linewidth=0.5,
				markersize=15.0, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
				markeredgewidth=0.5, label='Radiosonde')
	a1[0].plot(RET_DS['time_npdt'], RET_DS['prw'], color=c_H, linewidth=1.2, label="HATPRO")


	# visualize calibration times:
	for ct_hatpro in ct_HATPRO:
		if ct_hatpro <= enddate and ct_hatpro >= startdate:
			a1[0].plot([ct_hatpro, ct_hatpro], axlim, color=c_H, linestyle='dashed', linewidth=2)


	# legends and colorbars:
	lh, ll = a1[0].get_legend_handles_labels()
	a1[0].legend(handles=lh, labels=ll, loc='upper left', fontsize=fs, framealpha=0.65, markerscale=1.5)


	# set axis limits:
	a1[0].set_ylim(axlim[0], axlim[1])
	a1[0].set_xlim(startdate, enddate)
	a1[0].xaxis.set_major_formatter(dt_fmt)

	# axis ticks and tick labels:
	a1[0].tick_params(axis='both', labelsize=fs-2)

	# grid
	a1[0].grid(which='major', axis='both')

	# axis labels:
	a1[0].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)


	# adapt size of plot to get calibration legend next to it
	a1_pos = a1[0].get_position().bounds
	a1[0].set_position([a1_pos[0], a1_pos[1]+0.1, 1.6*a1_pos[2], a1_pos[3]*0.9])

	# second axis for enhanced legend:
	a1[1].axis('off')

	a2_pos = a1[1].get_position().bounds
	a1[1].set_position([a2_pos[0] + 0.4*a2_pos[2], a2_pos[1]+0.04, 0.4*a2_pos[2], a2_pos[3]])

	# dummy plot for legend:
	a1[1].plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linestyle='dashed', linewidth=2,
				label="$\\bf{HATPRO}$")
	for ct_hatpro in ct_HATPRO:
		if ct_hatpro <= enddate and ct_hatpro >= startdate:
			a1[1].plot([ct_hatpro, ct_hatpro], axlim, color=f1.get_facecolor(),
						label=numpydatetime64_to_datetime(ct_hatpro).strftime("%Y-%m-%d, %H:%M UTC"))

	# legend:
	clh, cll = a1[1].get_legend_handles_labels()
	yu = a1[1].legend(handles=clh, labels=cll, loc='upper left', fontsize=fs, title="Calibration")
	yu.get_title().set_fontsize(fs)
	yu.get_title().set_fontweight('bold')

	if set_dict['save_figures']:
		plot_name = "WALSEMA_HATPRO_sonde_IWV_time_series"
		f1.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')

	else:
		plt.show()


# visualize comparison of radiosonde and retrieved IWV:
if set_dict['iwv_scatter_plot']:
	# find indices when mwr specific time equals a sonde launch time (+ 15 min). then avg over
	# launchtime:launchtime + 15 min (also compute stddev):
	hatson_idx = np.asarray([np.where((RET_DS.time_sec.values >= lt) & (RET_DS.time_sec.values <= lt+set_dict['launch_window']))[0] for lt in sonde_dict['launch_time']])

	# mean and stddev of IWV around sonde launches:
	RET_DS['prw_mean_sonde'] = xr.DataArray(np.full((n_sondes,), np.nan), dims=['launch_time'], coords={'launch_time': sonde_dict['launch_time_npdt']})
	RET_DS['prw_std_sonde'] = xr.DataArray(np.full((n_sondes,), np.nan), dims=['launch_time'], coords={'launch_time': sonde_dict['launch_time_npdt']})
	for k, hat in enumerate(hatson_idx):
			RET_DS['prw_mean_sonde'][k] = np.nanmean(RET_DS['prw'][hat].values)
			RET_DS['prw_std_sonde'][k] = np.nanstd(RET_DS['prw'].values[hat])


	f1 = plt.figure(figsize=(10,10))
	a1 = plt.axes()

	axlim = np.asarray([0, 35])

	# compute comparison stats:
	ret_stat_dict = compute_retrieval_statistics(sonde_dict['IWV'][13:], RET_DS['prw_mean_sonde'].values[13:],
													compute_stddev=True)


	# plotting:
	a1.errorbar(sonde_dict['IWV'][:13], RET_DS['prw_mean_sonde'].values[:13], yerr=RET_DS['prw_std_sonde'].values[:13],
				ecolor=(0.6,0.6,0.6), elinewidth=1.6, capsize=3, markerfacecolor=(0.6,0.6,0.6), markeredgecolor=(0,0,0),
				linestyle="none", marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='Before calibration')
	a1.errorbar(sonde_dict['IWV'][13:], RET_DS['prw_mean_sonde'].values[13:], yerr=RET_DS['prw_std_sonde'].values[13:],
				ecolor=c_H, elinewidth=1.6, capsize=3, markerfacecolor=c_H, markeredgecolor=(0,0,0),
				linestyle="none", marker='.', markersize=10.0, linewidth=1.2, capthick=1.6, label='After calibration')

	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	nonnan_hatson = np.argwhere(~np.isnan(RET_DS['prw_mean_sonde'].values[13:]) &
						~np.isnan(sonde_dict['IWV'][13:])).flatten()
	y_fit = RET_DS['prw_mean_sonde'].values[13:][nonnan_hatson]
	x_fit = sonde_dict['IWV'][13:][nonnan_hatson]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
		m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
		a = m_fit[0]
		b = m_fit[1]

		ds_fit = a1.plot(axlim, a*axlim + b, color=c_H, linewidth=1.2, label="Best fit")

	# plot a line for orientation which would represent a perfect fit:
	a1.plot(axlim, axlim, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")

	# add statistics:
	a1.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmsd = %.2f \nstd. = %.2f \nR = %.3f"%(ret_stat_dict['N'], 
			np.nanmean(np.concatenate((sonde_dict['IWV'], RET_DS['prw_mean_sonde'].values), axis=0)),
			ret_stat_dict['bias'], ret_stat_dict['rmse'], ret_stat_dict['stddev'], ret_stat_dict['R']),
			horizontalalignment='right', verticalalignment='bottom', transform=a1.transAxes, fontsize=fs_small-2)


	# Legends:
	lh, ll = a1.get_legend_handles_labels()
	a1.legend(handles=lh, labels=ll, loc='upper center', bbox_to_anchor=(0.5, 1.00), fontsize=fs,
				framealpha=0.5)


	# set axis limits:
	a1.set_ylim(axlim[0], axlim[1])
	a1.set_xlim(axlim[0], axlim[1])

	# axis ticks and tick parameters:
	a1.minorticks_on()
	a1.tick_params(axis='both', labelsize=fs_small-2)

	# aspect ratio:
	a1.set_aspect('equal')

	# grid:
	a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# labels:
	a1.set_xlabel("IWV$_{\mathrm{Radiosonde}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	a1.set_ylabel("IWV$_{\mathrm{HATPRO}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if set_dict['save_figures']:
		plot_name = "WALSEMA_HATPRO_sonde_IWV_scatter_plot"
		f1.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
		# f1.savefig(path_plots + plot_name + ".jpg", dpi=400, bbox_inches='tight')

	else:
		plt.show()


# WAI overview:
if set_dict['wai_overview']:
	fs = 14
	fs_small = fs - 2
	fs_dwarf = fs - 4
	marker_size = 15


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


	# reduce the arrays for plotting:
	# # # # # # # # startdate_wai = dt.datetime(2022,7,15,12,0,0)
	# # # # # # # # enddate_wai = dt.datetime(2022,7,19,0,0,0)
	startdate_wai = dt.datetime.strptime(set_dict['date0'], "%Y-%m-%d")
	enddate_wai = dt.datetime.strptime(set_dict['date1'], "%Y-%m-%d")
	startdate_wai_sec = datetime_to_epochtime(startdate_wai)
	enddate_wai_sec = datetime_to_epochtime(enddate_wai)

	RET_DS = RET_DS.sel(time=slice(f"{set_dict['date0']}T00:00:00", f"{set_dict['date1']}T00:00:00"))
	# # # # # # # # # RET_DS = RET_DS.sel(time=slice(f"{startdate_wai:%Y-%m-%d}T00:00:00", f"{enddate_wai:%Y-%m-%d}T00:00:00"))
	RET_DS['ta_rm'] = (['time', 'height'], running_mean_time_2D(RET_DS.ta_combined.values, 300.0, RET_DS.time_sec.values, axis=0))

	wai_idx = np.where((sonde_dict['launch_time'] >= startdate_wai_sec) & (sonde_dict['launch_time'] <= enddate_wai_sec))[0]
	wai_idx = np.insert(wai_idx, 0, wai_idx[0]-1)		# add sonde before startdate; otherwise blank in figure
	wai_idx = np.append(wai_idx, wai_idx[-1]+1)			# add sonde after enddate; otherwise blank in figure
	sonde_dict['launch_time_npdt'] = sonde_dict['launch_time_npdt'][wai_idx]
	sonde_dict['IWV'] = sonde_dict['IWV'][wai_idx]
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
	iwv_lims = [0.0, 40.0]		# kg m-2
	lwp_lims = [0.0, 750.0]		# g m-2
	height_lims = [0, 8000]		# m
	time_lims = [startdate_wai, enddate_wai]

	rho_v_levels = np.arange(0.0, 10.01, 0.5)		# in g m-3
	temp_levels = np.arange(230.0, 295.001, 2)		# in K
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
	ax_iwv.plot(RET_DS.time.values, RET_DS.prw.values, color=c_H, linewidth=1.2)

	# dummy lines for the legend (thicker lines)
	ax_iwv.plot([np.nan, np.nan], [np.nan, np.nan], color=c_H, linewidth=2.0, label='HATPRO IWV')
	ax_iwv.plot([np.nan, np.nan], [np.nan, np.nan], color=c_LWP, linewidth=2.0, label='LWP')


	# LWP axis to read values:
	ax_lwp = ax_iwv.twinx()
	# ax_lwp.plot(RET_DS.time.values, RET_DS.clwvi.values*1000.0, color=c_LWP, linewidth=1.0, zorder=-100.5)


	if set_dict['with_ip']:
		# plot radiosonde humidity profile:
		xv, yv = np.meshgrid(sonde_dict['height'][0,:], sonde_dict['launch_time_npdt'])
		rho_v_rs_curtain = ax_hua_rs.contourf(yv, xv, 1000*sonde_dict['rho_v'], levels=rho_v_levels,
											cmap=rho_v_cmap, extend='max')


		print("Plotting HATPRO humidity profile....")
		# plot hatpro hum profile:
		xv, yv = np.meshgrid(RET_DS.height.values, RET_DS.time.values)
		rho_v_hat_curtain = ax_hua_hat.contourf(yv, xv, 1000*RET_DS.hua.values, levels=rho_v_levels,
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
		xv, yv = np.meshgrid(RET_DS['height'], RET_DS['time'])
		temp_hat_curtain = ax_ta_hat.contourf(yv, xv, RET_DS['ta_rm'], levels=temp_levels,
												cmap=temp_cmap, extend='both')

		# add black contour lines of some temperatures: (only every 500th value to avoid clabel overlap)
		temp_hat_contour = ax_ta_hat.contour(yv[::500,:], xv[::500,:], RET_DS['ta_rm'].values[::500,:] - 273.15, levels=temp_contour_levels,
												colors='black', linewidths=0.9, alpha=0.5)
		ax_ta_hat.clabel(temp_hat_contour, levels=temp_contour_levels, inline=True, fmt="%i$\,^{\circ}$C", 
						colors='black', inline_spacing=12, fontsize=fs_dwarf)

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


	# add lines to highlight radiosonde launch times:
	for lt in sonde_dict['launch_time_npdt']:
		ax_iwv.plot([lt, lt], [iwv_lims[0], iwv_lims[1]], linestyle='dashed', color=(0,0,0,0.5), linewidth=1.0)
		ax_hua_rs.plot([lt, lt], [height_lims[0], height_lims[1]], linestyle='dashed', color=(0,0,0,0.5), linewidth=1.0)
		ax_hua_hat.plot([lt, lt], [height_lims[0], height_lims[1]], linestyle='dashed', color=(0,0,0,0.5), linewidth=1.0)
		ax_ta_rs.plot([lt, lt], [height_lims[0], height_lims[1]], linestyle='dashed', color=(0,0,0,0.5), linewidth=1.0)
		ax_ta_hat.plot([lt, lt], [height_lims[0], height_lims[1]], linestyle='dashed', color=(0,0,0,0.5), linewidth=1.0)


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


	if set_dict['save_figures']:
		# # # # # plot_name = "WALSEMA_HATPRO_sonde_WAI_overview"
		plot_name = f"WALSEMA_HATPRO_sonde_{startdate_wai:%Y%m%d}-{enddate_wai:%Y%m%d}"
		if not set_dict['with_ip']: plot_name += "_no_ip"
		fig1.savefig(path_plots + plot_name + ".png", dpi=400, bbox_inches='tight')
		# fig1.savefig(path_plots + plot_name + ".jpg", dpi=400, bbox_inches='tight')
	else:
		plt.show()