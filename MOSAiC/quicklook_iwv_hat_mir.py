import numpy as np
import datetime as dt
# import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import xarray as xr
from import_data import *
from met_tools import *
from data_tools import *
from scipy import stats
# import sys
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit



path_hatpro_level2 = "/data/obs/campaigns/mosaic/hatpro/l2/"
path_mirac_level1 = "/data/obs/campaigns/mosaic/mirac-p/l1/"
path_mirac_level2 = "/data/obs/campaigns/mosaic/mirac-p/l2/"
path_radiosondes = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"
path_radiosondes_TB = "/net/blanc/awalbroe/Data/MOSAiC_radiosondes/fwd_sim/lhumpro_freq/test_011_Ell_10m_S_11_clear_sky/"
path_l1_mirac = "/data/obs/campaigns/mosaic/mirac-p/l1/"						# mirac-p TBs
path_l1_hatpro = "/data/obs/campaigns/mosaic/hatpro/l1/"						# hatpro TBs
path_fwd_sim = "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/"				# mirac-p fwd sim., ny alesund radiosondes
path_plots = "/net/blanc/awalbroe/Plots/"

which_retrievals = 'iwv'
instrument = 'mirac-p'			# can be 'hatpro', 'mirac-p', 'fwd_sim' and 'mirac-p_RPG' (.BRT files)
m_vers = 'i06'					# version of MiRAC-P mwr_pro output products (currently valid: 'i01', 'i02', 'i03')
minute_average = False
include_TBs = False
plot_IWV_time_series = True
plot_TB_vs_IWV = False			# requires include_TBs to be True
TB_vs_IWV_include_RS = False		# includes radiosonde IWV and simulated TBs in TB_vs_IWV plot

# # # rrange = [["2020-02-01", "2020-03-01"], ["2020-03-01", "2020-04-01"], ["2020-04-01", "2020-05-01"], ["2020-05-01", "2020-06-01"], ["2020-06-01", "2020-07-01"],
			# # # ["2020-07-01", "2020-08-01"], ["2020-08-01", "2020-09-01"], ["2020-09-01", "2020-10-01"]]

# # # for rr in rrange:
considered_period = 'user'
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					# # # 'user': [rr[0], rr[1]]}
					'user': ["2020-03-01", "2020-04-01"]}
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"


# HATPRO v01:
hatpro_dict = import_hatpro_level2a_daterange(path_hatpro_level2, date_start, date_end, 
												which_retrieval=which_retrievals, minute_avg=minute_average, verbose=1)

# Std RPG MiRAC:
mirac_dict = import_mirac_IWV_LWP_RPG_daterange(path_mirac_level1, date_start, date_end, which_retrieval=which_retrievals,
					minute_avg=minute_average, verbose=1)

# new MiRAC (i01, i02):
mirac_dict_new = import_mirac_level2a_daterange(path_mirac_level2, date_start, date_end, which_retrieval=which_retrievals,
												vers=m_vers, minute_avg=minute_average, verbose=1)

# radiosondes:
sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version='level_2', verbose=1)

# Time to datetime:
hatpro_dict['datetime'] = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in hatpro_dict['time']])
mirac_dict['datetime'] = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in mirac_dict['time']])
mirac_dict_new['datetime'] = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in mirac_dict_new['time']])
sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])


if include_TBs:		# to compare TBs to IWV, import MiRAC-P TBs

	if instrument == 'mirac-p_RPG':
		# Old MiRAC-P TBs (directly from RPG): .BRT files:
		TB_LEVEL_1 = import_mirac_BRT_RPG_daterange(path_mirac_level1, date_start, date_end, verbose=1)

	elif instrument == 'fwd_sim':
		# import and concat all fwd_sim files: Use zenith only
		files_fwd_sim = sorted(glob.glob(path_fwd_sim + "*.nc"))
		for idx, fwd_sim_file in enumerate(files_fwd_sim):
			FWD_SIM_DS = xr.open_dataset(fwd_sim_file)

			# create FWD SIM TB data array if not created yet:
			# same with FWD SIM IWV:
			if idx == 0:
				# elevation angle 90 deg = elevation angle index 0
				TB_FWD_SIM = xr.DataArray(FWD_SIM_DS.brightness_temperatures[:,0,0,:],
											dims=(['n_date', 'n_frequency']))
				IWV_FWD_SIM = xr.DataArray(FWD_SIM_DS.integrated_water_vapor, dims=(['n_date']))
			else:
				TB_FWD_SIM = xr.concat((TB_FWD_SIM, FWD_SIM_DS.brightness_temperatures[:,0,0,:]),
										dim='n_date')
				IWV_FWD_SIM = xr.concat((IWV_FWD_SIM, FWD_SIM_DS.integrated_water_vapor), dim='n_date')

		# add frequency coords:
		TB_FWD_SIM = TB_FWD_SIM.assign_coords({'frequency': FWD_SIM_DS.frequency})
		TB_FWD_SIM.values[TB_FWD_SIM.values == -99.0] = np.nan
		IWV_FWD_SIM.values[IWV_FWD_SIM.values < 0] = np.nan		

	else:
		# import all level 1 files and concatenate them:
		date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
		date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")
		n_days = (date_end - date_start).days + 1

		# cycle through all years, all months and days:
		time_init_lvl1 = 0	# before initialisation of the data array, this variable is zero; and 1 afterwards
		if instrument == 'hatpro':
			path_l1 = path_l1_hatpro
			file_ending_search = "*_mwr00_*_v01_*.nc"
		elif instrument == 'mirac-p':
			path_l1 = path_l1_mirac
			file_ending_search = "*_%s_*.nc"%m_vers

		for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

			print("Level 1, ", now_date)
			yyyy = now_date.year
			mm = now_date.month
			dd = now_date.day

			day_path = path_l1 + "%04i/%02i/%02i/"%(yyyy,mm,dd)
			if not os.path.exists(os.path.dirname(day_path)):
				continue

			# list of files:
			level_1_nc = sorted(glob.glob(day_path + file_ending_search))
			if len(level_1_nc) == 0:
				continue

			for lvl_1 in level_1_nc:
				TB_LEVEL_1_DS = xr.open_dataset(lvl_1)

				if time_init_lvl1 == 0:
					TB_lvl_1 = TB_LEVEL_1_DS.tb
					time_init_lvl1 = 1
				else:
					TB_lvl_1 = xr.concat((TB_lvl_1, TB_LEVEL_1_DS.tb), dim='time')


		# add frequency coords:
		TB_lvl_1 = TB_lvl_1.assign_coords({'frequency': TB_LEVEL_1_DS.freq_sb})

		if instrument == 'hatpro': # limit the number of frequencies to K band only
			TB_lvl_1 = TB_lvl_1[:,:7]		# limit to K band

	if TB_vs_IWV_include_RS:
		all_out_files = sorted(glob.glob(path_radiosondes_TB + "*.nc"))
		
		# Concatenate all output files along dimension x of PAMTRA output files:

		SONDE_FWD_SIM_DS = xr.open_mfdataset(all_out_files, concat_dim='grid_x', combine='nested',
										preprocess=pam_out_drop_useless_dims)

		# Apply double side band average on SONDE_FWD_SIM_DS TBs:
		SONDE_FWD_SIM_DS['tb_dsba'] = xr.DataArray(Gband_double_side_band_average(SONDE_FWD_SIM_DS.tb.values,
																			SONDE_FWD_SIM_DS.frequency.values)[0],
											coords={'grid_x': SONDE_FWD_SIM_DS.grid_x,
													'frequency_dsba': np.array([183.91, 184.81, 185.81, 186.81,
																				188.31, 190.81, 243., 340.])},
											dims=['grid_x', 'frequency_dsba'])

		# interpolate sonde_dict IWV on the simulated TB time axis (to quickly filter the sondes):
		sonde_IWV_clear_sky = np.interp(SONDE_FWD_SIM_DS.time.values, sonde_dict['launch_time'], sonde_dict['iwv'])



# convert date_end and date_start to datetime:
if isinstance(date_end, str):
	date_range_end = dt.datetime.strptime(date_end, "%Y-%m-%d") + dt.timedelta(days=1)
	date_range_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
else:
	date_range_end = date_end + dt.timedelta(days=1)
	date_range_start = date_start

# colors:
c_H = (0.067,0.29,0.769)	# HATPRO
c_M = (0,0.729,0.675)		# MiRAC-P
c_M2 = (1,0.435,0)			# New MiRAC-P: Level 2a
c_RS = (1,0.663,0)			# Radiosondes

# dt_fmt = mdates.DateFormatter("%Y-%m-%d") # ("%Y-%m-%d")
dt_fmt = mdates.DateFormatter("%b %d") # (e.g. "Feb 23")
datetick_auto = False
fs = 19		# fontsize

# create x_ticks depending on the date range: roughly 20 x_ticks are planned
# round off to days if number of days > 15:
date_range_delta = (date_range_end - date_range_start)
if date_range_delta < dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=6)
	# dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta < dt.timedelta(days=11) and date_range_delta >= dt.timedelta(days=6):
	x_tick_delta = dt.timedelta(hours=12)
	# dt_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")
	dt_fmt = mdates.DateFormatter("%b %d %H:%M")
elif date_range_delta >= dt.timedelta(days=11) and date_range_delta < dt.timedelta(21):
	x_tick_delta = dt.timedelta(days=1)
else:
	x_tick_delta = dt.timedelta(days=(date_range_delta/20).days)
	# x_tick_delta = dt.timedelta(days=2)

x_ticks_dt = mdates.drange(date_range_start, date_range_end, x_tick_delta)	# alternative if the xticklabel is centered
# x_ticks_dt = mdates.drange(date_range_start + x_tick_delta, date_range_end + x_tick_delta, x_tick_delta)


if plot_IWV_time_series:
	# IWV time series MiRAC-P, HATPRO and radiosonde
	fig1, ax1 = plt.subplots(1,2)
	fig1.set_size_inches(22,10)

	iwv_axlim = [0, 30]		# axis limits for IWV plot in kg m^-2
	MIRAC_IWV_plot = ax1[0].plot(mirac_dict['datetime'][mirac_dict['RF'] == 0], mirac_dict['IWV'][mirac_dict['RF'] == 0],
								color=c_M, linewidth=1.0, label="RPG: MiRAC-P")
	HATPRO_IWV_plot = ax1[0].plot(hatpro_dict['datetime'][hatpro_dict['flag'] == 0], hatpro_dict['prw'][hatpro_dict['flag'] == 0], 
								color=c_H, linewidth=1.0, label='Level 2a: HATPRO')
	MIRAC_NEW_IWV_plot = ax1[0].plot(mirac_dict_new['datetime'], mirac_dict_new['prw'],
								color=c_M2, linewidth=1.0, alpha=0.6, label='Level 2a: MiRAC-P')
	SONDE_IWV_plot = ax1[0].plot(sonde_dict['launch_time_dt'], sonde_dict['iwv'], linestyle='none', 
											marker='.', linewidth=0.5,
											markersize=7.5, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
											markeredgewidth=0.5, label='Radiosonde')


	# legend:
	iwv_leg_handles, iwv_leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=iwv_leg_handles, labels=iwv_leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	# set axis limits and labels:
	ax1[0].set_ylim(bottom=iwv_axlim[0], top=iwv_axlim[1])
	ax1[0].set_xlim(left=date_range_start, right=date_range_end)
	ax1[0].xaxis.set_major_formatter(dt_fmt)
	ax1[0].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
	the_title = ax1[0].set_title("Integrated Water Vapour (IWV), " + dt.datetime.strftime(date_range_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_range_end, "%Y-%m-%d"), fontsize=fs)
	the_title.set_position((0.5, 1.05))

	if datetick_auto:
		fig1.autofmt_xdate()
	else:
		ax1[0].set_xticks(x_ticks_dt)
		ax1[0].tick_params(axis='x', labelsize=fs-3, labelrotation=90)

	ax1[0].tick_params(axis='y', labelsize=fs-2)

	ax1[0].grid(which='major', axis='both')


	ax1_pos = ax1[0].get_position().bounds
	ax1[0].set_position([ax1_pos[0], ax1_pos[1]+0.1, 1.6*ax1_pos[2], ax1_pos[3]*0.9])

	# create the text box like Patrick Konjari:
	ax1[1].axis('off')

	ax2_pos = ax1[1].get_position().bounds
	ax1[1].set_position([ax2_pos[0] + 0.4*ax2_pos[2], ax2_pos[1]+0.04, 0.4*ax2_pos[2], ax2_pos[3]])

	fig1.savefig(path_plots + "IWV_time_series_with_new_MiRAC-P_ret_%s_%s-%s.png"%(m_vers,
					dt.datetime.strftime(date_range_start, "%Y%m%d"), dt.datetime.strftime(date_range_end, "%Y%m%d")), dpi=400)
	plt.show()


if plot_TB_vs_IWV:

	if instrument == 'mirac-p_RPG':
		# because the time sampling of the .IWV.NC and .BRT.NC files apparently isn't identical,
		# the overlapping times have to be found manually via interpolation on the smaller time
		# axis:
		n_iwv_rpg = len(mirac_dict['time'])
		n_brt_rpg = len(TB_LEVEL_1['time'])
		if (mirac_dict['time'][0] == TB_LEVEL_1['time'][0]) and (mirac_dict['time'][-1] == TB_LEVEL_1['time'][-1]):
			# no interpolation is required if the start and end time and the number of entries are identical

			if n_iwv_rpg > n_brt_rpg:	# in this case, interpolate to brt time axis:
				mirac_time_old = mirac_dict['time']
				for key in mirac_dict.keys():
					mirac_dict[key] = np.interp(TB_LEVEL_1['time'], mirac_time_old, mirac_dict[key])
			
			elif n_iwv_rpg < n_brt_rpg:	# in this case, interpolate to IWV time axis:
				mirac_time_old = TB_LEVEL_1['time']
				for key in TB_LEVEL_1.keys():
					if key == 'TBs':
						for k in range(len(TB_LEVEL_1[key][0,:])):
							TB_LEVEL_1[key][:len(mirac_dict['time']),k] = np.interp(mirac_dict['time'], mirac_time_old, TB_LEVEL_1[key][:,k])
						TB_LEVEL_1[key] = TB_LEVEL_1[key][:len(mirac_dict['time']),:]
					elif key == 'Freq':
						continue
					else:
						TB_LEVEL_1[key] = np.interp(mirac_dict['time'], mirac_time_old, TB_LEVEL_1[key])

		n_freq = len(TB_LEVEL_1['TBs'][0,:])

		# flag:
		flag_0 = np.where(TB_LEVEL_1['RF'] == 0)[0]

	elif instrument == 'fwd_sim':
		n_freq = len(TB_FWD_SIM.frequency)

	else:
		n_freq = len(TB_lvl_1.frequency)
		if instrument == 'mirac-p': flag_16 = np.where(mirac_dict_new['flag'] == 16)[0]
		if instrument == 'hatpro': flag_0 = np.where(hatpro_dict['flag'] == 0)[0]
			

	for k in range(n_freq):
		print(k)
		fig1, ax1 = plt.subplots(1,1)
		fig1.set_size_inches(11,11)

		iwv_lims = [0, 30]
		TB_lims = [0, 300]

		if instrument == 'hatpro':
			ax1.plot(TB_lvl_1.values[flag_0,k], hatpro_dict['prw'][flag_0], linestyle='none', marker='+', color=(0,0,0),
								markeredgecolor=(0,0,0), markersize=1.0, alpha=0.65)

		elif instrument == 'mirac-p':
			ax1.plot(TB_lvl_1.values[flag_16,k], mirac_dict_new['prw'][flag_16], linestyle='none', marker='+', color=(0,0,0),
						markeredgecolor=(0,0,0), markersize=1.0, alpha=0.65)

			if TB_vs_IWV_include_RS:
				ax1.plot(SONDE_FWD_SIM_DS.tb_dsba.values[:,k], sonde_IWV_clear_sky, linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1)

				# Dummy plots for legend:
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0,0,0),
								markeredgecolor=(0,0,0), markersize=6.5, alpha=0.65, label="MiRAC-P")
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1, label="Radiosonde")
				l_handles, l_labels = ax1.get_legend_handles_labels()
				ax1.legend(handles=l_handles, labels=l_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

		elif instrument == 'mirac-p_RPG':
			ax1.plot(TB_LEVEL_1['TBs'][flag_0,k], mirac_dict['IWV'][flag_0], linestyle='none', marker='+', color=(0,0,0),
						markeredgecolor=(0,0,0), markersize=1.0, alpha=0.65)
			freqs = [183.91, 184.81, 185.81, 186.81, 188.31, 190.81, 243, 340]


			if TB_vs_IWV_include_RS:
				ax1.plot(SONDE_FWD_SIM_DS.tb_dsba.values[:,k], sonde_IWV_clear_sky, linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1)

				# Dummy plots for legend:
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0,0,0),
								markeredgecolor=(0,0,0), markersize=6.5, alpha=0.65, label="MiRAC-P RPG")
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1, label="Radiosonde")
				l_handles, l_labels = ax1.get_legend_handles_labels()
				ax1.legend(handles=l_handles, labels=l_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

			ax1.set_title("Integrated Water Vapour (IWV) against %s Brightness Temperatures \n"%instrument +
							"during MOSAiC (TB). Channel: %.2f GHz"%freqs[k], fontsize=fs, pad=0)

		elif instrument == 'fwd_sim':
			ax1.plot(TB_FWD_SIM.values[:,k], IWV_FWD_SIM.values, linestyle='none', marker='+', color=(0,0,0),
						markeredgecolor=(0,0,0), markersize=3.0, alpha=0.65)


			if TB_vs_IWV_include_RS:
				ax1.plot(SONDE_FWD_SIM_DS.tb_dsba.values[:,k], sonde_IWV_clear_sky, linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1)

				# Dummy plots for legend:
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0,0,0),
								markeredgecolor=(0,0,0), markersize=6.5, alpha=0.65, label="Ny Alesund Radiosonde")
				ax1.plot([np.nan, np.nan], [np.nan, np.nan], linestyle='none', marker='+', color=(0.73,0.33,0.4),
							markeredgecolor=(0.73,0.33,0.4), markersize=6.5, alpha=1, label="MOSAiC Radiosonde")
				l_handles, l_labels = ax1.get_legend_handles_labels()
				ax1.legend(handles=l_handles, labels=l_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)


			ax1.set_title("Integrated Water Vapour (IWV) against %s Brightness Temperatures \n"%instrument +
							"during MOSAiC (TB). Channel: %.2f GHz"%TB_FWD_SIM.frequency.values[k], fontsize=fs, pad=0)


		if instrument in ['mirac-p', 'hatpro']:
			ax1.set_title("Integrated Water Vapour (IWV) against %s Brightness Temperatures \n"%instrument +
							"during MOSAiC (TB). Channel: %.2f GHz"%TB_lvl_1.frequency.values[k], fontsize=fs, pad=0)
		ax1.set_xlabel("TB (K)", fontsize=fs, labelpad=0.5)
		ax1.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax1.minorticks_on()
		ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax1.tick_params(axis='both', labelsize=fs-2)

		ax1_pos = ax1.get_position().bounds
		ax1.set_position([ax1_pos[0], ax1_pos[1] + 0.05*ax1_pos[3], ax1_pos[2], ax1_pos[3]*0.95])

		if instrument == 'mirac-p':
			fig1.savefig(path_plots + "IWV_vs_TB_channel_%02i_%s_%s.png"%(k, instrument, m_vers), dpi=400)
		else:
			fig1.savefig(path_plots + "IWV_vs_TB_channel_%02i_%s.png"%(k, instrument), dpi=400)

# pdb.set_trace()