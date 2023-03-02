import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import pdb
import datetime as dt
import os
from import_data import *

"""
	Compare forward simulated Ny Alesund radiosondes (MiRAC-P frequencies) with
	observed MiRAC-P TBs during the MOSAiC campaign using a histogram with
	relative frequency of occurrence.
"""

# specify instrument: 'hatpro' or 'mirac-p':
instrument = 'mirac-p'
which_training = 'ny_alesund'		# choose the training data for mwr_pro MiRAC-P retrieval for plotting the histogram
									# options: 'era_interim', 'ny_alesund'

if instrument == 'hatpro' and which_training == 'era_interim':
	raise ValueError("Instrument 'hatpro' only has training data 'ny_alesund' available.")

log_plot = True
plot_histogram = True
plot_TB_time_series = False
plot_TB_time_series_allinone = False
plot_TB_RPG_and_level1 = False		# plotting TBs of .BRT.NC and level 1 files for instrument = 'mirac-p' only
print_count_low_TBs = False

# Paths:
if instrument == 'mirac-p':
	path_fwd_sim = {'ny_alesund': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
					'era_interim': "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/combined/"} # mirac-p
	path_fwd_sim = path_fwd_sim[which_training]
	path_l1 = "/data/obs/campaigns/mosaic/mirac-p/l1/"						# mirac-p
elif instrument == 'hatpro':
	path_fwd_sim = "/net/aure/kebell/stp_data/nya/hatpro_89/"				# hatpro
	path_l1 = "/data/obs/campaigns/mosaic/hatpro/l1/"						# hatpro
path_plots = "/net/blanc/awalbroe/Plots/"

# prr = [["2020-02-09", "2020-02-19"], ["2020-03-04", "2020-03-15"], ["2020-03-22", "2020-03-25"], ["2020-04-07", "2020-04-15"], ["2020-05-05", "2020-05-09"],
		# ["2020-05-17", "2020-05-25"], ["2020-05-29", "2020-05-31"], ["2020-06-14", "2020-06-15"], ["2020-06-22", "2020-06-23"], ["2020-06-29", "2020-06-29"], 
		# ["2020-09-06", "2020-09-08"], ["2020-09-16", "2020-09-22"], ["2020-09-27", "2020-09-30"]]
# for rrr in prr:
date_start = "2019-09-20"		#rrr[0]		#"2020-02-09"
date_end = "2020-10-12"			#rrr[1]		#"2020-02-19"


# import and concat all fwd_sim files: Use zenith only
files_fwd_sim = sorted(glob.glob(path_fwd_sim + "*.nc"))
for idx, fwd_sim_file in enumerate(files_fwd_sim):
	FWD_SIM_DS = xr.open_dataset(fwd_sim_file)

	# create FWD SIM TB data array if not created yet:
	if idx == 0:
		# elevation angle 90 deg = elevation angle index 0
		TB_FWD_SIM = xr.DataArray(FWD_SIM_DS.brightness_temperatures[:,0,0,:],
									dims=(['n_date', 'n_frequency']))
	else:
		TB_FWD_SIM = xr.concat((TB_FWD_SIM, FWD_SIM_DS.brightness_temperatures[:,0,0,:]),
								dim='n_date')

# add frequency coords:
TB_FWD_SIM = TB_FWD_SIM.assign_coords({'frequency': FWD_SIM_DS.frequency})
TB_FWD_SIM.values[TB_FWD_SIM.values == -99.0] = np.nan


# import all .BRT files and concatenate them:
# do the same for the level 1 mwr_pro files:
date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")
n_days = (date_end - date_start).days + 1

# cycle through all years, all months and days:
time_init_lvl1 = 0	# before initialisation of the data array, this variable is zero; and 1 afterwards
if instrument == 'hatpro':
	file_ending_search = "*_mwr00_*_v01_*.nc"
elif instrument == 'mirac-p':
	file_ending_search = "*_i01_*.nc"

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
	TB_FWD_SIM = TB_FWD_SIM[:,:7]	# limit to K band

if plot_TB_RPG_and_level1:	# load RPG TBs:
	TB_LEVEL_1 = import_mirac_BRT_RPG_daterange(path_l1, dt.datetime.strftime(date_start, "%Y-%m-%d"), 
												dt.datetime.strftime(date_end, "%Y-%m-%d"), verbose=1)
	TB_LEVEL_1['datetime'] = np.asarray([dt.datetime.utcfromtimestamp(trr) for trr in TB_LEVEL_1['time']])
	TB_LEVEL_1['freqs'] = np.array([183.91, 184.81, 185.81, 186.81, 188.31, 190.81, 243, 340])


# plot histogram: one histogram for each frequency:
fs = 10
# colours:
c_fwd = (0.067,0.29,0.769)
c_brt = (0,0.729,0.675)
c_lvl1 = (1,0.435,0)

if plot_histogram:
	fig1, ax1 = plt.subplots(3,3)
	fig1.set_size_inches(12,10)
	ax1 = ax1.flatten()

	if instrument == 'mirac-p':
		x_lim = [[100, 290], [80, 290], [60, 290], [20, 290], [20, 290], [5, 290], [5, 290], [5, 290]]
		y_lim = [[0, 0.08], [0, 0.07], [0, 0.05], [0, 0.035], [0, 0.02], [0, 0.01], [0, 0.016], [0, 0.018]]
	elif instrument == 'hatpro':
		# x_lim = [[5, 160], [5, 160], [5, 160], [5, 160], [5, 160], [5, 160], [5, 160]]
		x_lim = [[5, 50], [5, 50], [5, 50], [5, 50], [5, 50], [5, 50], [5, 50]]
		
	n_freq = len(TB_lvl_1.frequency)

	for k in range(n_freq):
		ax1[k].hist(TB_lvl_1.values[:,k], bins=np.arange(x_lim[k][0],x_lim[k][1]), density=True, color=c_lvl1,
					label='Level 1', alpha=0.5)
		ax1[k].hist(TB_FWD_SIM.values[:,k], bins=np.arange(x_lim[k][0],x_lim[k][1]), density=True, color=c_fwd,
					label='Fwd. sim. %s'%(which_training.replace("_", "-")), alpha=0.5)

		if print_count_low_TBs:
			ax1_ylims = ax1[k].get_ylim()
			ax1_xlims = ax1[k].get_xlim()
			TB_lims = [[0, 208], [0, 189], [0, 150], [0, 121], [0, 90], [0, 59], [0, 33], [0, 89]]
			if ax1_xlims[0] > 0:
				ax1[k].plot([ax1_xlims[0]+1, ax1_xlims[0]+1], [ax1_ylims[0], ax1_ylims[1]], color=(1,0,0), linewidth=1.25, label='TB limit')
			else:
				ax1[k].plot([TB_lims[k][0], TB_lims[k][0]], [ax1_ylims[0], ax1_ylims[1]], color=(1,0,0), linewidth=1.25, label='TB limit')
			ax1[k].plot([TB_lims[k][1], TB_lims[k][1]], [ax1_ylims[0], ax1_ylims[1]], color=(1,0,0), linewidth=1.25)
			ax1[k].set_xlim(ax1_xlims)

		ax1[k].set_title("%.2f GHz"%TB_lvl_1.frequency.values[k], fontsize=fs, pad=0)
		if k >= 6:
			ax1[k].set_xlabel("TB (K)", fontsize=fs)

		# ax1[k].set_ylim(bottom=y_lim[k][0], top=y_lim[k][1])
		if log_plot: ax1[k].set_yscale('log')

	if instrument == 'hatpro': ax1[7].axis('off')
	ax1[8].axis('off')
	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()
	ax1[8].legend(handles=leg_handles, labels=leg_labels, loc='upper center', fontsize=fs+2)

	fig1.suptitle("Relative frequency of TBs", fontsize=fs+2, y=0.94)
	if instrument == 'mirac-p':
		if log_plot:
			fig1.savefig(path_plots + "TB_hist_fwdsim_%s_vs_mirac_obs_log.png"%(which_training.replace("_", "-")), dpi=400)
		else:
			fig1.savefig(path_plots + "TB_hist_fwdsim_%s_vs_mirac_obs.png"%(which_training.replace("_", "-")), dpi=400)
	if instrument == 'hatpro':
		if log_plot:
			fig1.savefig(path_plots + "TB_hist_fwdsim_vs_hatpro_obs_log.png", dpi=400)
		else:
			fig1.savefig(path_plots + "TB_hist_fwdsim_vs_hatpro_obs.png", dpi=400)
	plt.show()
	pdb.set_trace()

	if print_count_low_TBs:
		TB_lims = [[0, 208], [0, 189], [0, 150], [0, 121], [0, 90], [0, 59], [0, 33], [0, 89]]
		for k in range(n_freq):
			n_tbs_fwdsim = np.count_nonzero((TB_FWD_SIM.values[:,k] > TB_lims[k][0]) & (TB_FWD_SIM.values[:,k] <= TB_lims[k][1]))
			n_tbs_obs = np.count_nonzero((TB_lvl_1.values[:,k] > TB_lims[k][0]) & (TB_lvl_1.values[:,k] <= TB_lims[k][1]))
			print("%.2f"%TB_FWD_SIM.frequency[k])
			print("FWD_SIM: %i, %.2f %%"%(n_tbs_fwdsim, 100*n_tbs_fwdsim / len(TB_FWD_SIM.values[:,k])))
			print("OBS: %i, %.2f %%"%(n_tbs_obs, 100*n_tbs_obs / len(TB_lvl_1.values[:,k])))
			print("##### \n")


if plot_TB_time_series:
	fig1, ax1 = plt.subplots(4,1)
	fig1.set_size_inches(12,10)
	ax1 = ax1.flatten()

	n_freq = len(TB_lvl_1.frequency)
	# cmap = plt.cm.get_cmap('Dark2', n_freq)

	for k in range(4):
		ax1[k].plot(TB_lvl_1.time, TB_lvl_1[:,2*k], color=c_lvl1, linewidth=1.2, label="%.2f"%TB_lvl_1.frequency.values[2*k],
					alpha=0.65)
		ax1[k].plot(TB_lvl_1.time, TB_lvl_1[:,2*k+1], color=c_fwd, linewidth=1.2, label="%.2f"%TB_lvl_1.frequency.values[2*k+1],
					alpha=0.65)

		leg_handles, leg_labels = ax1[k].get_legend_handles_labels()
		ax1[k].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs)

		ax1[k].set_ylabel("TB (K)", fontsize=fs)

	fig1.suptitle("TB time series " + dt.datetime.strftime(date_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_end, "%Y-%m-%d"), fontsize=fs+2, y=0.93)

	fig1.savefig(path_plots + "TB_time_series_" + instrument + "_" + dt.datetime.strftime(date_start, "%Y%m%d") +
						"-" + dt.datetime.strftime(date_end, "%Y%m%d") + ".png", dpi=400)


if plot_TB_time_series_allinone:
	fig1, ax1 = plt.subplots(2,1)
	fig1.set_size_inches(12,10)
	ax1 = ax1.flatten()

	n_freq = len(TB_lvl_1.frequency)
	cmap = plt.cm.get_cmap('viridis', n_freq)

	for k in range(6):
		ax1[0].plot(TB_lvl_1.time, TB_lvl_1[:,k], color=cmap(k), linewidth=1.2, label="%.2f"%TB_lvl_1.frequency.values[k],
					alpha=0.65)

	leg_handles, leg_labels = ax1[0].get_legend_handles_labels()
	ax1[0].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-2)
	ax1[0].set_ylabel("TB (K)", fontsize=fs)

	for k in range(6,8):
		ax1[1].plot(TB_lvl_1.time, TB_lvl_1[:,k], color=cmap(k), linewidth=1.2, label="%.2f"%TB_lvl_1.frequency.values[k],
					alpha=0.65)

	leg_handles, leg_labels = ax1[1].get_legend_handles_labels()
	ax1[1].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-2)
	ax1[1].set_ylabel("TB (K)", fontsize=fs)

	fig1.suptitle("TB time series " + dt.datetime.strftime(date_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_end, "%Y-%m-%d"), fontsize=fs+2, y=0.95)
	# fig1.savefig(path_plots + "TB_time_series_AIO_" + instrument + "_" + dt.datetime.strftime(date_start, "%Y%m%d") +
						# "-" + dt.datetime.strftime(date_end, "%Y%m%d") + ".png", dpi=400)
	plt.show()


if plot_TB_RPG_and_level1:
	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(12,9)

	n_freq = len(TB_lvl_1.frequency)
	cmap_blue = plt.cm.get_cmap('Blues', 3+n_freq)
	cmap_reds = plt.cm.get_cmap('Reds', 3+n_freq)

	for k in range(n_freq):
		ax1.plot(TB_lvl_1.time, TB_lvl_1[:,k], color=cmap_blue(k+3), linewidth=1.2, label="lvl 1: %.2f"%TB_lvl_1.frequency.values[k],
					alpha=1.0)
		ax1.plot(TB_LEVEL_1['datetime'], TB_LEVEL_1['TBs'][:,k], color=cmap_reds(k+3), linewidth=1.2,
					label="BRT: %.2f"%TB_LEVEL_1['freqs'][k], alpha=0.2)

	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-3)
	ax1.set_ylabel("TB (K)", fontsize=fs)

	ax1.set_title("TB time series " + dt.datetime.strftime(date_start, "%Y-%m-%d") +
						" - " + dt.datetime.strftime(date_end, "%Y-%m-%d"), fontsize=fs+2, y=0.95)
	plt.show()
	1/0
	fig1.savefig(path_plots + "TB_time_series_AIO_" + instrument + "_" + dt.datetime.strftime(date_start, "%Y%m%d") +
						"-" + dt.datetime.strftime(date_end, "%Y%m%d") + ".png", dpi=400)