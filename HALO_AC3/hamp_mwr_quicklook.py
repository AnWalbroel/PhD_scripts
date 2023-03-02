import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import sys
import pdb
import os
import datetime as dt
from geopy import distance
from halo_classes import MWR, BAHAMAS

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')
from data_tools import select_MWR_channels, numpydatetime64_to_datetime, compute_retrieval_statistics


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
	- Load MWR data
	- Load BAHAMAS data
	- Plot quicklook overview
"""


path_data = {'mwr': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/",				# base path of HALO HAMP data
			'dropsondes': "/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/pam_out_sonde_ocean/",	# base path of fwd sim dropsonde data
			'bahamas': "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/"}	# BAHAMAS data
path_output = "/net/blanc/awalbroe/Plots/HALO_AC3_quicklooks/"
save_figures = False		# if True, saved to PDF at path_output
with_dropsondes = False	# if True, forward sim. dropsondes (created with ./fwd_sim_dropsondes/) are also plotted
time_series = True		# simple TB time series
scatter_plot = False		# scatterplot comparing HAMP TBs with those simulated from dropsondes (only possible if with_dropsondes=True)

if not with_dropsondes: scatter_plot = False

dates = [	# must be in yyyymmdd
			# "20220225",
			# "20220311",
			# "20220312",
			# "20220313",
			# "20220314",
			"20220315",
			# "20220316",
			# "20220320",
			# "20220321",
			# "20220328",
			# "20220329",
			# "20220330",
			# "20220401",
			# "20220404",
			# "20220407",
			# "20220408",
			# "20220410",
			# "20220411",
			# "20220412",
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
	HAMP_MWR = MWR(path_data['mwr'], which_date)

	if with_dropsondes:
		# import dropsondes:
		HALO_dropsondes = MWR(path_data['dropsondes'], which_date, version="synthetic_dropsonde")

	# import BAHAMAS data:
	bah = BAHAMAS(path_data['bahamas'], which_date, return_DS=True)


	# some aux info read out of the data:
	n_freq_kv = len(HAMP_MWR.freq['KV'])
	n_freq_11990 = len(HAMP_MWR.freq['11990'])
	n_freq_183 = len(HAMP_MWR.freq['183'])
	n_freq_k = 7
	n_freq_v = 7

	# time extent:
	time_extent = numpydatetime64_to_datetime(bah.time[-1]) - numpydatetime64_to_datetime(bah.time[0])
	# # # # # time_0 = dt.datetime(2022,3,14,14,41,29)
	# # # # # time_1 = dt.datetime(2022,3,14,15,18,0)
	# # # # # time_extent = time_1 - time_0


	# Plotting:
	# For each receiver, one plot pdf (or at least: KV = plot 1, 11990 + 183 = plot 2

	fs = 14
	fs_small = fs - 2
	fs_dwarf = fs - 4
	marker_size = 15


	# colors:
	cmap_kv = mpl.cm.get_cmap('tab10', n_freq_kv)
	cmap_11990 = mpl.cm.get_cmap('tab10', n_freq_11990)
	cmap_183 = mpl.cm.get_cmap('tab10', n_freq_183)
	cmap_k = mpl.cm.get_cmap('tab10', n_freq_k)
	cmap_v = mpl.cm.get_cmap('tab10', n_freq_v)

	import locale
	dt_fmt = mdates.DateFormatter("%H:%M") # (e.g. "12:00")
	datetick_auto = True


	if time_series:

		# Plot: first option: quite combined (K, V, 11990, 183) <- four plots
		fig, ax = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

		# ax lims:
		time_lims = [bah.time_npdt[0], bah.time_npdt[-1]]
		# # # # # # # # # # # time_lims = [np.datetime64("2022-03-14T14:41:29"), np.datetime64("2022-03-14T15:18:00")]
		tb_lims_k = [120, 280]
		# tb_lims_k = [120, 190]
		# tb_lims_v = [180, 300]
		tb_lims_v = [200, 280]
		# tb_lims_wf = [160, 280]
		tb_lims_wf = [170, 270]
		# tb_lims_g = [210, 290]
		tb_lims_g = [210, 280]


		# plotting:
		# k band:
		if HAMP_MWR.avail['KV']:
			freq_idx = select_MWR_channels(HAMP_MWR.TB['KV'], HAMP_MWR.freq['KV'], band='K', return_idx=2)
			for k in range(n_freq_k):
				ax[0].scatter(HAMP_MWR.time_npdt['KV'], HAMP_MWR.TB['KV'][:,freq_idx[k]], s=1, color=cmap_k(k), linewidth=0,
								label=f"{HAMP_MWR.freq['KV'][freq_idx[k]]:.2f} GHz")

		# v band:		
		if HAMP_MWR.avail['KV']:
			freq_idx = select_MWR_channels(HAMP_MWR.TB['KV'], HAMP_MWR.freq['KV'], band='V', return_idx=2)
			for k in range(n_freq_v):
				ax[1].scatter(HAMP_MWR.time_npdt['KV'], HAMP_MWR.TB['KV'][:,freq_idx[k]], s=1, color=cmap_v(k), linewidth=0,
								label=f"{HAMP_MWR.freq['KV'][freq_idx[k]]:.2f} GHz")

		# w and f band:
		if HAMP_MWR.avail['11990']:
			freq_idx = select_MWR_channels(HAMP_MWR.TB['11990'], HAMP_MWR.freq['11990'], band='W+F', return_idx=2)
			for k in range(n_freq_11990):
				ax[2].scatter(HAMP_MWR.time_npdt['11990'], HAMP_MWR.TB['11990'][:,freq_idx[k]], s=1, color=cmap_11990(k), linewidth=0,
								label=f"{HAMP_MWR.freq['11990'][freq_idx[k]]:.2f} GHz")

		# g band:
		if HAMP_MWR.avail['183']:
			freq_idx = select_MWR_channels(HAMP_MWR.TB['183'], HAMP_MWR.freq['183'], band='G', return_idx=2)
			for k in range(n_freq_183):
				ax[3].scatter(HAMP_MWR.time_npdt['183'], HAMP_MWR.TB['183'][:,freq_idx[k]], s=1, color=cmap_183(k), linewidth=0,
								label=f"{HAMP_MWR.freq['183'][freq_idx[k]]:.2f} GHz")

		# plot dropsondes eventually:
		if with_dropsondes:
			freq_idx = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='K', return_idx=2)
			for k in range(n_freq_k):
				ax[0].plot(HALO_dropsondes.time_npdt['dropsonde'], HALO_dropsondes.TB['dropsonde'][:,freq_idx[k]], marker='o', color=cmap_k(k), 
								markeredgecolor=(0,0,0), linestyle="none")

			for launch in HALO_dropsondes.time_npdt['dropsonde']:
				ax[0].plot(launch, tb_lims_k[1], linestyle='none', marker="v", color=(0,0,0), markersize=9)
				ax[0].plot([launch, launch], [tb_lims_k[0], tb_lims_k[1]], linestyle='dashed', color=(0,0,0), linewidth=1.0)

			freq_idx = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='V', return_idx=2)
			for k in range(n_freq_v):
				ax[1].plot(HALO_dropsondes.time_npdt['dropsonde'], HALO_dropsondes.TB['dropsonde'][:,freq_idx[k]], marker='o', color=cmap_v(k), 
							markeredgecolor=(0,0,0), linestyle="none")

			for launch in HALO_dropsondes.time_npdt['dropsonde']:
				ax[1].plot(launch, tb_lims_v[1], linestyle='none', marker="v", color=(0,0,0), markersize=9)
				ax[1].plot([launch, launch], [tb_lims_v[0], tb_lims_v[1]], linestyle='dashed', color=(0,0,0), linewidth=1.0)

			freq_idx = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='W+F', return_idx=2)
			for k in range(n_freq_11990):
				ax[2].plot(HALO_dropsondes.time_npdt['dropsonde'], HALO_dropsondes.TB['dropsonde'][:,freq_idx[k]], marker='o', color=cmap_11990(k),
							markeredgecolor=(0,0,0), linestyle="none")

			for launch in HALO_dropsondes.time_npdt['dropsonde']:
				ax[2].plot(launch, tb_lims_wf[1], linestyle='none', marker="v", color=(0,0,0), markersize=9)
				ax[2].plot([launch, launch], [tb_lims_wf[0], tb_lims_wf[1]], linestyle='dashed', color=(0,0,0), linewidth=1.0)

			freq_idx = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='G', return_idx=2)
			for k in range(n_freq_183):
				ax[3].plot(HALO_dropsondes.time_npdt['dropsonde'], HALO_dropsondes.TB['dropsonde'][:,freq_idx[k]], marker='o', color=cmap_183(k), 
							markeredgecolor=(0,0,0), linestyle="none")

			for launch in HALO_dropsondes.time_npdt['dropsonde']:
				ax[3].plot(launch, tb_lims_g[1], linestyle='none', marker="v", color=(0,0,0), markersize=9)
				ax[3].plot([launch, launch], [tb_lims_g[0], tb_lims_g[1]], linestyle='dashed', color=(0,0,0), linewidth=1.0)


		# add figure identifier of subplots: a), b), ...
		ax[0].text(0.02, 0.95, f"a) HALO-AC3_HALO_hamp_K_{which_date}_{RF_now}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax[0].transAxes)
		ax[1].text(0.02, 0.95, f"b) HALO-AC3_HALO_hamp_V_{which_date}_{RF_now}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax[1].transAxes)
		ax[2].text(0.02, 0.95, f"c) HALO-AC3_HALO_hamp_11990_{which_date}_{RF_now}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax[2].transAxes)
		ax[3].text(0.02, 0.95, f"d) HALO-AC3_HALO_hamp_183_{which_date}_{RF_now}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax[3].transAxes)


		# legends or colorbars:
		lh, ll = ax[0].get_legend_handles_labels()
		lg = ax[0].legend(handles=lh, labels=ll, loc="lower center", bbox_to_anchor=(0.5,1), ncol=4, handletextpad=0,
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_k(k))

		lh, ll = ax[1].get_legend_handles_labels()
		lg = ax[1].legend(handles=lh, labels=ll, loc="lower center", bbox_to_anchor=(0.5,1), ncol=4, handletextpad=0,
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_v(k))

		lh, ll = ax[2].get_legend_handles_labels()
		lg = ax[2].legend(handles=lh, labels=ll, loc="lower center", bbox_to_anchor=(0.5,1), ncol=3, handletextpad=0,
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_11990(k))

		lh, ll = ax[3].get_legend_handles_labels()
		lg = ax[3].legend(handles=lh, labels=ll, loc="lower center", bbox_to_anchor=(0.5,1), ncol=3, handletextpad=0,
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_183(k))


		# set axis limits:
		for axx in ax:
			axx.set_xlim(left=time_lims[0], right=time_lims[1])

		ax[0].set_ylim(bottom=tb_lims_k[0], top=tb_lims_k[1])
		ax[1].set_ylim(bottom=tb_lims_v[0], top=tb_lims_v[1])
		ax[2].set_ylim(bottom=tb_lims_wf[0], top=tb_lims_wf[1])
		ax[3].set_ylim(bottom=tb_lims_g[0], top=tb_lims_g[1])


		# set x ticks and tick labels:
		for axx in ax: axx = set_xticks_and_xlabels(ax=axx, time_extend=time_extent)


		# x tick parameters:
		ax[3].tick_params(axis='x', labelsize=fs_small)

		# ytick parameters and grid and y labels:
		for axx in ax:
			axx.tick_params(axis='y', labelsize=fs_small)
			axx.grid(which='major', axis='both', alpha=0.4)
			axx.set_ylabel("TB (K)", fontsize=fs)

		# set labels:
		ax[3].set_xlabel(f"Time (HH:MM) of {str(bah.time_npdt[round(len(bah.time)/2)])[:10]}", fontsize=fs)


		# Limit axis spacing:
		plt.subplots_adjust(hspace=0.35)			# removes space between subplots

		if not os.path.exists(path_output):
			os.makedirs(path_output)

		if save_figures:
			plot_name = f"HALO-AC3_HALO_hamp_radiometer_{which_date}_{RF_now}.png"
			fig.savefig(path_output + plot_name, dpi=400, bbox_inches='tight')
			print(f"Plot saved to {path_output + plot_name}.")
		else:
			plt.show()


	if scatter_plot:

		# Plot: first option: quite combined (K, V, 11990, 183) <- four plots
		fig, ax = plt.subplots(2, 2, figsize=(10, 10))

		# ax lims:
		tb_lims_k = [120, 200]
		tb_lims_v = [180, 280]
		tb_lims_wf = [160, 280]
		tb_lims_g = [200, 280]


		# plotting:
		# reducing to time steps when 23.84 GHz showed < 180 K:
		time_idx_kv = np.where(HAMP_MWR.TB['KV'][:,2] < 180)[0]
		time_kv = HAMP_MWR.time['KV'][time_idx_kv]
		for hamp_key in ['KV', '11990', '183']:
			if hamp_key == 'KV':
				HAMP_MWR.time[hamp_key] = HAMP_MWR.time[hamp_key][time_idx_kv]
				HAMP_MWR.TB[hamp_key] = HAMP_MWR.TB[hamp_key][time_idx_kv,:]
				HAMP_MWR.flag[hamp_key] = HAMP_MWR.flag[hamp_key][time_idx_kv]
			else:
				time_old = HAMP_MWR.time[hamp_key]
				HAMP_MWR.time[hamp_key] = np.interp(HAMP_MWR.time['KV'], time_old, HAMP_MWR.time[hamp_key])
				new_TB = np.full((HAMP_MWR.TB['KV'].shape[0], HAMP_MWR.TB[hamp_key].shape[1]), np.nan)
				for fff in range(HAMP_MWR.TB[hamp_key].shape[1]):
					new_TB[:,fff] = np.interp(HAMP_MWR.time['KV'], time_old, HAMP_MWR.TB[hamp_key][:,fff])
				HAMP_MWR.TB[hamp_key] = new_TB
				HAMP_MWR.flag[hamp_key] = np.interp(HAMP_MWR.time['KV'], time_old, HAMP_MWR.flag[hamp_key])


		n_sondes = len(HALO_dropsondes.time['dropsonde'])
		HAMP_MWR.TB_mean = {'KV': np.full((n_sondes, n_freq_kv), np.nan),
							'11990': np.full((n_sondes, n_freq_11990), np.nan),
							'183': np.full((n_sondes, n_freq_183), np.nan)}
		HAMP_MWR.TB_std = {'KV': np.full((n_sondes, n_freq_kv), np.nan),
							'11990': np.full((n_sondes, n_freq_11990), np.nan),
							'183': np.full((n_sondes, n_freq_183), np.nan)}

		# k and v band:
		if HAMP_MWR.avail['KV']:
			# filter +/-15 sec around dropsonde launches:
			mwrson_idx = np.asarray([np.where((HAMP_MWR.time['KV'] >= lt - 15) &
									(HAMP_MWR.time['KV'] <= lt + 15))[0] for lt in HALO_dropsondes.time['dropsonde']])
			for kkk, msi in enumerate(mwrson_idx):
				HAMP_MWR.TB_mean['KV'][kkk,:] = np.nanmean(HAMP_MWR.TB['KV'][msi,:], axis=0)
				HAMP_MWR.TB_std['KV'][kkk,:] = np.nanstd(HAMP_MWR.TB['KV'][msi,:], axis=0)

			freq_idx = select_MWR_channels(HAMP_MWR.TB['KV'], HAMP_MWR.freq['KV'], band='K', return_idx=2)
			freq_idx_ds = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='K', return_idx=2)
			for k in range(n_freq_k):
				ax[0,0].errorbar(HALO_dropsondes.TB['dropsonde'][:,freq_idx_ds[k]], HAMP_MWR.TB_mean['KV'][:, freq_idx[k]],
								yerr=HAMP_MWR.TB_std['KV'][:, freq_idx[k]],
								ecolor=cmap_k(k), elinewidth=1.25, capsize=3, markerfacecolor=cmap_k(k), markeredgecolor=(0,0,0),
								linestyle='none', marker='.', markersize=14.0, linewidth=1.2, capthick=1.6,
								label=f"{HAMP_MWR.freq['KV'][freq_idx[k]]:.2f} GHz")


			# add main diagonal for orientation:
			ax[0,0].plot(tb_lims_k, tb_lims_k, color=(0,0,0,0.33), linewidth=1.0)

			# add statistics:
			ax[0,0].text(0.99, 0.01, f"N = {n_sondes}",
					horizontalalignment='right', verticalalignment='bottom', transform=ax[0,0].transAxes, fontsize=fs-2)


			# v band:
			freq_idx = select_MWR_channels(HAMP_MWR.TB['KV'], HAMP_MWR.freq['KV'], band='V', return_idx=2)
			freq_idx_ds = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='V', return_idx=2)
			for k in range(n_freq_v):
				ax[0,1].errorbar(HALO_dropsondes.TB['dropsonde'][:,freq_idx_ds[k]], HAMP_MWR.TB_mean['KV'][:, freq_idx[k]],
								yerr=HAMP_MWR.TB_std['KV'][:, freq_idx[k]],
								ecolor=cmap_v(k), elinewidth=1.25, capsize=3, markerfacecolor=cmap_v(k), markeredgecolor=(0,0,0),
								linestyle='none', marker='.', markersize=14.0, linewidth=1.2, capthick=1.6,
								label=f"{HAMP_MWR.freq['KV'][freq_idx[k]]:.2f} GHz")

			# add main diagonal for orientation:
			ax[0,1].plot(tb_lims_v, tb_lims_v, color=(0,0,0,0.33), linewidth=1.0)

			# add statistics:
			ax[0,1].text(0.99, 0.01, f"N = {n_sondes}",
					horizontalalignment='right', verticalalignment='bottom', transform=ax[0,1].transAxes, fontsize=fs-2)


		# w and f band:
		if HAMP_MWR.avail['11990']:
			# filter +/-15 sec around dropsonde launches:
			mwrson_idx = np.asarray([np.where((HAMP_MWR.time['11990'] >= lt - 15) &
									(HAMP_MWR.time['11990'] <= lt + 15))[0] for lt in HALO_dropsondes.time['dropsonde']])
			for kkk, msi in enumerate(mwrson_idx):
				HAMP_MWR.TB_mean['11990'][kkk,:] = np.nanmean(HAMP_MWR.TB['11990'][msi,:], axis=0)
				HAMP_MWR.TB_std['11990'][kkk,:] = np.nanstd(HAMP_MWR.TB['11990'][msi,:], axis=0)

			freq_idx = select_MWR_channels(HAMP_MWR.TB['11990'], HAMP_MWR.freq['11990'], band='W+F', return_idx=2)
			freq_idx_ds = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='W+F', return_idx=2)
			for k in range(n_freq_11990):
				ax[1,0].errorbar(HALO_dropsondes.TB['dropsonde'][:,freq_idx_ds[k]], HAMP_MWR.TB_mean['11990'][:, freq_idx[k]],
								yerr=HAMP_MWR.TB_std['11990'][:, freq_idx[k]],
								ecolor=cmap_11990(k), elinewidth=1.25, capsize=3, markerfacecolor=cmap_11990(k), markeredgecolor=(0,0,0),
								linestyle='none', marker='.', markersize=14.0, linewidth=1.2, capthick=1.6,
								label=f"{HAMP_MWR.freq['11990'][freq_idx[k]]:.2f} GHz")

			# add main diagonal for orientation:
			ax[1,0].plot(tb_lims_wf, tb_lims_wf, color=(0,0,0,0.33), linewidth=1.0)

			# add statistics:
			ax[1,0].text(0.99, 0.01, f"N = {n_sondes}",
					horizontalalignment='right', verticalalignment='bottom', transform=ax[1,0].transAxes, fontsize=fs-2)


		# g band:
		if HAMP_MWR.avail['183']:
			# filter +/-15 sec around dropsonde launches:
			mwrson_idx = np.asarray([np.where((HAMP_MWR.time['183'] >= lt - 15) &
									(HAMP_MWR.time['183'] <= lt + 15))[0] for lt in HALO_dropsondes.time['dropsonde']])
			for kkk, msi in enumerate(mwrson_idx):
				HAMP_MWR.TB_mean['183'][kkk,:] = np.nanmean(HAMP_MWR.TB['183'][msi,:], axis=0)
				HAMP_MWR.TB_std['183'][kkk,:] = np.nanstd(HAMP_MWR.TB['183'][msi,:], axis=0)

			freq_idx = select_MWR_channels(HAMP_MWR.TB['183'], HAMP_MWR.freq['183'], band='G', return_idx=2)
			freq_idx_ds = select_MWR_channels(HALO_dropsondes.TB['dropsonde'], HALO_dropsondes.freq['dropsonde'], band='G', return_idx=2)
			for k in range(n_freq_183):
				ax[1,1].errorbar(HALO_dropsondes.TB['dropsonde'][:,freq_idx_ds[k]], HAMP_MWR.TB_mean['183'][:, freq_idx[k]],
								yerr=HAMP_MWR.TB_std['183'][:, freq_idx[k]],
								ecolor=cmap_183(k), elinewidth=1.25, capsize=3, markerfacecolor=cmap_183(k), markeredgecolor=(0,0,0),
								linestyle='none', marker='.', markersize=14.0, linewidth=1.2, capthick=1.6,
								label=f"{HAMP_MWR.freq['183'][freq_idx[k]]:.2f} GHz")

			# add main diagonal for orientation:
			ax[1,1].plot(tb_lims_g, tb_lims_g, color=(0,0,0,0.33), linewidth=1.0)

			# add statistics:
			ax[1,1].text(0.99, 0.01, f"N = {n_sondes}",
					horizontalalignment='right', verticalalignment='bottom', transform=ax[1,1].transAxes, fontsize=fs-2)


		# legends or colorbars:
		lh, ll = ax[0,0].get_legend_handles_labels()
		lg = ax[0,0].legend(handles=lh, labels=ll, loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1, 
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_k(k))

		lh, ll = ax[0,1].get_legend_handles_labels()
		lg = ax[0,1].legend(handles=lh, labels=ll, loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1, 
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_v(k))

		lh, ll = ax[1,0].get_legend_handles_labels()
		lg = ax[1,0].legend(handles=lh, labels=ll, loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1, 
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_11990(k))

		lh, ll = ax[1,1].get_legend_handles_labels()
		lg = ax[1,1].legend(handles=lh, labels=ll, loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1,
								   columnspacing=1, borderaxespad=0, handlelength=0)
		for item in lg.legendHandles:
				item.set_visible(False)
		for k, text in enumerate(lg.get_texts()):
			text.set_color(cmap_183(k))


		# set axis limits:
		ax[0,0].set_xlim(left=tb_lims_k[0], right=tb_lims_k[1])
		ax[0,1].set_xlim(left=tb_lims_v[0], right=tb_lims_v[1])
		ax[1,0].set_xlim(left=tb_lims_wf[0], right=tb_lims_wf[1])
		ax[1,1].set_xlim(left=tb_lims_g[0], right=tb_lims_g[1])

		ax[0,0].set_ylim(bottom=tb_lims_k[0], top=tb_lims_k[1])
		ax[0,1].set_ylim(bottom=tb_lims_v[0], top=tb_lims_v[1])
		ax[1,0].set_ylim(bottom=tb_lims_wf[0], top=tb_lims_wf[1])
		ax[1,1].set_ylim(bottom=tb_lims_g[0], top=tb_lims_g[1])


		# tick parameters:
		ax[0,0].tick_params(axis='both', labelsize=fs_small)
		ax[0,1].tick_params(axis='both', labelsize=fs_small)
		ax[1,0].tick_params(axis='both', labelsize=fs_small)
		ax[1,1].tick_params(axis='both', labelsize=fs_small)

		ax[0,0].minorticks_on()
		ax[0,1].minorticks_on()
		ax[1,0].minorticks_on()
		ax[1,1].minorticks_on()


		# grid:
		ax[0,0].grid(which='major', axis='both', alpha=0.4)
		ax[0,1].grid(which='major', axis='both', alpha=0.4)
		ax[1,0].grid(which='major', axis='both', alpha=0.4)
		ax[1,1].grid(which='major', axis='both', alpha=0.4)
		

		# Limit axis spacing:
		plt.subplots_adjust(hspace=0.0, wspace=0.55)


		# Aspect ratio:
		ax[0,0].set_aspect('equal')
		ax[0,1].set_aspect('equal')
		ax[1,0].set_aspect('equal')
		ax[1,1].set_aspect('equal')


		# set labels:
		ax[1,0].set_xlabel("TB$_{\mathrm{dropsonde}}$ (K)", fontsize=fs)
		ax[1,1].set_xlabel("TB$_{\mathrm{dropsonde}}$ (K)", fontsize=fs)
		ax[0,0].set_ylabel("TB$_{\mathrm{HAMP}}$ (K)", fontsize=fs)
		ax[1,0].set_ylabel("TB$_{\mathrm{HAMP}}$ (K)", fontsize=fs)

		fig.suptitle(f"HALO-AC3_HALO_hamp_dropsondes_{which_date}_{RF_now}", y=0.90, fontsize=fs)


		if not os.path.exists(path_output):
			os.makedirs(path_output)

		if save_figures:
			plot_name = f"HALO-AC3_HALO_hamp_radiometer_dropsonde_{which_date}_{RF_now}.png"
			fig.savefig(path_output + plot_name, dpi=400, bbox_inches='tight')
			print(f"Plot saved to {path_output + plot_name}.")
		else:
			plt.show()


print("Done....")