import numpy as np
import datetime as dt
import pdb
import matplotlib.pyplot as plt
import os
import glob
import warnings
from import_data import *
from met_tools import *
from data_tools import *
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000		# to avoid a bug with OverflowError: In draw_path: Exceeded cell block limit


'''
	Code to create quicklook plots of temperature and specific humidity profiles of radiosondes
	launched during MOSAiC. Input data was originally (e.g.):
	/data/testbed/datasets/MOSAiC/rs41/ps200913.w02.nc
'''


path_sondes = "/data/testbed/datasets/MOSAiC/rs41/"
path_sondes_old = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/"
path_plots = "/net/blanc/awalbroe/Plots/MOSAiC_radiosonde_quicklook/"


# wanted_periods = [["2019-12-29", "2020-01-16"],
					# ["2020-03-24", "2020-04-02"],
					# # ["2020-02-05", "2020-02-21"],
					# ["2020-07-14", "2020-08-01"],
					# # ["2020-09-24", "2020-10-02"],
					# # ["2019-11-05", "2019-11-23"],
					# # ["2019-12-23", "2020-01-05"],
					# ["2020-01-21", "2020-02-06"],
					# # ["2020-02-21", "2020-03-02"],
					# ["2020-05-23", "2020-06-10"]]
# Options of this script:
considered_period = 'mosaic'		# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# 'user': user defined
plot_T_and_q_prof = False
plot_T_and_q_prof_diff = False		# difference of T and hum profiles between the psYYMMDD.wHH and mossonde-curM1 radiosondes
plot_corr_T_and_q_vs_delta_q = True
corr_plot_whole_period = True
save_figures = True

# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-05-23", "2020-06-10"]}	
date_start = daterange_options[considered_period][0]				# def: "2019-09-30"
date_end = daterange_options[considered_period][1]					# def: "2020-10-02"
if not date_start and not date_end: raise ValueError("Please specify a date range in yyyy-mm-dd format.")


# Import radiosonde data, compute IWV and convert time to datetime:
sonde_dict = import_radiosonde_daterange(path_sondes, date_start, date_end, s_version='psYYMMDDwHH', verbose=1)
n_sondes = len(sonde_dict['launch_time'])
sonde_dict['iwv'] = np.full((n_sondes,), np.nan)
for k in range(n_sondes): sonde_dict['iwv'][k] = compute_IWV_q(sonde_dict['q'][k,:], sonde_dict['pres'][k,:])
sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])

# Exclude sondes, where Sandro Dahlke's balloon burst altitude is <= 10000 m.
# First: load the time of failed sondes. Then find, which launches they correspond to.
failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()
which_failed = []
for ft in failed_sondes_t:
	it_failed = np.argwhere(np.abs(ft - sonde_dict['launch_time']) <= 1800).flatten()
	if it_failed.size > 0:
		which_failed.append(it_failed[0])
these_did_not_fail = np.full((n_sondes,), True)
these_did_not_fail[which_failed] = False

if plot_T_and_q_prof_diff or plot_corr_T_and_q_vs_delta_q:
	# Import old radiosonde data, compute IWV and convert time to datetime:
	sonde_dict_old = import_radiosonde_daterange(path_sondes_old, date_start, date_end, s_version='mossonde', verbose=1)
	n_sondes_old = len(sonde_dict_old['launch_time'])
	sonde_dict_old['iwv'] = np.full((n_sondes_old,), np.nan)
	for k in range(n_sondes_old): sonde_dict_old['iwv'][k] = compute_IWV_q(sonde_dict_old['q'][k,:], sonde_dict_old['pres'][k,:])
	sonde_dict_old['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict_old['launch_time']])

	# Exclude sondes, where Sandro Dahlke's balloon burst altitude is <= 10000 m.
	# First: load the time of failed sondes. Then find, which launches they correspond to.
	which_failed_old = []
	for ft in failed_sondes_t:
		it_failed_old = np.argwhere(np.abs(ft - sonde_dict_old['launch_time']) <= 1800).flatten()
		if it_failed_old.size > 0:
			which_failed_old.append(it_failed_old[0])
	


fs = 19		# fontsize
if plot_T_and_q_prof:
	for k in range(n_sondes):
		if k not in which_failed:
			print(k/n_sondes)
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(10,22)

			ax.plot(sonde_dict['temp'][k,:], sonde_dict['geopheight'][k,:], color=(0,0,0), linewidth=1.2)

			ax.set_ylim(bottom=0, top=10000)
			ax.set_xlim(left=210, right=290)

			ax.set_xlabel("T (K)", fontsize=fs-1)
			ax.set_ylabel("Height (m)", fontsize=fs-1)
			ax.set_title(r"MOSAiC Radiosonde temperature (T) and specific humidity (q) profile" + " \n" +
							sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S") +
							", IWV: %.2f"%sonde_dict['iwv'][k] + "$\mathrm{kg}\,\mathrm{m}^{-2}$", fontsize=fs)

			ax.grid(which='major', axis='both')

			ax.tick_params(axis='both', labelsize=fs-3)

			# q profile:
			ax2 = ax.twiny()
			q_color = (0,0.58,1)
			ax2.plot(1000*sonde_dict['q'][k,:], sonde_dict['geopheight'][k,:], color=q_color, linewidth=1.2)

			if sonde_dict['iwv'][k] > 15:
				ax2.set_xlim(left=0, right=10)
			else:
				ax2.set_xlim(left=0, right=5)
			ax2.set_xlabel("q ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-1, color=q_color)
			ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
			


			ax_pos = ax.get_position().bounds
			ax_pos = ax2.get_position().bounds
			ax.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
			ax_pos = ax2.get_position().bounds
			ax2.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])

			filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
			if save_figures: fig.savefig(path_plots + "Radiosonde_Tq_" + filename_suffix + ".png", dpi=400)

			plt.close()


if plot_T_and_q_prof_diff:
	
	for k in range(n_sondes):
		if k not in which_failed:
			print(k/n_sondes)
			fig, ax = plt.subplots(1,1)
			fig.set_size_inches(10,22)

			ax.plot(sonde_dict['temp'][k,:], sonde_dict['geopheight'][k,:], color=(0,0,0), linewidth=1.2, label='T - psYYMMDD.wHH')

			# compute difference of sondes:
			# old sonde launch time must be within 30 minutes of new sonde launch time:
			thisisit = np.argwhere(np.abs(sonde_dict['launch_time'][k] - sonde_dict_old['launch_time']) <= 1800).flatten()
			if thisisit.size > 0:
				match_old_new = thisisit[0]
				ax.plot(sonde_dict_old['temp'][match_old_new,:], sonde_dict_old['geopheight'][match_old_new,:], color=(0,0,0),
						linestyle='dashed', linewidth=1.2, label='T - mossonde')

			# legend:
			leg_handles, leg_labels = ax.get_legend_handles_labels()
			ax.legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-3,
							framealpha=0.65, bbox_to_anchor=(1.0, 1.0))

			ax.set_ylim(bottom=0, top=10000)
			ax.set_xlim(left=210, right=290)

			ax.set_xlabel("T (K)", fontsize=fs-1)
			ax.set_ylabel("Height (m)", fontsize=fs-1)
			ax.set_title(r"MOSAiC Radiosonde temperature (T) and specific humidity (q) profile" + " \n" +
							sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S") +
							", IWV: %.2f"%sonde_dict['iwv'][k] + "$\mathrm{kg}\,\mathrm{m}^{-2}$", fontsize=fs)

			ax.grid(which='major', axis='both')

			ax.tick_params(axis='both', labelsize=fs-3)

			# q profile:
			ax2 = ax.twiny()
			q_color = (0,0.58,1)
			ax2.plot(1000*sonde_dict['q'][k,:], sonde_dict['geopheight'][k,:], color=q_color, linewidth=1.2, label='q - psYYMMDD.wHH')

			if thisisit.size > 0:
				ax2.plot(1000*sonde_dict_old['q'][match_old_new,:], sonde_dict_old['geopheight'][match_old_new,:], color=q_color,
						linestyle='dashed', linewidth=1.2, label='q - mossonde')

			# legend:
			leg_handles, leg_labels = ax2.get_legend_handles_labels()
			ax2.legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=fs-3,
							framealpha=0.65, bbox_to_anchor=(1.0, 0.9))

			if sonde_dict['iwv'][k] > 13:
				ax2.set_xlim(left=0, right=10)
			else:
				ax2.set_xlim(left=0, right=5)
			ax2.set_xlabel("q ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-1, color=q_color)
			ax2.tick_params(axis='x', labelcolor=q_color, labelsize=fs-3)
			


			ax_pos = ax.get_position().bounds
			ax_pos = ax2.get_position().bounds
			ax.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
			ax_pos = ax2.get_position().bounds
			ax2.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])

			filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
			if save_figures: fig.savefig(path_plots + "Radiosonde_Tq_oldnew_" + filename_suffix + ".png", dpi=400)

			plt.close()


if plot_corr_T_and_q_vs_delta_q:

	if corr_plot_whole_period:
		# find out, which sonde launches from old match with the new ones
		match_old_new = []
		match_new_old = []		# pendant to be applied to the psYYMMDD sonde dict
		for k in range(n_sondes):
			if k not in which_failed:
				thisisit = np.argwhere(np.abs(sonde_dict['launch_time'][k] - sonde_dict_old['launch_time']) <= 1800).flatten()
				if thisisit.size > 0:
					match_old_new.append(thisisit[0])
					match_new_old.append(k)

		# then convert the T and q profiles to flat arrays to allow all data points to be plotted at once
		mossonde_T_period = sonde_dict_old['temp'][match_old_new,:].flatten()
		psYYMMDD_T_period = sonde_dict['temp'][match_new_old,:].flatten()

		mossonde_q_period = sonde_dict_old['q'][match_old_new,:].flatten()
		psYYMMDD_q_period = sonde_dict['q'][match_new_old,:].flatten()

		# and compute delta q:
		delta_q = mossonde_q_period - psYYMMDD_q_period

		fig, ax = plt.subplots(1,2)
		fig.set_size_inches(22,10)
		q_color = (0,0.58,1)

		ax[0].plot(psYYMMDD_T_period, 1000*delta_q, color=(0,0,0), marker='.', 
				markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), markersize=2.0,
				linestyle='none', linewidth=0.75)
	

		ax[0].set_ylim(bottom=0, top=0.4)
		ax[0].set_xlim(left=210, right=290)

		ax[0].set_xlabel("T (K)", fontsize=fs-2)
		ax[0].set_ylabel("$\mathrm{q}_{\mathrm{mossonde}} - \mathrm{q}_{\mathrm{psYYMMDD.wHH}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
		ax[0].set_title(r"Scatterplot of temperature (T) and specific humidity offset ($\Delta \mathrm{q}$)" + " \n" +
						"Radiosonde " + sonde_dict['launch_time_dt'][match_new_old][0].strftime("%Y-%m-%d %H:%M:%S") + " - "
						+ sonde_dict['launch_time_dt'][match_new_old][-1].strftime("%Y-%m-%d %H:%M:%S"), fontsize=fs-1)

		ax[0].grid(which='major', axis='both')
		# to create truely square like plots, we must use the data ratio because matplotlib cannot do it on its own
		# even after 10 years of existence
		ax[0].set_aspect(1./ax[0].get_data_ratio())
		ax[0].tick_params(axis='both', labelsize=fs-4)

		# q against delta q
		ax[1].plot(1000*psYYMMDD_q_period, 1000*delta_q, color=q_color, marker='.', 
				markerfacecolor=q_color, markeredgecolor=q_color, markersize=2.0,
				linestyle='none', linewidth=0.75)

		ax[1].set_ylim(bottom=0, top=0.4)

		if np.any(sonde_dict['iwv'] > 13):
			ax[1].set_xlim(left=0, right=10)
		else:
			ax[1].set_xlim(left=0, right=5)
		ax[1].set_xlabel("q ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
		ax[1].set_ylabel("$\mathrm{q}_{\mathrm{mossonde}} - \mathrm{q}_{\mathrm{psYYMMDD.wHH}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
		ax[1].set_title(r"Scatterplot of specific humidity (q) and its offset ($\Delta \mathrm{q}$)" + " \n" +
						"Radiosonde " + sonde_dict['launch_time_dt'][match_new_old][0].strftime("%Y-%m-%d %H:%M:%S") + " - "
						+ sonde_dict['launch_time_dt'][match_new_old][-1].strftime("%Y-%m-%d %H:%M:%S"), fontsize=fs-1)

		ax[1].grid(which='major', axis='both')
		ax[1].tick_params(axis='both', labelsize=fs-4)
		ax[1].set_aspect(1./ax[1].get_data_ratio())
		


		# ax_pos = ax.get_position().bounds
		# ax_pos = ax[1].get_position().bounds
		# ax.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
		# ax_pos = ax[1].get_position().bounds
		# ax[1].set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
		plt.show()
		filename_suffix = (sonde_dict['launch_time_dt'][match_new_old][0].strftime("%Y%m%d_%H%M%SZ") + "-" +
							sonde_dict['launch_time_dt'][match_new_old][-1].strftime("%Y%m%d_%H%M%SZ"))
		if save_figures: fig.savefig(path_plots + "Radiosonde_scatter_q_offset_correlation_psYYvsmossonde_" + filename_suffix + ".png", dpi=400)

		plt.close()

	else:

		for k in range(n_sondes):
			if k not in which_failed:
				print(k/n_sondes)

				# old sonde launch time must be within 30 minutes of new sonde launch time:
				thisisit = np.argwhere(np.abs(sonde_dict['launch_time'][k] - sonde_dict_old['launch_time']) <= 1800).flatten()
				if thisisit.size > 0:
					match_old_new = thisisit[0]
					delta_q = sonde_dict_old['q'][match_old_new,:] - sonde_dict['q'][k,:]

					fig, ax = plt.subplots(1,2)
					fig.set_size_inches(22,10)
					q_color = (0,0.58,1)

					ax[0].plot(sonde_dict['temp'][k,:], 1000*delta_q, color=(0,0,0), marker='.', 
							markerfacecolor=(0,0,0), markeredgecolor=(0,0,0), markersize=2.5,
							linestyle='none', linewidth=0.75)
				

					ax[0].set_ylim(bottom=0, top=0.4)
					ax[0].set_xlim(left=210, right=290)

					ax[0].set_xlabel("T (K)", fontsize=fs-2)
					ax[0].set_ylabel("$\mathrm{q}_{\mathrm{mossonde}} - \mathrm{q}_{\mathrm{psYYMMDD.wHH}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
					ax[0].set_title(r"Correlation of temperature (T) and specific humidity offset ($\Delta \mathrm{q}$)" + " \n" +
									"Radiosonde " + sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S"), fontsize=fs-1)

					ax[0].grid(which='major', axis='both')
					# to create truely square like plots, we must use the data ratio because matplotlib cannot do it on its own
					# even after 10 years of existence
					ax[0].set_aspect(1./ax[0].get_data_ratio())
					ax[0].tick_params(axis='both', labelsize=fs-4)

					# q against delta q
					ax[1].plot(1000*sonde_dict['q'][k,:], 1000*delta_q, color=q_color, marker='.', 
							markerfacecolor=q_color, markeredgecolor=q_color, markersize=2.5,
							linestyle='none', linewidth=0.75)

					ax[1].set_ylim(bottom=0, top=0.4)

					if sonde_dict['iwv'][k] > 13:
						ax[1].set_xlim(left=0, right=10)
					else:
						ax[1].set_xlim(left=0, right=5)
					ax[1].set_xlabel("q ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
					ax[1].set_ylabel("$\mathrm{q}_{\mathrm{mossonde}} - \mathrm{q}_{\mathrm{psYYMMDD.wHH}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs-2)
					ax[1].set_title(r"Radiosonde correlation of specific humidity (q) and its offset ($\Delta \mathrm{q}$)" + " \n" +
									sonde_dict['launch_time_dt'][k].strftime("%Y-%m-%d %H:%M:%S"), fontsize=fs-1)

					ax[1].grid(which='major', axis='both')
					ax[1].tick_params(axis='both', labelsize=fs-4)
					ax[1].set_aspect(1./ax[1].get_data_ratio())
					


					# ax_pos = ax.get_position().bounds
					# ax_pos = ax[1].get_position().bounds
					# ax.set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
					# ax_pos = ax[1].get_position().bounds
					# ax[1].set_position([ax_pos[0], ax_pos[1], ax_pos[2], ax_pos[3]*0.95])
					filename_suffix = sonde_dict['launch_time_dt'][k].strftime("%Y%m%d_%H%M%SZ")
					if save_figures: fig.savefig(path_plots + "Radiosonde_scatter_q_offset_correlation_psYYvsmossonde_" + filename_suffix + ".png", dpi=400)

					plt.close()