import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pdb


"""
	Simple script to plot TB differences (observed - simulated) against frequency for each calibration
	period and instrument.
"""

path_data = "/net/blanc/awalbroe/Data/MOSAiC_radiometers_offsets/"
path_plots = {	'hatpro': ("/net/blanc/awalbroe/Plots/TB_comparison_mwr_pamtrasonde/hatpro_freq/" +
						"test_011_Ell_10m_S_11_clear_sky/bias_frequency/"),
				'mirac-p': ("/net/blanc/awalbroe/Plots/TB_comparison_mwr_pamtrasonde/lhumpro_freq/" +
						"test_011_Ell_10m_S_11_clear_sky/bias_frequency/")}

plot_all_calibs_in_one = True		# do not splot calibration periods into several subplots but all in
									# one plot (with differing colours)
save_figures = True


# status display if offset file is available:
status_hatpro = 0
status_mirac_p = 0

files = glob.glob(path_data + "*.nc")

# Load data:
for file in files:
	if "_hatpro_" in file:
		status_hatpro = 1
		HATPRO_DS = xr.open_dataset(file)
		n_calib_HATPRO = len(HATPRO_DS.time)

	if "_mirac-p_" in file:
		status_mirac_p = 1
		MIRAC_DS = xr.open_dataset(file)
		n_calib_MIRAC = len(MIRAC_DS.time)

# define some band limits for the plot
K_band_limits = np.array([22.24, 31.4])
V_band_limits = np.array([51.26, 58.0])
G_band_limits = np.array([183.91, 190.81])


# Plotting:
fs = 15		# fontsize

if not plot_all_calibs_in_one:

	if status_hatpro:
		instrument = 'hatpro'
		n_freq = len(HATPRO_DS.frequency)

		# only do max 4 calibration periods per figure:
		calib_period_ranges = [np.arange(0, 4), np.arange(4,7)]
		for cpr in calib_period_ranges:

			n_subplots = len(cpr)
			fig1, ax0 = plt.subplots(nrows=n_subplots, ncols=1, sharex=True, squeeze=True,
									gridspec_kw={'hspace': 0.2}, figsize=(10,20))

			for a_idx, k in enumerate(cpr):
				# convert numpy datetime to string for title:
				title_text = "Calibration period %i: %s - %s"%(k+1,
										np.datetime_as_string(HATPRO_DS.calibration_period_start.values[k], unit="D"),
										np.datetime_as_string(HATPRO_DS.calibration_period_end.values[k], unit="D"))

				x_data = np.arange(0, n_freq+2)
				ax0[a_idx].plot(x_data[1:-1], HATPRO_DS.bias[k,:], color=(0,0,0), marker='.', markersize=10)

				# 0 line:
				ax0[a_idx].plot([x_data[0], x_data[-1]], [0,0], linewidth=1.25, color=(0,0,0))

				if a_idx == n_subplots-1:
					ax0[a_idx].set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=0.5)

				ax0[a_idx].set_ylim(bottom=-10, top=10)
				ax0[a_idx].set_xlim(left=x_data[0], right=x_data[-1])

				# x ticks = frequency labels:
				ax0[a_idx].set_xticks(x_data[1:-1])
				x_labels = ["%.2f"%frq for frq in HATPRO_DS.frequency]
				ax0[a_idx].set_xticklabels(x_labels)
				ax0[a_idx].tick_params(axis='both', labelsize=fs-4)

				ax0[a_idx].yaxis.set_minor_locator(AutoMinorLocator())
				ax0[a_idx].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
				ax0[a_idx].set_title(title_text, fontsize=fs, pad=0)

			# One y axis label for the entire figure:
			fig1.text(0.05, 0.5, "TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", rotation='vertical', 
						va='center', ha='left', fontsize=fs)
			plt.suptitle(instrument.upper(), fontsize=fs+2)

			if save_figures:
				fig1.savefig(path_plots[instrument] + 
							"MOSAiC_%s_radiometer_TB_bias_vs_frequency_period_%i-%i.png"%(instrument, cpr[0], cpr[-1]), dpi=400)
			else:
				plt.show()


	if status_mirac_p:
		instrument = 'mirac-p'
		n_freq = len(MIRAC_DS.frequency)

		# only do max 4 calibration periods per figure:
		calib_period_ranges = [np.arange(0, 3), np.arange(3,5)]
		for cpr in calib_period_ranges:

			n_subplots = len(cpr)
			fig1, ax0 = plt.subplots(nrows=n_subplots, ncols=1, sharex=True, squeeze=True,
									gridspec_kw={'hspace': 0.2}, figsize=(10,20))

			for a_idx, k in enumerate(cpr):
				# convert numpy datetime to string for title:
				title_text = "Calibration period %i: %s - %s"%(k+1,
										np.datetime_as_string(MIRAC_DS.calibration_period_start.values[k], unit="D"),
										np.datetime_as_string(MIRAC_DS.calibration_period_end.values[k], unit="D"))

				x_data = np.arange(0, n_freq+2)
				ax0[a_idx].plot(x_data[1:-1], MIRAC_DS.bias[k,:], color=(0,0,0), marker='.', markersize=10)

				# 0 line:
				ax0[a_idx].plot([x_data[0], x_data[-1]], [0,0], linewidth=1.25, color=(0,0,0))

				if a_idx == n_subplots-1:
					ax0[a_idx].set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=0.5)

				ax0[a_idx].set_ylim(bottom=-10, top=10)
				ax0[a_idx].set_xlim(left=x_data[0], right=x_data[-1])

				# x ticks = frequency labels:
				ax0[a_idx].set_xticks(x_data[1:-1])
				# x_labels = ["%.2f"%frq for frq in MIRAC_DS.frequency]
				x_labels = ["183.31$\pm$0.6", "183.31$\pm$1.5", "183.31$\pm$2.5", 
							"183.31$\pm$3.5", "183.31$\pm$5.0", "183.31$\pm$7.5",
							"243.00", "340.00"]
				ax0[a_idx].set_xticklabels(x_labels)
				ax0[a_idx].tick_params(axis='x', labelsize=fs-6)
				ax0[a_idx].tick_params(axis='y', labelsize=fs-4)

				ax0[a_idx].yaxis.set_minor_locator(AutoMinorLocator())
				ax0[a_idx].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
				ax0[a_idx].set_title(title_text, fontsize=fs, pad=0)

			# One y axis label for the entire figure:
			fig1.text(0.05, 0.5, "TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", rotation='vertical', 
						va='center', ha='left', fontsize=fs)
			plt.suptitle(instrument.upper(), fontsize=fs+2)

			if save_figures:
				fig1.savefig(path_plots[instrument] + 
							"MOSAiC_%s_radiometer_TB_bias_vs_frequency_period_%i-%i.png"%(instrument, cpr[0], cpr[-1]), dpi=400)
			else:
				plt.show()

else:

	fs=22
	if status_hatpro:
		instrument = 'hatpro'
		n_freq = len(HATPRO_DS.frequency)
		n_calib = len(HATPRO_DS.calibration_period_start)

		fig1, ax0 = plt.subplots(1,1)
		fig1.set_size_inches(18,12)

		# convert numpy datetime to string for title:
		x_data = np.arange(0, n_freq+2)
		cmap = plt.cm.get_cmap('Dark2', n_calib)
		for k in range(n_calib):
			legend_text = "%i: %s - %s"%(k+1,
								np.datetime_as_string(HATPRO_DS.calibration_period_start.values[k], unit="D"),
								np.datetime_as_string(HATPRO_DS.calibration_period_end.values[k], unit="D"))
			ax0.plot(x_data[1:-1], HATPRO_DS.bias[k,:], color=cmap(k), marker='.', markersize=16, linewidth=2.5,
						label=legend_text)


		# 0 line:
		ax0.plot([x_data[0], x_data[-1]], [0,0], linewidth=1.25, color=(0,0,0))

		# add legend:
		legh, legl = ax0.get_legend_handles_labels()
		ax0.legend(handles=legh, labels=legl, loc='upper left', fontsize=fs-6, framealpha=1.0)

		ax0.set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=1.0)
		ax0.set_ylabel("TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", fontsize=fs)

		ax0.set_ylim(bottom=-10, top=10)
		ax0.set_xlim(left=x_data[0], right=x_data[-1])

		# x ticks = frequency labels:
		ax0.set_xticks(x_data[1:-1])
		x_labels = ["%.2f"%frq for frq in HATPRO_DS.frequency]
		ax0.set_xticklabels(x_labels)
		ax0.tick_params(axis='both', labelsize=fs-2)

		ax0.yaxis.set_minor_locator(AutoMinorLocator())
		ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
		ax0.set_title("%s clear sky offsets"%(instrument.upper()), fontsize=fs, pad=0)


		if save_figures:
			fig1.savefig(path_plots[instrument] + 
						"MOSAiC_%s_radiometer_TB_bias_vs_frequency_all_CPs.png"%(instrument), dpi=400)
		else:
			plt.show()


	if status_mirac_p:
		instrument = 'mirac-p'
		n_freq = len(MIRAC_DS.frequency)
		n_calib = len(MIRAC_DS.calibration_period_start)

		fig1, ax0 = plt.subplots(1,1)
		fig1.set_size_inches(18,12)

		# convert numpy datetime to string for title:
		x_data = np.arange(0, n_freq+2)
		cmap = plt.cm.get_cmap('Dark2', n_calib)
		for k in range(n_calib):
			legend_text = "%i: %s - %s"%(k+1,
								np.datetime_as_string(MIRAC_DS.calibration_period_start.values[k], unit="D"),
								np.datetime_as_string(MIRAC_DS.calibration_period_end.values[k], unit="D"))
			ax0.plot(x_data[1:-1], MIRAC_DS.bias[k,:], color=cmap(k), marker='.', markersize=16, linewidth=2.5,
						label=legend_text)


		# 0 line:
		ax0.plot([x_data[0], x_data[-1]], [0,0], linewidth=1.25, color=(0,0,0))

		# add legend:
		legh, legl = ax0.get_legend_handles_labels()
		ax0.legend(handles=legh, labels=legl, loc='upper left', fontsize=fs-6, framealpha=1.0)

		ax0.set_xlabel("Frequency (GHz)", fontsize=fs, labelpad=1.0)
		ax0.set_ylabel("TB$_{\mathrm{obs}}$ - TB$_{\mathrm{sim}}$ (K)", fontsize=fs)

		ax0.set_ylim(bottom=-10, top=10)
		ax0.set_xlim(left=x_data[0], right=x_data[-1])

		# x ticks = frequency labels:
		ax0.set_xticks(x_data[1:-1])
		x_labels = ["%.2f"%frq for frq in MIRAC_DS.frequency]
		ax0.set_xticklabels(x_labels)
		ax0.tick_params(axis='both', labelsize=fs-2)

		ax0.yaxis.set_minor_locator(AutoMinorLocator())
		ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
		ax0.set_title("%s clear sky offsets"%(instrument.upper().replace("I", "i")), fontsize=fs, pad=0)


		if save_figures:
			fig1.savefig(path_plots[instrument] + 
						"MOSAiC_%s_radiometer_TB_bias_vs_frequency_all_CPs.png"%(instrument), dpi=400)
		else:
			plt.show()
	

print("Done....")