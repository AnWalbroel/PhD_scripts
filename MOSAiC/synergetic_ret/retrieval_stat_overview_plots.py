import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.family': "monospace"})

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import AutoMinorLocator

import sys
import os
import glob
import pdb

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import break_str_into_lines

"""
	This script is used to visualize retrieval statistics .nc output generated with 
	NN_retrieval.py. The metrics are i.e., test data loss, RMSE (Root Mean Square Error),
	bias, standard deviation. These metrics will be visualized here to get an idea which 
	settings work best on the predicted vs. test data predictands.

	- load netCDF
	- plot
"""

if len(sys.argv) == 1:
		test_no = "000"
elif len(sys.argv) == 2:
		test_no = sys.argv[1]

path_plots = "/net/blanc/awalbroe/Plots/synergetic_ret/tests_00/ret_stat/"
path_data = "/net/blanc/awalbroe/Data/synergetic_ret/tests_00/output/ret_stat/"
set_dict = {'save_figures': True,
			'test_no': test_no,
			'predictands': ["iwv", "lwp", "temp", "q"]}
files = sorted(glob.glob(path_data + f"NN_syn_ret_retrieval_*{set_dict['test_no']}.nc"))



for file in files:
	print(file.replace(path_data,''))

	RET_DS = xr.open_dataset(file)


	# Create x-axis labels from information given in the retrieval dataset:
	"""
	'batch_size', 'epochs', 'activation', 'seed', 'feature_range0', 'feature_range1'
	e.g., : B000_E00_exponential(-5,0)_00
	"""
	n_tests = len(RET_DS.test_id.values)
	custom_x_tick_labels = np.full((n_tests,), ' '*45)
	for k in range(n_tests):
		bs_info = f"B{RET_DS.batch_size.values[k]}_"
		e_info = f"E{RET_DS.epochs.values[k]}_"
		se_info = f"{RET_DS.seed.values[k]:02}_"

		if RET_DS.activation.values[k] == 'exponential':
			affr_info = (f"{RET_DS.activation.values[k][:3]}({int(RET_DS.feature_range0.values[k]):2}," +
						f"{int(RET_DS.feature_range1.values[k]):2})_")
		else:
			affr_info = (f"{RET_DS.activation.values[k]}({int(RET_DS.feature_range0.values[k]):2}," +
						f"{int(RET_DS.feature_range1.values[k]):2})_")

		if 'learning_rate' in RET_DS.data_vars:
			lr_info = f"LR{str(RET_DS.learning_rate.values[k])}_"

		# before new line and last part of the label don't need the "_" --> [:-1]
		custom_x_tick_labels[k] = bs_info + e_info + lr_info[:-1] + "\n" + affr_info + se_info[:-1]


	# Plot:
	fs = 12
	fs_small = fs-2
	fs_dwarf = fs_small-2
	fs_micro = fs_dwarf-2
	c_bot = (0,0,0.75)
	c_top = (0.75,0,0)
	xlims = [RET_DS.test_id[0], RET_DS.test_id[-1]]		# 'IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'

	for predictand in set_dict['predictands']:

		if predictand in RET_DS.predictands:

			# create folder if not existing:
			path_plots_predictand = path_plots + predictand + "/"
			path_plots_dir = os.path.dirname(path_plots_predictand)
			if not os.path.exists(path_plots_dir):
				os.makedirs(path_plots_dir)

			fig1 = plt.figure(figsize=(12,14))
			axd = plt.subplot2grid((6,2), (0,0), rowspan=1, colspan=1)		# description
			axL = plt.subplot2grid((6,2), (0,1), rowspan=1, colspan=1)		# legend
			ax0 = plt.subplot2grid((6,2), (1,0), rowspan=1, colspan=2)		# test loss
			ax1 = plt.subplot2grid((6,2), (2,0), rowspan=2, colspan=2)		# rmse
			ax2 = plt.subplot2grid((6,2), (4,0), rowspan=2, colspan=2)		# bias


			# description of the tests: break string into two or more lines if too long:
			descr_text = break_str_into_lines(predictand + ": " + RET_DS.test_purpose, n_max=40)
				
			axd.text(0.02, 0.98, descr_text, ha='left', va='top', transform=axd.transAxes, fontsize=fs)
			axd.axis('off')

			# legend
			axL.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=1.2, marker='.', 
						ms=9, label='total')
			axL.plot([np.nan, np.nan], [np.nan, np.nan], color=c_bot, linewidth=1.2, marker='.',
						ms=9, label=(predictand + " in " + RET_DS[f"{predictand}_bias_bot"].range))
			axL.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linestyle='dotted', 
						linewidth=1.2, marker='.', ms=6, label=(predictand + " in " + RET_DS[f"{predictand}_bias_mid"].range))
			axL.plot([np.nan, np.nan], [np.nan, np.nan], color=c_top, linewidth=1.2, 
						marker='.', ms=9, label=(predictand + " in " + RET_DS[f"{predictand}_bias_top"].range))

			lh, ll = axL.get_legend_handles_labels()
			axL.legend(handles=lh, labels=ll, loc="upper center", fontsize=fs_small)
			axL.axis('off')


			# plot test_loss:
			ax0.plot(RET_DS.test_id.values, RET_DS.test_loss.values, color=(0,0,0), linewidth=1.2, marker='.', ms=9)

			ax0.set_ylim(bottom=0.0)
			ax0.set_xlim(left=xlims[0], right=xlims[1])
			ax0_ylabel = ax0.set_ylabel(f"test loss ({RET_DS.test_loss.units})", fontsize=fs, labelpad=12.0)

			# ax0.minorticks_on()
			ax0.yaxis.set_minor_locator(AutoMinorLocator())
			ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

			ax0.tick_params(axis='both', labelsize=fs_small)
			ax0.xaxis.set_ticks(RET_DS.test_id.values)
			ax0.xaxis.set_ticklabels([])


			# plot RMSE:
			if predictand in ['iwv', 'lwp']:
				ax1.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_rmse_tot'].values, color=(0,0,0), linewidth=1.2, marker='.', 
							ms=9)
			ax1.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_rmse_bot'].values, color=c_bot, linewidth=1.2, marker='.',
						ms=9)
			ax1.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_rmse_mid'].values, color=(0,0,0), linestyle='dotted', 
						linewidth=1.2, marker='.', ms=6)
			ax1.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_rmse_top'].values, color=c_top, linewidth=1.2, 
						marker='.', ms=9)


			ax1.set_xlim(left=xlims[0], right=xlims[1])
			ax1.set_ylim(bottom=0.0)
			ax1_ylabel = ax1.set_ylabel(f"RMSE ({RET_DS[predictand + '_rmse_tot'].units})", fontsize=fs, labelpad=2.0)

			# ax1.minorticks_on()
			ax1.yaxis.set_minor_locator(AutoMinorLocator())
			ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

			ax1.tick_params(axis='both', labelsize=fs_small)
			ax1.xaxis.set_ticks(RET_DS.test_id.values)
			ax1.xaxis.set_ticklabels([])


			# plot bias:
			if predictand in ['iwv', 'lwp']:
				ax2.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_bias_tot'].values, color=(0,0,0), linewidth=1.2, marker='.',
							ms=9)
			ax2.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_bias_bot'].values, color=c_bot, linewidth=1.2, marker='.',
						ms=9)
			ax2.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_bias_mid'].values, color=(0,0,0), linestyle='dotted',
						linewidth=1.2, marker='.', ms=6)
			ax2.plot(RET_DS.test_id.values, RET_DS[f'{predictand}_bias_top'].values, color=c_top, linewidth=1.2, marker='.',
						ms=9)

			# dummpy line to stress bias = 0:
			ax2.plot([xlims[0]-1, xlims[1]+1], [0,0], color=(0,0,0), linewidth=1.0)


			ax2.set_xlim(left=xlims[0], right=xlims[1])
			ax2_ylabel = ax2.set_ylabel(f"Bias ({RET_DS[predictand + '_bias_tot'].units})", fontsize=fs, labelpad=2.0)

			# ax2.minorticks_on()
			ax2.yaxis.set_minor_locator(AutoMinorLocator())
			ax2.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

			ax2.tick_params(axis='both', labelsize=fs_small)
			ax2.xaxis.set_ticks(RET_DS.test_id.values)
			ax2.xaxis.set_ticklabels(custom_x_tick_labels)
			ax2.tick_params(axis='x', rotation=90, labelsize=fs_micro)


			# save figure if desired:
			if set_dict['save_figures']:
				fig1.savefig(file.replace(".nc", "_OVERVIEW.png").replace(path_data, path_plots_predictand), dpi=400)

			else:
				plt.show()

			plt.close()

print("Done....")