import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

import os
import glob
import sys
import pdb
# sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")


"""
	Visualise the frequency distribution of IWV for predictions and reference IWV. Requires to 
	have the reference IWV data and the prediction saved to file.
	- import predicted and reference data
	- compute the IWV frequency distributions
	- visualise
"""


# paths
path_data = {	'predic_n_ref': "/mnt/d/heavy_data/synergetic_ret/synergetic_ret/tests_00/prediction_and_reference/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/synergetic_ret/tests_00/iwv/"

# additional settings:
set_dict = {'save_figures': True,				# if True, figure is saved to file
			'plot_iwv_freq_distribution': False,		# plots iwv frequency distribution if True
			'plot_cum_iwv_freq_dis': True,			# cumulative frequency distribution if True
			'x_lims': np.array([0.0, 35.0])}	# x axis limits

set_dict['bins'] = np.arange(set_dict['x_lims'][0], set_dict['x_lims'][1]+0.00001, 0.5)

# check existence of plot path:
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)


# import data:
files = sorted(glob.glob(path_data['predic_n_ref'] + "*.nc"))
DS_dict = dict()
hist_dict = dict()
for ix, file in enumerate(files):
	kk = str(ix)
	DS_dict[kk] = xr.open_dataset(file)


	# compute the histogram, but don't plot histogram bars: first, filter out nans:
	ok_idx = np.where( (~np.isnan(DS_dict[kk].prediction.values)) & (~np.isnan(DS_dict[kk].reference.values)))[0]
	DS_dict[kk] = DS_dict[kk].isel(n_s=ok_idx)
	n_data = DS_dict[kk].n_s.shape[0]
	DS_dict[kk]['weights'] = xr.DataArray(np.ones((int(n_data),)) / n_data, dims=['n_s'])

	hist_dict[kk] = {'prediction': np.histogram(DS_dict[kk].prediction.values, bins=set_dict['bins'], density=True),
					'reference': np.histogram(DS_dict[kk].reference.values, bins=set_dict['bins'], density=True)}

	# alternatively, np.histogram(...weights=DS_dict[kk].weights.values) instead of density when sum of freq. occurrence
	# over all bins == 1 is desired.


# create cumulative histogram if needed:
if set_dict['plot_cum_iwv_freq_dis']:

	cumulative_hist = dict()
	for key in hist_dict.keys():
		cumulative_hist[key] = {'prediction': np.cumsum(hist_dict[key]['prediction'][0])*np.diff(hist_dict[key]['prediction'][1]),
								'reference': np.cumsum(hist_dict[key]['reference'][0])*np.diff(hist_dict[key]['reference'][1])}


# visualize:
fs = 20
fs_small = fs - 2
fs_dwarf = fs_small - 2
fs_micro = fs_dwarf - 2
msize = 7.0
colours = {	'0': (0.067,0.29,0.769),	# HATPRO
			'1': (0,0.779,0.615),		# MiRAC-P
			'2': (0.8,0,0)}				# combined
labels = {'0': "HATPRO only", '1': "MiRAC-P only", "2": "HATPRO and MiRAC-P"}

if set_dict['plot_iwv_freq_distribution']:

	# plotting:

	f1 = plt.figure(figsize=(12,5))
	a1 = plt.axes()

	ax_lims = np.asarray([0.0, 35.0])


	# plot predictions:
	for key in DS_dict.keys():
		bins_plot = 0.5*np.diff(hist_dict[key]['prediction'][1]) + hist_dict[key]['prediction'][1][:-1]
		a1.plot(bins_plot, hist_dict[key]['prediction'][0], linewidth=1.25,
			color=colours[key], label=labels[key])

	# plotting reference:
	bins_plot = 0.5*np.diff(hist_dict['0']['reference'][1]) + hist_dict['0']['reference'][1][:-1]
	a1.plot(bins_plot, hist_dict['0']['reference'][0], linewidth=1.5, linestyle='dashed',
			color=(0,0,0), label='Reference', zorder=10000.0)


	# add statistics or aux info


	# Legends:
	lh, ll = a1.get_legend_handles_labels()
	a1.legend(handles=lh, labels=ll, loc='upper right', fontsize=fs_small,
				markerscale=1.5)

	# set axis limits:
	a1.set_xlim(set_dict['x_lims'][0], set_dict['x_lims'][1])
	a1.set_ylim(bottom=0.0)

	# set axis ticks, ticklabels and tick parameters:
	# a1.minorticks_on()
	a1.tick_params(axis='both', labelsize=fs_small)


	# grid:
	a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# labels:
	a1.set_ylabel("Frequency of occurrence", fontsize=fs)
	a1.set_xlabel("Integrated water vapour ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if set_dict['save_figures']:

		# for file name, append all test setting IDs:
		all_cases = ""
		for key in DS_dict.keys():
			all_cases += "_" + DS_dict[key].setup_id

		plotname = f"NN_syn_ret_eval_iwv_freq_dist{all_cases}"
		plotfile = path_plots + plotname + ".png"
		f1.savefig(plotfile, dpi=300, bbox_inches='tight')
		print(f"Saved {plotfile}")
	else:
		plt.show()
		pdb.set_trace()

	plt.close()


if set_dict['plot_cum_iwv_freq_dis']:

	# plotting:

	f1 = plt.figure(figsize=(12,5))
	a1 = plt.axes()

	ax_lims = np.asarray([0.0, 35.0])


	# plot predictions:
	for key in DS_dict.keys():
		bins_plot = 0.5*np.diff(hist_dict[key]['prediction'][1]) + hist_dict[key]['prediction'][1][:-1]
		a1.plot(bins_plot, cumulative_hist[key]['prediction'], linewidth=1.25,
			color=colours[key], label=labels[key])

	# plotting reference:
	bins_plot = 0.5*np.diff(hist_dict['0']['reference'][1]) + hist_dict['0']['reference'][1][:-1]
	a1.plot(bins_plot, cumulative_hist['0']['reference'], linewidth=1.5, linestyle='dashed',
			color=(0,0,0), label='Reference', zorder=10000.0)


	# add statistics or aux info


	# Legends:
	lh, ll = a1.get_legend_handles_labels()
	a1.legend(handles=lh, labels=ll, loc='lower right', fontsize=fs_small,
				markerscale=1.5)

	# set axis limits:
	a1.set_xlim(set_dict['x_lims'][0], set_dict['x_lims'][1])
	a1.set_ylim(bottom=0.0, top=1.0)

	# set axis ticks, ticklabels and tick parameters:
	# a1.minorticks_on()
	a1.tick_params(axis='both', labelsize=fs_small)


	# grid:
	a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	# labels:
	a1.set_ylabel("Cumulative frequency", fontsize=fs)
	a1.set_xlabel("Integrated water vapour ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)

	if set_dict['save_figures']:

		# for file name, append all test setting IDs:
		all_cases = ""
		for key in DS_dict.keys():
			all_cases += "_" + DS_dict[key].setup_id

		plotname = f"NN_syn_ret_eval_iwv_cumulative_freq_dist{all_cases}"
		plotfile = path_plots + plotname + ".png"
		f1.savefig(plotfile, dpi=300, bbox_inches='tight')
		print(f"Saved {plotfile}")
	else:
		plt.show()
		pdb.set_trace()

	plt.close()