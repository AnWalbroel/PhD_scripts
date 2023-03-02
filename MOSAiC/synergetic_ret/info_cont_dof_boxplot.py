import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *


"""
	Script to visualise the variability of Degrees of Freedom over a (test) data set, based on the output of
	info_content.py.save_info_content_data. Multiple settings can be compared (i.e., using different predictors).
	- Import the DOF data
	- bring different runs (different Neural Network settings) together
	- visualise as boxplot
"""


# Paths:
path_data = {'dof': "/mnt/d/heavy_data/synergetic_ret/synergetic_ret/tests_00/info_content/",
			}
path_plots = "/mnt/d/Studium_NIM/work/Plots/synergetic_ret/info_content/"


# some other settings: Choose date range and location boundaries:
set_dict = {'save_figures': True,		# if True, saved to .png, if False, just show plot
			'DOF_boxplot': True,		# visualises DOF variability for multiple settings
			}

# check if plot folder exists:
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

# find all and import DOF files: save them 
files = sorted(glob.glob(path_data['dof'] + "MOSAiC_synergetic_ret_info_content*.nc"))
DS_dict = dict()
for fix, file in enumerate(files):
	DS_dict[str(fix)] = xr.open_dataset(file)


# identify which dataset belongs to which setting:
# eventually, manually sort in which order the DOFs will be shown
labels_boxplot = [DS_dict[key].predictor_TBs for key in DS_dict.keys()]
labels_boxplot = [labels_boxplot[0], labels_boxplot[3], labels_boxplot[2], labels_boxplot[1]]
DS_dict_plot = {'0': DS_dict['0'], '1': DS_dict['3'], '2': DS_dict['2'], '3': DS_dict['1']}

# visualise:
fs = 20
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


if set_dict['DOF_boxplot']:

	# number of boxes to show and identify where the boxes are supposed to be on the x axis:
	n_boxes = len(labels_boxplot)
	pos_boxes = [k+1 for k in range(n_boxes)]


	def make_boxplot_great_again(bp):	# change linewidth to 1.5
		plt.setp(bp['boxes'], color=(0,0,0), linewidth=1.5)
		plt.setp(bp['whiskers'], color=(0,0,0), linewidth=1.5)
		plt.setp(bp['caps'], color=(0,0,0), linewidth=1.5)
		plt.setp(bp['medians'], color=(0,0,0), linewidth=1.5)

	f1 = plt.figure(figsize=(8,8))
	a1 = plt.axes()

	ax_lims = np.array([0.0, 4.0])	# DOF
	x_lims = np.array([pos_boxes[0]-1, pos_boxes[-1]+1])
	# whis_lims = [5,95]			# whisker limits

	# BOXPLOT of data:
	bp_plots = dict()
	for k in range(n_boxes):
		bp_plots[str(k)] = a1.boxplot(DS_dict_plot[str(k)].DOF.values, positions=[pos_boxes[k]], sym='.', widths=0.5)
		make_boxplot_great_again(bp_plots[str(k)])

		# add aux info (texts, annotations):
		a1.text(pos_boxes[k], 1.00*np.diff(ax_lims)+ax_lims[0], f"{np.median(DS_dict_plot[str(k)].DOF.values):.2f}",
				color=(0,0,0), fontsize=fs_dwarf, ha='center', va='bottom', 
				transform=a1.transData)

	a1.text(x_lims[0], 1.0*np.diff(ax_lims)+ax_lims[0], "Median:",
				color=(0,0,0), fontsize=fs_dwarf, ha='left', va='bottom', 
				transform=a1.transData)

	# legend/colorbar:


	# set axis limits:
	a1.set_ylim(ax_lims[0], ax_lims[1])
	a1.set_xlim(x_lims[0], x_lims[1])

	# set ticks and tick labels and parameters: for plotting, break string into several lines if too long
	a1.set_xticks(pos_boxes)
	labels_boxplot = [break_str_into_lines(lab, 9, split_at='+', keep_split_char=True) for lab in labels_boxplot]
	a1.set_xticklabels(labels_boxplot)
	a1.tick_params(axis='both', labelsize=fs_small)

	# grid:
	a1.grid(which='both', axis='y', color=(0.5,0.5,0.5), alpha=0.5)
	a1.set_axisbelow(True)


	# set labels:
	a1.set_xlabel("TB bands as predictors", fontsize=fs)
	a1.set_ylabel("Degrees Of Freedom q profile", fontsize=fs)

	if set_dict['save_figures']:

		plotname = "MOSAiC_synergetic_ret_info_content_boxplot"

		# add all test identifiers to the plot name:
		all_cases = ""
		for key in DS_dict_plot.keys():
			all_cases += "_" + DS_dict_plot[key].setup_id
		plotname = plotname + all_cases + ".png"
		plot_file = path_plots + plotname
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Saved " + plot_file)					

	else:
		plt.show()
		pdb.set_trace()