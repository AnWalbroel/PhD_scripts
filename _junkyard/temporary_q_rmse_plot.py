import xarray as xr
import numpy as np
import datetime as dt
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'monospace'})
mpl.use("WebAgg")
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *

path_data = "/mnt/d/heavy_data/synergetic_ret/synergetic_ret/tests_00/prediction_and_reference/"
path_plots = "/mnt/d/Studium_NIM/work/Plots/synergetic_ret/tests_00/q/rmse/min/"
files = sorted(glob.glob(path_data + "*_q_*.nc"))

DS_dict = dict()
error_dict = dict()
for ix, file in enumerate(files):
	kk = str(ix)
	DS_dict[kk] = xr.open_dataset(file)


	error_dict[kk] = dict()

	# on x axis: reference; y axis: prediction
	x_stuff = DS_dict[kk].reference.values
	y_stuff = DS_dict[kk].prediction.values

	# Compute statistics for entire profile:
	error_dict[kk]['rmse_tot'] = compute_RMSE_profile(y_stuff, x_stuff, which_axis=0)
	error_dict[kk]['bias_tot'] = np.nanmean(y_stuff - x_stuff, axis=0)
	error_dict[kk]['stddev'] = compute_RMSE_profile(y_stuff - error_dict[kk]['bias_tot'], x_stuff, which_axis=0)



# reduce unnecessary dimensions of height:
if DS_dict['0'].height.ndim == 2:
	height = DS_dict['0'].height[0,:]
RMSE_044 = error_dict['0']['rmse_tot']*1000.0		# in g kg-1
RMSE_045 = error_dict['1']['rmse_tot']*1000.0		# in g kg-1


fs = 28
fs_small = fs - 2
fs_dwarf = fs_small - 2
fs_micro = fs_dwarf - 2
msize = 7.0

f1 = plt.figure(figsize=(8,14))
a1 = plt.axes()

y_lim = np.array([0.0, height.max()])


# bias profiles:
a1.plot(RMSE_044, height, color=(0.067,0.29,0.769), linewidth=1.75, label='HATPRO K-band')
a1.plot(RMSE_045, height, color=(0.8,0,0), linewidth=1.75, label='All HATPRO and MiRAC-P freqs')


# legends:
lh, ll = a1.get_legend_handles_labels()
a1.legend(lh, ll, loc='upper right', fontsize=fs_micro-2)

# axis lims:
a1.set_ylim(bottom=y_lim[0], top=y_lim[1])
a1.set_xlim(left=0, right=0.9)

# set axis ticks, ticklabels and tick parameters:
a1.minorticks_on()
a1.tick_params(axis='both', labelsize=fs_dwarf)

# grid:
a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

# labels:
a1.set_xlabel("RMSE$_{\mathrm{q}}$ ($\mathrm{g}\,\mathrm{kg}^{-1}$)", fontsize=fs)
a1.set_ylabel("Height (m)", fontsize=fs)


plotname = f"NN_syn_ret_eval_q_rmse_profile_044_045"
f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')

pdb.set_trace()