import numpy as np
import pdb
import xarray as xr
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import *


"""
	Very simple LWP retrieval based on TB amplitudes of low freq
	channels with sufficient transparency (i.e., K band or 90 GHz).

	- Import TBs
	- Identify LWP from filename
	- visualize TB amplitude against LWP
	- profit
"""


path_data = ("/net/blanc/awalbroe/Data/HALO_AC3/fwd_sim_dropsondes/" +
			"pam_out_sonde_art_cloud/HALO-AC3_HALO_Dropsondes_20220321_RF08/")
path_output = "/net/blanc/awalbroe/Plots/HALO_AC3_quicklooks/"
save_figures = True

# import data:
files = sorted(glob.glob(path_data + "*.nc"))
DS = xr.open_mfdataset(files, concat_dim='grid_x', combine='nested')
TB = DS.tb.values[:,0,0,0,:,:].mean(axis=-1)
TB, freq = Fband_double_side_band_average(TB, DS.frequency.values)
TB, freq = Gband_double_side_band_average(TB, freq)
LWP = np.array([0,50,100,150,250,500,750])



# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15
for ix, ff in enumerate(freq):

	TB_amp = TB[:,ix] - TB[0,ix]

	# Plot:
	fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True)

	# ax lims:
	y_lims = [0, 800]


	# plotting:
	plt.plot(TB_amp, LWP, color=(0,0,0), marker='o', markeredgecolor=(0,0,0), markerfacecolor=(0.65,0.65,0.65),
				markersize=9, linewidth=1.15)

	# add figure identifier of subplots: a), b), ...
	ax.text(0.02, 0.98, f"{freq[ix]:.2f}", fontsize=fs_small, fontweight='bold', ha='left', va='top', transform=ax.transAxes)


	# set axis limits:
	ax.set_ylim(bottom=y_lims[0], top=y_lims[1])


	# x tick parameters:
	ax.tick_params(axis='x', labelsize=fs_small)

	# ytick parameters and grid and y labels:
	ax.tick_params(axis='y', labelsize=fs_small)
	ax.grid(which='major', axis='both', alpha=0.4)
	ax.set_ylabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)

	# set labels:
	ax.set_xlabel("TB$_{\mathrm{LWP}}$ - TB$_{\mathrm{cloudfree}}$ (K)", fontsize=fs)

	if not os.path.exists(path_output):
		os.makedirs(path_output)

	if save_figures:
		plot_name = f"HALO-AC3_HALO_hamp_LWP_{ff:03.2f}.png"
		fig.savefig(path_output + plot_name, dpi=400, bbox_inches='tight')
		print(f"Plot saved to {path_output + plot_name}.")
		plt.close()
	else:
		plt.show()