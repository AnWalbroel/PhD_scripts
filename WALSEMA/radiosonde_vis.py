import matplotlib as mpl
mpl.use("TkAgg")

import numpy as np
import pdb
import sys
import os
import matplotlib.pyplot as plt
import glob

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from import_data import import_radiosondes_PS131_txt
from met_tools import wspeed_wdir_to_u_v


"""
	Simple script to visualize radiosondes gathered during Polarstern cruise PS131.
	IWV, temperature and humidity profiles, and wind barbs will be visualized.
	- import radiosonde data
	- eventually compute some more meteorological variables
	- visualize
"""


# Paths
path_data = "/mnt/d/Studium_NIM/work/Data/WALSEMA/radiosondes/"		# path of tab data
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/radiosondes/"


# more settings:
fs = 16		# font size in plots
fs_small = fs - 2
fs_dwarf = fs_small -2
save_figures = True


# import data:
files = sorted(glob.glob(path_data + "*.txt"))[-10:]
sonde_dict = import_radiosondes_PS131_txt(files)


# visualization:
c_temp = (0.75,0,0)
c_relhum = (0,0,0.75)
c_barbs = (0,0,0)

# axis limits
height_lims_tropo = [0, 15000]
height_lims_bl = [0, 2000]
temp_lims_tropo = [-70.0, 25.0]
temp_lims_bl = [-10.0, 25.0]
relhum_lims = [0, 100]
height_ip = np.arange(height_lims_tropo[0], height_lims_tropo[1]+0.01, 500.0)

for k, idx_str in enumerate(sonde_dict.keys()):

	# compute u and v component of wind and interpolate to coarser grid:
	sonde_dict[idx_str]['u'], sonde_dict[idx_str]['v'] = wspeed_wdir_to_u_v(sonde_dict[idx_str]['wspeed'], 
																			sonde_dict[idx_str]['wdir'], 
																			convention='from')
	sonde_dict[idx_str]['u'] = np.interp(np.arange(height_lims_tropo[0], height_lims_tropo[1]+0.01, 500.0),
											sonde_dict[idx_str]['height'], sonde_dict[idx_str]['u'])
	sonde_dict[idx_str]['v'] = np.interp(np.arange(height_lims_tropo[0], height_lims_tropo[1]+0.01, 500.0),
											sonde_dict[idx_str]['height'], sonde_dict[idx_str]['v'])


	f1 = plt.figure(figsize=(14,10))
	ax_tropo = plt.subplot2grid((1,2), (0,0))	# whole troposphere (0 - 15 km)
	ax_bl = plt.subplot2grid((1,2), (0,1))		# boundary layer (0 - 2 km)


	# plotting:
	ax_tropo.plot(sonde_dict[idx_str]['temp']-273.15, sonde_dict[idx_str]['height'], color=c_temp,
					linewidth=1.2)
	ax_tropo_rh = ax_tropo.twiny()
	ax_tropo_rh.plot(sonde_dict[idx_str]['relhum']*100.0, sonde_dict[idx_str]['height'], color=c_relhum,
					linewidth=1.2)
	ax_tropo.barbs(np.full((height_ip.shape), 0.95*(temp_lims_tropo[1] - temp_lims_tropo[0]) + temp_lims_tropo[0]), 
					height_ip, sonde_dict[idx_str]['u'], sonde_dict[idx_str]['v'],
					length=4.5, pivot='middle', barbcolor=(0,0,0,0.65), zorder=9999,
					rounding=True,
					)

	ax_bl.plot(sonde_dict[idx_str]['temp']-273.15, sonde_dict[idx_str]['height'], color=c_temp,
					linewidth=1.2)
	ax_bl_rh = ax_bl.twiny()
	ax_bl_rh.plot(sonde_dict[idx_str]['relhum']*100.0, sonde_dict[idx_str]['height'], color=c_relhum,
					linewidth=1.2)


	# add text:
	ax_tropo.text(0.9, 0.98, (f"IWV: {sonde_dict[idx_str]['IWV']:.2f}" + "$\,$mm" + "\n"
					"T$_{850\,\mathrm{hpa}}$: " + f"{np.interp(85000.0, sonde_dict[idx_str]['pres'][::-1], sonde_dict[idx_str]['temp'][::-1])-273.15:.2f}" + 
					"$\,^{\circ}$C"), 
					fontsize=fs, ha='right', va='top', transform=ax_tropo.transAxes,
					bbox={'facecolor': (1.0, 1.0, 1.0, 0.65), 'edgecolor': (0,0,0), 'boxstyle': 'round'})


	# set axis limits:
	ax_tropo.set_xlim(temp_lims_tropo[0], temp_lims_tropo[1])
	ax_tropo.set_ylim(height_lims_tropo[0], height_lims_tropo[1])
	ax_tropo_rh.set_xlim(relhum_lims[0], relhum_lims[1])

	ax_bl.set_xlim(temp_lims_bl[0], temp_lims_bl[1])
	ax_bl.set_ylim(height_lims_bl[0], height_lims_bl[1])
	ax_bl_rh.set_xlim(relhum_lims[0], relhum_lims[1])


	# set tick params:
	ax_tropo.tick_params(axis='x', labelsize=fs_small, labelcolor=c_temp)
	ax_tropo_rh.tick_params(axis='x', labelsize=fs_small, labelcolor=c_relhum)
	ax_tropo.tick_params(axis='y', labelsize=fs_small)

	ax_bl.tick_params(axis='x', labelsize=fs_small, labelcolor=c_temp)
	ax_bl_rh.tick_params(axis='x', labelsize=fs_small, labelcolor=c_relhum)
	ax_bl.tick_params(axis='y', labelsize=fs_small)


	# grid:
	ax_tropo.grid(which='major', axis='both', alpha=0.4)
	ax_bl.grid(which='major', axis='both', alpha=0.4)


	# set labels:
	ax_tropo.set_xlabel("Temperature ($\,^{\circ}$C)", fontsize=fs)
	ax_tropo_rh.set_xlabel("Rel. humidity ($\%$)", fontsize=fs)
	ax_bl.set_xlabel("Temperature ($\,^{\circ}$C)", fontsize=fs)
	ax_bl_rh.set_xlabel("Rel. humidity ($\%$)", fontsize=fs)
	ax_tropo.set_ylabel("Height (m)", fontsize=fs)
	f1.suptitle(f"{sonde_dict[idx_str]['launch_time_npdt']}", fontweight='bold', fontsize=fs).set_position((0.5,0.95))

	if save_figures:
		plot_file = path_plots + os.path.basename(files[k])[:-4] + ".png"
		f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		print("Plot saved to: " + plot_file)
	else:
		plt.show()

	plt.close()