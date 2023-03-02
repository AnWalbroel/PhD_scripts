import numpy as np
import xarray as xr
import glob
import sys
import pdb
import datetime as dt
from PIL import Image

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import *


"""
	Simple and dynamic script for TB quicklooks and quickly checking out
	some TB features:
"""


# paths:
path_data = {	'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l1/",
				'mirac-p': "/data/obs/campaigns/WALSEMA/atm/mirac-p/l1/",
				'gopro': "/data/obs/campaigns/WALSEMA/sky_cam/vis/",
				'ir': "/data/obs/campaigns/WALSEMA/sky_cam/ir/"}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/tb_quicklooks/"


# settings:
set_dict = {'time_lims': ["18:45", "19:30"]}


# import data:
file = path_data['mirac-p'] + "2022/07/24/WALSEMA_uoc_lhumpro-243-340_l1_tb_v01_20220724000000.nc"
MIR_DS = xr.open_dataset(file)

file = path_data['hatpro'] + "2022/07/24/ioppol_tro_mwr00_l1_tb_v00_20220724000000.nc"
HAT_DS = xr.open_dataset(file)

# limit time:
HAT_DS = HAT_DS.sel(time=slice(f"2022-07-24T{set_dict['time_lims'][0]}:00", f"2022-07-24T{set_dict['time_lims'][1]}:00"))
MIR_DS = MIR_DS.sel(time=slice(f"2022-07-24T{set_dict['time_lims'][0]}:00", f"2022-07-24T{set_dict['time_lims'][1]}:00"))


# gopro and IR: import via PIL
chosen_times = ["18:53", "19:13", "19:20", "19:25"]
gopro_images = dict()
ir_images = dict()
for k, ct in enumerate(chosen_times):
	file = glob.glob(path_data['gopro'] + "20220724/" + f"GOPRO*{ct.replace(':', '')}*.JPG")[0]
	gopro_images[str(k)] = {'image': Image.open(file),
							'time': dt.datetime.strptime(file[-18:-4], "%Y%m%d%H%M%S")}

	file_ir = glob.glob(path_data['ir'] + "2022/07/24/" + f"*20220724{ct.replace(':', '')}*.jpg")[0]
	ir_images[str(k)] = {	'image': Image.open(file_ir),
							'time': dt.datetime.strptime(file_ir[-16:-4], "%Y%m%d%H%M")}



# visualize: 
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15


# define axes:
f1 = plt.figure(figsize=(14,10))
a_hat = plt.subplot2grid((4,4), (0,0), colspan=4)
a_mir = plt.subplot2grid((4,4), (1,0), colspan=4)
a_go0 = plt.subplot2grid((4,4), (2,0))
a_go1 = plt.subplot2grid((4,4), (2,1))
a_go2 = plt.subplot2grid((4,4), (2,2))
a_go3 = plt.subplot2grid((4,4), (2,3))
a_ir0 = plt.subplot2grid((4,4), (3,0))
a_ir1 = plt.subplot2grid((4,4), (3,1))
a_ir2 = plt.subplot2grid((4,4), (3,2))
a_ir3 = plt.subplot2grid((4,4), (3,3))

dt_fmt = mpl.dates.DateFormatter("%H:%M")


# axis lims:
time_lims = [np.datetime64(f"2022-07-24T{set_dict['time_lims'][0]}"), np.datetime64(f"2022-07-24T{set_dict['time_lims'][1]}")]
ax_lims = [[14.0, 36.0], [152.5, 168.0]]


# plot:

# hatpro TBs:
K_band_idx = select_MWR_channels(HAT_DS.tb.values, HAT_DS.freq_sb.values, band='K', return_idx=2)
for kk, ff in enumerate(HAT_DS.freq_sb.values[K_band_idx]):
	a_hat.plot(HAT_DS.time.values, HAT_DS.tb.values[:,kk], linewidth=1.2, label=f"{ff:.2f} GHz")

# mirac-p TBs:
a_mir.plot(MIR_DS.time.values, MIR_DS.tb.values[:,6], linewidth=1.2, label=f"{MIR_DS.freq_sb.values[6]:.2f} GHz")

# gopro images:
for k, ax in enumerate([a_go0, a_go1, a_go2, a_go3]):
	ax.imshow(gopro_images[str(k)]['image'])

# ir images:
for k, ax in enumerate([a_ir0, a_ir1, a_ir2, a_ir3]):
	ax.imshow(ir_images[str(k)]['image'])


# add auxiliary lines:
for k, ax in enumerate([a_hat, a_mir]):
	for ct in chosen_times:
		ax.plot(np.array([np.datetime64(f"2022-07-24T{ct}"), np.datetime64(f"2022-07-24T{ct}")]),
				ax_lims[k], linestyle='dashed', color=(0,0,0), linewidth=0.75)


# add time stamp texts for gopro and IR images:
for k, ax in enumerate([a_go0, a_go1, a_go2, a_go3]):
	ax.text(0.5, -0.01, f"{gopro_images[str(k)]['time']:%Y-%m-%d %H:%M}", fontsize=fs_small,
			ha='center', va='top', transform=ax.transAxes)

for k, ax in enumerate([a_ir0, a_ir1, a_ir2, a_ir3]):
	ax.text(0.5, -0.01, f"{ir_images[str(k)]['time']:%Y-%m-%d %H:%M}", fontsize=fs_small,
			ha='center', va='top', transform=ax.transAxes)


# add figure identifier of subplots: a), b), ...
a_hat.text(0.02, 0.95, "HATPRO", fontsize=fs, fontweight='bold', ha='left', va='top',
			transform=a_hat.transAxes)
a_mir.text(0.02, 0.95, "MiRAC-P", fontsize=fs, fontweight='bold', ha='left', va='top',
			transform=a_mir.transAxes)
a_go0.text(0.02, 1.01, "GOPRO", fontsize=fs, fontweight='bold', ha='left', va='bottom',
			transform=a_go0.transAxes)
a_ir0.text(0.02, 1.01, "IR", fontsize=fs, fontweight='bold', ha='left', va='bottom',
			transform=a_ir0.transAxes)


# legends and colorbars:
lh, ll = a_hat.get_legend_handles_labels()
a_hat.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper right', markerscale=1.5)

lh, ll = a_mir.get_legend_handles_labels()
a_mir.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc="upper right", markerscale=1.5)


# set axis limits:
a_hat.set_xlim(time_lims)
a_mir.set_xlim(time_lims)
a_hat.set_ylim(ax_lims[0])
a_mir.set_ylim(ax_lims[1])


# set ticks and tick labels and parameters:
a_hat.xaxis.set_major_formatter(dt_fmt)
a_mir.xaxis.set_major_formatter(dt_fmt)

for ax in [a_go0, a_go1, a_go2, a_go3, a_ir0, a_ir1, a_ir2, a_ir3]:
	ax.xaxis.set_ticks([])
	ax.yaxis.set_ticks([])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])


# grid:
a_hat.grid(axis='y', which='major')
a_mir.grid(axis='y', which='major')


# set labels:
a_hat.set_ylabel("TB (K)", fontsize=fs)
a_mir.set_xlabel("2022-07-24", fontsize=fs)
a_mir.set_ylabel("TB (K)", fontsize=fs)
f1.suptitle("Fog onset 2022-07-24", fontweight='bold', fontsize=fs)


# adjust axis spacing and geometries:
plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35)

# plt.show()
f1.savefig(path_plots + "WALSEMA_hatpro_mirac-p_skycam_fog_onset_20220724.pdf", dpi=1000, bbox_inches='tight')
# f1.savefig(path_plots + "WALSEMA_hatpro_mirac-p_skycam_fog_onset_20220724.jpg", dpi=1000, bbox_inches='tight')