import numpy as np
import xarray as xr
import glob
import sys
import pdb
import datetime as dt
import os
from PIL import Image

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *


"""
	Simple and dynamic script for TB quicklooks and quickly checking out
	some TB features:
"""


# paths:
path_data = {	'hatpro': "/mnt/e/MiRAC-P/Y2022/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/tb_quicklooks/"


# settings:

nowdate = dt.datetime(2022,8,12)
enddate = dt.datetime.utcnow()
while nowdate <= enddate:

	date_mm = nowdate.month
	date_dd = nowdate.day
	# nc_files = sorted(glob.glob(path_data['hatpro'] + f"M{date_mm:02}/D{date_dd:02}/22{date_mm:02}{date_dd:02}.BRT.NC"))
	nc_files = sorted(glob.glob(path_data['hatpro'] + f"M{date_mm:02}/D{date_dd:02}/*.BRT.NC"))
	


	if len(nc_files) == 1:
		HAT_DS = xr.open_dataset(nc_files[0])



		# visualize: 
		fs = 14
		fs_small = fs - 2
		fs_dwarf = fs - 4
		marker_size = 15


		# define axes:
		f1 = plt.figure(figsize=(14,9))
		a_hat = plt.subplot2grid((2,1), (0,0))
		a_hat2 = plt.subplot2grid((2,1), (1,0))

		dt_fmt = mpl.dates.DateFormatter("%H:%M")


		# axis lims:

		# plot:

		# hatpro TBs:
		# # # freqs = np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
							# # # 51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000])
		# # # K_band_idx = range(7)
		# # # for kk in K_band_idx:
			# # # a_hat.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

		# # # V_band_idx = range(7,14)
		# # # for kk in V_band_idx:
			# # # a_hat2.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

		freqs = HAT_DS.Freq.values
		G_band_idx = range(6)
		for kk in G_band_idx:
			a_hat.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

		H_band_idx = range(6,7)
		for kk in H_band_idx:
			a_hat2.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

		# threshold = np.ones(HAT_DS.time.values.shape)*0.05
		# a_hat.plot(HAT_DS.time.values, HAT_DS.stability_rec1.values, color=(0,0,0))
		# a_hat2.plot(HAT_DS.time.values, HAT_DS.stability_rec2.values, color=(0,0,0))
		# a_hat.plot(HAT_DS.time.values, threshold, linestyle='dashed', color=(0,0,0))
		# a_hat2.plot(HAT_DS.time.values, threshold, linestyle='dashed', color=(0,0,0))


		# add figure identifier of subplots: a), b), ...


		# legends and colorbars:
		lh, ll = a_hat.get_legend_handles_labels()
		a_hat.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper right', markerscale=1.5)

		lh, ll = a_hat2.get_legend_handles_labels()
		a_hat2.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper right', markerscale=1.5)


		# set axis limits:
		# a_hat.set_ylim([0.0, 0.10])
		# a_hat2.set_ylim([0.0, 0.10])


		# set ticks and tick labels and parameters:
		a_hat.xaxis.set_major_formatter(dt_fmt)
		a_hat2.xaxis.set_major_formatter(dt_fmt)


		# grid:
		a_hat.grid(axis='y', which='major')
		a_hat2.grid(axis='y', which='major')


		# set labels:
		a_hat.set_ylabel("TB (K)", fontsize=fs)
		a_hat2.set_ylabel("TB (K)", fontsize=fs)
		a_hat.set_title(f"{os.path.basename(nc_file)}", fontsize=fs)


		# adjust axis spacing and geometries:

		plt.show()
		# f1.savefig(path_plots + "WALSEMA_hatpro_mirac-p_skycam_fog_onset_20220724.png", dpi=300, bbox_inches='tight')


		nowdate += dt.timedelta(days=1)

	elif len(nc_files) >= 2:
		# pdb.set_trace()
		for nc_file in nc_files:
			HAT_DS = xr.open_dataset(nc_file)


			# visualize: 
			fs = 14
			fs_small = fs - 2
			fs_dwarf = fs - 4
			marker_size = 15


			# define axes:
			f1 = plt.figure(figsize=(14,9))
			a_hat = plt.subplot2grid((2,1), (0,0))
			a_hat2 = plt.subplot2grid((2,1), (1,0))

			dt_fmt = mpl.dates.DateFormatter("%H:%M")


			# axis lims:

			# plot:

			# hatpro TBs:
			# freqs = np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
								# 51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000])
			# K_band_idx = range(7)
			# for kk in K_band_idx:
				# a_hat.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

			# V_band_idx = range(7,14)
			# for kk in V_band_idx:
				# a_hat2.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

			freqs = HAT_DS.Freq.values
			G_band_idx = range(6)
			for kk in G_band_idx:
				a_hat.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

			H_band_idx = range(6,7)
			for kk in H_band_idx:
				a_hat2.plot(HAT_DS.time.values, HAT_DS.TBs.values[:,kk], linewidth=1.2, label=f"{freqs[kk]:.2f} GHz")

			# threshold = np.ones(HAT_DS.time.values.shape)*0.05
			# a_hat.plot(HAT_DS.time.values, HAT_DS.stability_rec1.values, color=(0,0,0))
			# a_hat2.plot(HAT_DS.time.values, HAT_DS.stability_rec2.values, color=(0,0,0))
			# a_hat.plot(HAT_DS.time.values, threshold, linestyle='dashed', color=(0,0,0))
			# a_hat2.plot(HAT_DS.time.values, threshold, linestyle='dashed', color=(0,0,0))


			# add figure identifier of subplots: a), b), ...


			# legends and colorbars:
			lh, ll = a_hat.get_legend_handles_labels()
			a_hat.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper right', markerscale=1.5)

			lh, ll = a_hat2.get_legend_handles_labels()
			a_hat2.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper right', markerscale=1.5)


			# set axis limits:
			# a_hat.set_ylim([0.0, 0.10])
			# a_hat2.set_ylim([0.0, 0.10])


			# set ticks and tick labels and parameters:
			a_hat.xaxis.set_major_formatter(dt_fmt)
			a_hat2.xaxis.set_major_formatter(dt_fmt)


			# grid:
			a_hat.grid(axis='y', which='major')
			a_hat2.grid(axis='y', which='major')


			# set labels:
			a_hat.set_ylabel("stab (K)", fontsize=fs)
			a_hat2.set_ylabel("stab (K)", fontsize=fs)
			a_hat.set_title(f"{os.path.basename(nc_file)}", fontsize=fs)


			# adjust axis spacing and geometries:

			plt.show()
			# f1.savefig(path_plots + "WALSEMA_hatpro_mirac-p_skycam_fog_onset_20220724.png", dpi=300, bbox_inches='tight')


		nowdate += dt.timedelta(days=1)


	else:
		nowdate += dt.timedelta(days=1)
		continue