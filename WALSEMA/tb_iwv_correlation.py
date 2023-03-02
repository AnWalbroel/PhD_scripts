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
from met_tools import convert_rh_to_abshum
from import_data import import_radiosondes_PS131_txt


"""
	Simple and dynamic script for TB quicklooks and quickly checking out
	some TB features:
"""


# paths:
path_data = {	'hatpro': "/data/obs/campaigns/WALSEMA/atm/hatpro/l2/",
				'mirac-p': "/data/obs/campaigns/WALSEMA/atm/mirac-p/l1/"}
path_plots = "/net/blanc/awalbroe/Plots/WALSEMA/tb_quicklooks/"


# settings:
set_dict = {'save_figures': False,
			'date': dt.datetime(2022,7,16)}


# import data:
date_month = set_dict['date'].month
date_day = set_dict['date'].day
file = sorted(glob.glob((path_data['mirac-p'] + f"2022/{date_month:02}/{date_day:02}/WALSEMA_uoc_lhumpro-243-340_l1_tb_v01_" + 
							f"{set_dict['date'].strftime('%Y%m%d')}*.nc")))[0]
MIR_DS = xr.open_dataset(file)


# HATPRO: prw and clwvi
ret_vars = ['prw']
HAT_DS_dict = dict()
for var in ret_vars:
	file = sorted(glob.glob(path_data['hatpro'] + f"2022/{date_month:02}/{date_day:02}/ioppol_tro_mwr00_l2_{var}_v00_{set_dict['date'].strftime('%Y%m%d')}*.nc"))[0]
	HAT_DS_dict[var] = xr.open_dataset(file)

# Merge in retrieval dataset:
RET_DS = xr.Dataset({'time_sec': (['time'], numpydatetime64_to_epochtime(HAT_DS_dict['prw'].time.values))},
					coords=	{	'time': (['time'], HAT_DS_dict['prw']['time'].values)})
for var in ret_vars:
	RET_DS[var] = xr.DataArray(HAT_DS_dict[var][var].values, dims=['time'])


# limit time:
date_str_long = set_dict['date'].strftime('%Y-%m-%d')
RET_DS = RET_DS.sel(time=f"{date_str_long}")
MIR_DS = MIR_DS.sel(time=f"{date_str_long}")


# merge time grids: and need to repair RPGs time axis:
unique_time, time_idx = np.unique(MIR_DS.time.values, return_index=True)
MIR_DS = MIR_DS.isel(time=time_idx)
MIR_DS = MIR_DS.interp(coords={'time': RET_DS.time})


# filter for non-nan time steps:
nonnan_mir = np.where(~np.isnan(np.sum(MIR_DS.tb.values, axis=1)))[0]
nonnan_hat = np.where(~np.isnan(RET_DS.prw.values))[0]
nonnan_idx = np.intersect1d(nonnan_mir, nonnan_hat)

MIR_DS = MIR_DS.isel(time=nonnan_idx)
RET_DS = RET_DS.isel(time=nonnan_idx)


# Compute correlation for each channel for each hour:
n_freq = len(MIR_DS.freq_sb.values)
corr_coeff = np.ones((24,n_freq))
for jj in range(24):

	if jj < 23:
		MIR_DS_h = MIR_DS.sel(time=slice(f"{date_str_long} {jj:02}:00", f"{date_str_long} {jj+1:02}:00"))
		RET_DS_h = RET_DS.sel(time=slice(f"{date_str_long} {jj:02}:00", f"{date_str_long} {jj+1:02}:00"))

	else:
		MIR_DS_h = MIR_DS.sel(time=slice(f"{date_str_long} {jj:02}:00:00", f"{date_str_long} {jj:02}:59:59"))
		RET_DS_h = RET_DS.sel(time=slice(f"{date_str_long} {jj:02}:00:00", f"{date_str_long} {jj:02}:59:59"))

	for k in range(n_freq):
		corr_coeff[jj,k] = np.corrcoef(MIR_DS_h.tb.values[:,k], RET_DS_h.prw.values)[0,1]

overall_corr_coeff = np.ones((n_freq,))
for k in range(n_freq):
	overall_corr_coeff[k] = np.corrcoef(MIR_DS.tb.values[:,k], RET_DS.prw.values)[0,1]


for k in range(n_freq):
	plt.plot(corr_coeff[:,k], label=f"{MIR_DS.freq_sb.values[k]:.2f}")
plt.legend()
plt.xlabel("hour of 2022-07-16")
plt.ylabel("Pearson correlation coeff.")
plt.show()


# plotting some stuff:
for k in range(n_freq):
	f1 = plt.figure()
	a1 = plt.axes()


	a2 = a1.twinx()

	a1.plot(MIR_DS.time, MIR_DS.tb.values[:,k], label=f"{MIR_DS.freq_sb.values[k]:.2f}")
	a2.plot(RET_DS.time, RET_DS.prw.values, color=(0,0,0))

	a1.legend()

	plt.show()