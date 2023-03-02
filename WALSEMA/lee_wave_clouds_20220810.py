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

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from data_tools import *
from met_tools import convert_rh_to_abshum
from import_data import import_radiosondes_PS131_txt


"""
	Simple and dynamic script for TB quicklooks and quickly checking out
	some TB features:
"""


# paths:
path_data = {	'hatpro': "/mnt/e/HATPRO/Y2022/",
				'mirac-p': "/mnt/e/MiRAC-P/Y2022/",
				'gopro': "/mnt/d/heavy_data/WALSEMA/SkyCam_GOPRO/",
				'ir': "/mnt/d/heavy_data/WALSEMA/SkyCam_IR/",
				'coeffs': "/mnt/d/Studium_NIM/work/Data/HATPRO_regression/",
				'radiosondes': "/mnt/d/Studium_NIM/work/Data/WALSEMA/radiosondes/"}
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/tb_quicklooks/"


# settings:
set_dict = {'time_lims': ["07:30", "08:45"],
			'save_figures': True,
			'date': dt.datetime(2022,8,10)}


# import data:
date_month = set_dict['date'].month
date_day = set_dict['date'].day
file = path_data['mirac-p'] + f"M{date_month:02}/D{date_day:02}/ELE90_{set_dict['date'].strftime('%Y%m%d')[2:]}.BRT.NC"
MIR_DS = xr.open_dataset(file)

file = path_data['hatpro'] + f"M{date_month:02}/D{date_day:02}/ELE90_{set_dict['date'].strftime('%Y%m%d')[2:]}.BRT.NC"
HAT_DS = xr.open_dataset(file)

# limit time:
date_str_long = set_dict['date'].strftime('%Y-%m-%d')
HAT_DS = HAT_DS.sel(time=slice(f"{date_str_long}T{set_dict['time_lims'][0]}:00", f"{date_str_long}T{set_dict['time_lims'][1]}:00"))
MIR_DS = MIR_DS.sel(time=slice(f"{date_str_long}T{set_dict['time_lims'][0]}:00", f"{date_str_long}T{set_dict['time_lims'][1]}:00"))


# gopro and IR: import via PIL
chosen_times = ["07:40", "08:07", "08:30"]
gopro_images = dict()
ir_images = dict()
for k, ct in enumerate(chosen_times):
	# watch out that digits after selected time are respected:
	file = glob.glob(path_data['gopro'] + f"{set_dict['date'].strftime('%Y%m%d')}/" + 
						f"GOPRO{set_dict['date'].strftime('%Y%m%d')}{ct.replace(':', '')}[0-9][0-9]*.JPG")[0]
	gopro_images[str(k)] = {'image': Image.open(file),
							'time': dt.datetime.strptime(file[-18:-4], "%Y%m%d%H%M%S")}

	file_ir = glob.glob(path_data['ir'] + f"{set_dict['date'].year}/{date_month:02}/{date_day:02}/" + 
						f"*{set_dict['date'].strftime('%Y%m%d')}{ct.replace(':', '')}.jpg")[0]
	ir_images[str(k)] = {	'image': Image.open(file_ir),
							'time': dt.datetime.strptime(file_ir[-16:-4], "%Y%m%d%H%M")}


# retrieve IWV and LWP from HATPRO data:
# load coefficients:
COEFF_DS = xr.open_dataset(path_data['coeffs'] + "MOSAiC_HATPRO_regression_coeffs.nc")
COEFF_BL_DS = xr.open_dataset(path_data['coeffs'] + "MOSAiC_HATPRO_regression_coeffs_BL.nc")
HAT_DS['freq'] = xr.DataArray(np.array([22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
								51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000]), dims=['number_frequencies'])
HAT_DS['frequencies'] = HAT_DS['freq']
K_idx = select_MWR_channels(HAT_DS['TBs'].values, HAT_DS['freq'].values, band='K', return_idx=2)
V_idx = select_MWR_channels(HAT_DS['TBs'].values, HAT_DS['freq'].values, band='V', return_idx=2)

n_height_ret = len(COEFF_DS.height.values)
ret_vars = ['prw', 'clwvi']
RET_DS = xr.Dataset({	'time_sec': (['time'], numpydatetime64_to_epochtime(HAT_DS.time.values))},
					coords=	{	'time': (['time'], HAT_DS['time'].values)})

# build observation matrix according to the regression (for each height, if a profile
# is considered).
for var in ret_vars:
	if var not in ['ta', 'ta_bl']:
		K_reg_obs = build_K_reg(HAT_DS['TBs'].values[:,K_idx], order=2)
		RET_DS[var] = xr.DataArray(K_reg_obs.dot(COEFF_DS[f"c_{var}"].values), dims=['time'])


# load radiosonde data: concat each sonde to generate a (sonde_launch x height) dict.
# interpolation of radiosonde data to regular grid requried to get a 2D array
files = sorted(glob.glob(path_data['radiosondes'] + "*.txt"))
sonde_dict_temp = import_radiosondes_PS131_txt(files)
new_height = np.arange(0.0, 20000.0001, 20.0)
n_height = len(new_height)
n_sondes = len(sonde_dict_temp.keys())
sonde_dict = {'temp': np.full((n_sondes, n_height), np.nan),
				'pres': np.full((n_sondes, n_height), np.nan),
				'relhum': np.full((n_sondes, n_height), np.nan),
				'height': np.full((n_sondes, n_height), np.nan),
				'wdir': np.full((n_sondes, n_height), np.nan),
				'wspeed': np.full((n_sondes, n_height), np.nan),
				'q': np.full((n_sondes, n_height), np.nan),
				'IWV': np.full((n_sondes,), np.nan),
				'launch_time': np.zeros((n_sondes,)),			# in sec since 1970-01-01 00:00:00 UTC
				'launch_time_npdt': np.full((n_sondes,), np.datetime64("1970-01-01T00:00:00"))}

# interpolate to new grid:
for idx in sonde_dict_temp.keys():
	for key in sonde_dict_temp[idx].keys():
		if key not in ["height", "IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx),:] = np.interp(new_height, sonde_dict_temp[idx]['height'], sonde_dict_temp[idx][key])
		elif key == "height":
			sonde_dict[key][int(idx),:] = new_height
		elif key in ["IWV", 'launch_time', 'launch_time_npdt']:
			sonde_dict[key][int(idx)] = sonde_dict_temp[idx][key]
del sonde_dict_temp

# compute absolute humidity:
sonde_dict['rho_v'] = convert_rh_to_abshum(sonde_dict['temp'], sonde_dict['relhum'])


# visualize: 
fs = 14
fs_small = fs - 2
fs_dwarf = fs - 4
marker_size = 15
c_RS = (1,0.435,0)			# radiosondes
c_H = (0.067,0.29,0.769)	# HATPRO IWV


# define axes:
f1 = plt.figure(figsize=(11,10))
a_hat = plt.subplot2grid((4,3), (0,0), colspan=3)
a_mir = plt.subplot2grid((4,3), (1,0), colspan=3)
a_go0 = plt.subplot2grid((4,3), (2,0))
a_go1 = plt.subplot2grid((4,3), (2,1))
a_go2 = plt.subplot2grid((4,3), (2,2))
a_ir0 = plt.subplot2grid((4,3), (3,0))
a_ir1 = plt.subplot2grid((4,3), (3,1))
a_ir2 = plt.subplot2grid((4,3), (3,2))

dt_fmt = mpl.dates.DateFormatter("%H:%M")


# axis lims:
time_lims = [np.datetime64(f"{date_str_long}T{set_dict['time_lims'][0]}"), np.datetime64(f"{date_str_long}T{set_dict['time_lims'][1]}")]
ax_lims = [[10.0, 20.0], [175.0, 195.0]]
lwp_lims = [0.0, 150.0]


# plot:

# hatpro products:
a_hat.plot(RET_DS.time.values, RET_DS.prw.values, linewidth=1.2, color=c_H, label="HATPRO IWV")
a_hat.plot(sonde_dict['launch_time_npdt'], sonde_dict['IWV'], linestyle='none', 
				marker='.', linewidth=0.5,
				markersize=marker_size, markerfacecolor=c_RS, markeredgecolor=(0,0,0), 
				markeredgewidth=0.5, label='Radiosonde IWV')
a_hat.plot([np.nan, np.nan], [np.nan, np.nan], linewidth=1.2, color=(0,0,0), label="LWP")
a_hat_lwp = a_hat.twinx()
a_hat_lwp.plot(RET_DS.time.values, RET_DS.clwvi.values*1000.0, linewidth=1.2, color=(0,0,0))

# mirac-p TBs:
a_mir.plot(MIR_DS.time.values, MIR_DS.TBs.values[:,6], linewidth=1.2, label=f"{MIR_DS.Freq.values[6]:.2f} GHz")

# gopro images:
for k, ax in enumerate([a_go0, a_go1, a_go2]):
	ax.imshow(gopro_images[str(k)]['image'])

# ir images:
for k, ax in enumerate([a_ir0, a_ir1, a_ir2]):
	ax.imshow(ir_images[str(k)]['image'])


# add auxiliary lines:
for k, ax in enumerate([a_hat, a_mir]):
	for ct in chosen_times:
		ax.plot(np.array([np.datetime64(f"{date_str_long}T{ct}"), np.datetime64(f"{date_str_long}T{ct}")]),
				ax_lims[k], linestyle='dashed', color=(0,0,0), linewidth=0.75)


# add time stamp texts for gopro and IR images:
for k, ax in enumerate([a_go0, a_go1, a_go2]):
	ax.text(0.5, -0.01, f"{gopro_images[str(k)]['time']:%Y-%m-%d %H:%M}", fontsize=fs_small,
			ha='center', va='top', transform=ax.transAxes)

for k, ax in enumerate([a_ir0, a_ir1, a_ir2]):
	ax.text(0.5, -0.01, f"{ir_images[str(k)]['time']:%Y-%m-%d %H:%M}", fontsize=fs_small,
			ha='center', va='top', transform=ax.transAxes)


# add figure identifier of subplots: a), b), ...
a_hat.text(0.02, 0.95, "(a) IWV/LWP", fontsize=fs, fontweight='bold', ha='left', va='top',
			transform=a_hat.transAxes)
a_mir.text(0.02, 0.95, "(b) MiRAC-P", fontsize=fs, fontweight='bold', ha='left', va='top',
			transform=a_mir.transAxes)
a_go0.text(0.02, 1.01, "(c) GOPRO", fontsize=fs, fontweight='bold', ha='left', va='bottom',
			transform=a_go0.transAxes)
a_ir0.text(0.02, 1.01, "(d) IR", fontsize=fs, fontweight='bold', ha='left', va='bottom',
			transform=a_ir0.transAxes)


# legends and colorbars:
lh, ll = a_hat.get_legend_handles_labels()
a_hat.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc='upper center', markerscale=1.5)

lh, ll = a_mir.get_legend_handles_labels()
a_mir.legend(handles=lh, labels=ll, fontsize=fs_dwarf, ncol=7, loc="upper center", markerscale=1.5)


# set axis limits:
a_hat.set_xlim(time_lims)
a_hat_lwp.set_xlim(time_lims)
a_mir.set_xlim(time_lims)
a_hat.set_ylim(ax_lims[0])
a_hat_lwp.set_ylim(lwp_lims)
a_mir.set_ylim(ax_lims[1])


# set ticks and tick labels and parameters:
a_hat.xaxis.set_major_formatter(dt_fmt)
a_mir.xaxis.set_major_formatter(dt_fmt)

for ax in [a_go0, a_go1, a_go2, a_ir0, a_ir1, a_ir2]:
	ax.xaxis.set_ticks([])
	ax.yaxis.set_ticks([])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])


# grid:
a_hat.grid(axis='y', which='major')
a_mir.grid(axis='y', which='major')


# set labels:
a_hat.set_ylabel("IWV (kg$\,$m$^{-2}$)", fontsize=fs, color=c_H)
a_hat_lwp.set_ylabel("LWP (g$\,$m$^{-2}$)", fontsize=fs)
a_mir.set_xlabel(f"{date_str_long}", fontsize=fs)
a_mir.set_ylabel("TB (K)", fontsize=fs)
f1.suptitle(f"Lee waves {date_str_long}", fontweight='bold', fontsize=fs)


# adjust axis spacing and geometries:
plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35)


if set_dict['save_figures']:
	f1.savefig(path_plots + f"WALSEMA_hatpro_mirac-p_skycam_lee_waves_Scoresby_Sund_{set_dict['date'].strftime('%Y%m%d')}.pdf", dpi=1000, bbox_inches='tight')
	# f1.savefig(path_plots + f"WALSEMA_hatpro_mirac-p_skycam_lee_waves_Scoresby_Sund_{set_dict['date'].strftime('%Y%m%d')}.png", dpi=1000, bbox_inches='tight')
else:
	plt.show()