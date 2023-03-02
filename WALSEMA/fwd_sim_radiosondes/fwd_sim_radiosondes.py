import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import glob
import pdb
import os

import sys
sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from import_data import import_radiosondes_PS131_txt
from met_tools import wspeed_wdir_to_u_v
from data_tools import Gband_double_side_band_average, select_MWR_channels
from run_pamtra import run_pamtra_run

"""
	This script is used to forward simulate the PS131 radiosondes using PAMTRA.
	No emission by cloud liquid and no scattering by ice particles is assumed.
	
"""


# path of data:
path_radiosondes = "/mnt/d/Studium_NIM/work/Data/WALSEMA/radiosondes/"		# path of tab data
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/radiosondes/fwd_sim_radiosondes/tb_zenith_atm/"
path_output = path_radiosondes + "fwd_sim_radiosondes/"

# create plot dir if not available:
if not os.path.exists(path_plots):
	os.makedirs(path_plots)


# Other settings:
settings_dict = {
				'interpolate': True,				# if True, the radiosonde data will be interpolated to
													# a regular grid
				'elevation': 90.0,					# elevation angle of sim. TBs in deg
				'sst': 273.15,						# sea surface temperature in K
				'lat': 79.0,						# dummy latitude in deg N
				'lon': 5.0,							# dummy longitude in deg E
				'obs_height': np.array([20.0]),		# height of radiosonde launch in m
				'salinity': 32.5,					# sea water salinity in PSU
				'freqs': np.array([	22.240, 23.040, 23.840, 25.440, 26.240, 27.840, 31.400,
									51.260, 52.280, 53.860, 54.940, 56.660, 57.300, 58.000,
									175.810, 178.310, 179.810, 180.810, 181.810, 182.710,
									183.910, 184.810, 185.810, 186.810, 188.310, 190.810,
									243.000]),
				'sfc_refl': 'S',
				'liq_mod': "TKC"					# default: TKC
				}
set_dict = {
				'plot_tb_ele_angles': False,		# plots TBs of a radiosonde for different elevation angles
				'print_zenith_tbs': False,			# prints zenith TBs in terminal
				'save_tbs': True,					# save TBs to netCDF
			}


# import data:
files = sorted(glob.glob(path_radiosondes + "*.txt"))
filter_sondes = [17,18,42,43,46,51,54,55,57,116,129]
files_old = files
files = list()
for filtered in filter_sondes: files.append(files_old[filtered])
sonde_dict = import_radiosondes_PS131_txt(files)

# find the lowest common max altitude reached by the sondes (as top for interpolation):
min_common_max_alt = np.floor(np.array([sonde_dict[key]['height'].max() for key in sonde_dict.keys()]).min()*0.001)*1000.0

for jj, idx in enumerate(sonde_dict.keys()):

	# interpolate if desired: interpolate to the lowest common max altitude reached by the sondes:
	if settings_dict['interpolate']:
		new_height = np.arange(10.0, min_common_max_alt + 0.0001, 20.0)
		sonde_dict[idx]['temp'] = np.interp(new_height, sonde_dict[idx]['height'], sonde_dict[idx]['temp'])
		sonde_dict[idx]['pres'] = np.interp(new_height, sonde_dict[idx]['height'], sonde_dict[idx]['pres'])
		sonde_dict[idx]['relhum'] = np.interp(new_height, sonde_dict[idx]['height'], sonde_dict[idx]['relhum'])
		sonde_dict[idx]['wdir'] = np.interp(new_height, sonde_dict[idx]['height'], sonde_dict[idx]['wdir'])
		sonde_dict[idx]['wspeed'] = np.interp(new_height, sonde_dict[idx]['height'], sonde_dict[idx]['wspeed'])
		sonde_dict[idx]['height'] = new_height

	# compute u and v components of wind
	sonde_dict[idx]['u'], sonde_dict[idx]['v'] = wspeed_wdir_to_u_v(sonde_dict[idx]['wspeed'], 
																sonde_dict[idx]['wdir'], 
																	convention='from')

	# convert relative humidity to %:
	sonde_dict[idx]['relhum'] *= 100.0

	# start radiosonde simulation
	pam = run_pamtra_run(sonde_dict[idx], settings_dict)


	# print TBs: angles: 0: nadir; -1: zenith <<->> angle[0] = 180.0; angle[-1] = 0.0
	if set_dict['print_zenith_tbs']:
		pdb.set_trace()
		TB = pam.r['tb'][0,0,0,-1,:,:].mean(axis=-1)
		TB, freqs = Gband_double_side_band_average(np.broadcast_to(TB, (1,TB.shape[0])), settings_dict['freqs'])

		print(f"Radiosonde {idx}: {files[jj][-22:-12]}Z")
		for ii, ff in enumerate(freqs):	print(f"{ff:.2f} GHz:\t{TB[0,ii]:.2f} K")

		print(32*"-")
		print("\n")


	fs = 16
	fs_small = fs - 2
	fs_dwarf = fs_small - 2

	if set_dict['plot_tb_ele_angles']:

		# get TBs: angles: 0: nadir; -1: zenith <<->> angle[0] = 180.0; angle[-1] = 0.0
		TB = pam.r['tb'][0,0,0,:,:,:].mean(axis=-1)
		TB, freqs = Gband_double_side_band_average(TB, settings_dict['freqs'])
		
		# select some desired angles:
		# idx_ang = np.where((pam.r['angles_deg'] >= 0.0) & (pam.r['angles_deg'] <= 90.0))[0]		# upper hemisphere (sky)
		idx_ang = np.where((pam.r['angles_deg'] >= 90.0) & (pam.r['angles_deg'] <= 180.0))[0]		# lower hemisphere (ocean)
		angles_deg = pam.r['angles_deg'][idx_ang]
		TB = TB[idx_ang,:]


		###########################
		import pandas as pd
		TB_obs = pd.read_csv("/mnt/d/Studium_NIM/work/Data/WALSEMA/ocean_scan_mirror_test_hatpro.csv")
		###########################

		# select frequencies:
		for band in ['K', 'V', 'G', '243/340']:
			TB_sel, freqs_sel = select_MWR_channels(TB, freqs, band=band)

			cmap = plt.cm.get_cmap('Dark2', len(freqs_sel))

			f1 = plt.figure(figsize=(10,7))
			a1 = plt.axes()

			# plotting:
			for kk, freq in enumerate(freqs_sel):
				a1.plot(angles_deg, TB_sel[:,kk], color=cmap(kk), linewidth=1.2, label=f"{freq:.2f} GHz")

			#########################
				if freq == 58.:
					a1.plot(180.0-TB_obs['zenith_angle'], TB_obs["58"], color=cmap(kk), linewidth=1.2, 
							linestyle='dotted')
				else:
					a1.plot(180.0-TB_obs['zenith_angle'], TB_obs[str(freq)], color=cmap(kk), linewidth=1.2, 
							linestyle='dotted')

			a1.plot([np.nan,np.nan], [np.nan,np.nan], linestyle='dotted', linewidth=1.2, color=(0,0,0),
						label='obs')
			#########################


			# legends or colorbars:
			lh, ll = a1.get_legend_handles_labels()
			a1.legend(handles=lh, labels=ll, loc='best', fontsize=fs_small, markerscale=1.5)

			# set axis limits:
			###
			###

			# set tick parameters:
			a1.tick_params(axis='both', labelsize=fs_dwarf)

			# grid:
			a1.grid(which='both', axis='both', alpha=0.4)

			# set labels:
			a1.set_xlabel("Zenith angle ($^{\circ}$)", fontsize=fs)
			a1.set_ylabel("TB (K)", fontsize=fs)
			a1.set_title(f"Radiosonde: {files[jj][-22:-12]}Z", fontsize=fs)


			# plot_file = path_plots + f"Simulated_TBs_{band.replace('/', '-')}_zen_angles_radiosonde_{files[jj][-22:-12]}Z.png"
			plot_file = path_plots + f"Simulated_TBs_{band.replace('/', '-')}_zen_angles_ocean_radiosonde_{files[jj][-22:-12]}Z.png"
			f1.savefig(plot_file, dpi=300, bbox_inches='tight')
		# plt.show()

	if set_dict['save_tbs']:

		# save output:
		filename_out = path_output + os.path.basename(files[jj])[:-4] + "_pamtra.nc"
		pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)
		# pam.r['tb'].shape is x,y,outlevel,angles(2x as many as for emissivity),freq,polarization; first angle=nadir, last: zenith