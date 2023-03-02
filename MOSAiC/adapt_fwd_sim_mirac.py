import numpy as np
import glob
import pdb
import os
import datetime as dt
import xarray as xr
from import_data import *
from data_tools import *
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, PercentFormatter)


"""
	Here, forward simulated radiosondes will be imported to edit the brightness temperatures (TBs).
	The TBs have been simulated for each single frequency along the G band water vapour absorption
	line, but the MiRAC-P (to which the retrieval is applied to) on the Polarstern measures with 
	averaged double side bands along the G band.
	Therefore the TBs must also be averaged in the fwd. sim., so that during the retrieval development
	coefficients will be trained that work for double side band averaged measurements.	
"""

# Data paths:
path_fwd_sim = {'ny_alesund': "/net/blanc/awalbroe/Data/mir_fwd_sim/",
				'era_interim': "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/combined/"}
path_plots = "/net/blanc/awalbroe/Plots/"
path_radiosondes = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"

which_training = 'ny_alesund'		# choose the training data for mwr_pro MiRAC-P retrieval for plotting the histogram
									# options: 'era_interim', 'ny_alesund'
create_IWV_histogram = True		# create distribution overview of IWV plot
plot_density = True
include_RS = True			# include MOSAiC radiosondes IWV in the histogram, recommended only with
							# plot_density = True and create_IWV_histogram = True

# select data path:
path_fwd_sim = path_fwd_sim[which_training]
ret_files = sorted(glob.glob(path_fwd_sim + "rt*.nc"))


# open each file as xarray dataset:
ndate = 0
if create_IWV_histogram:
	if which_training == 'era_interim':
		IWV_all = np.full((24836,), np.nan)	# will contain all training / test data IWV

		for ret_file in ret_files:
			RET_DS = xr.open_dataset(ret_file)

			ndate_tmp = len(RET_DS.n_date.values)
			IWV_all[ndate:(ndate+ndate_tmp)] = RET_DS.integrated_water_vapor.values
			ndate = ndate + ndate_tmp
			# print(ndate)
			# print(np.amin(RET_DS.integrated_water_vapor.values))
			

			RET_DS.close()

		if include_RS:
			date_start = "2019-09-20"
			date_end = "2020-10-12"
			sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version='level_2', verbose=1)

	elif which_training == 'ny_alesund':
		IWV_all = np.full((2746,), np.nan)	# will contain all training / test data IWV

		for ret_file in ret_files:
			RET_DS = xr.open_dataset(ret_file)

			ndate_tmp = len(RET_DS.n_date.values)
			IWV_all[ndate:(ndate+ndate_tmp)] = RET_DS.integrated_water_vapor.values
			ndate = ndate + ndate_tmp
			# print(ndate)
			# print(np.amin(RET_DS.integrated_water_vapor.values))
			

			RET_DS.close()

		if include_RS:
			date_start = "2019-09-20"
			date_end = "2020-10-12"
			sonde_dict = import_radiosonde_daterange(path_radiosondes, date_start, date_end, s_version='level_2', verbose=1)

else:
	for ret_file in ret_files:
		RET_DS = xr.open_dataset(ret_file)

		# rename old frequency dimension 'n_frequency':
		RET_DS = RET_DS.rename_dims({'n_frequency': 'n_frequency_orig'})

		# rename original TBs and frequencies:
		RET_DS = RET_DS.rename({'brightness_temperatures': 'brightness_temperatures_orig'})
		RET_DS = RET_DS.rename({'frequency': 'frequency_orig'})


		# Double side band average for G band if G band frequencies are available, which must first be clarified:
		# Determine, which frequencies are around the G band w.v. absorption line:
		g_upper_end = 183.31 + 15
		g_lower_end = 183.31 - 15
		g_freq = np.where((RET_DS.frequency_orig.values > g_lower_end) & (RET_DS.frequency_orig.values < g_upper_end))[0]
		non_g_freq = np.where(~((RET_DS.frequency_orig.values > g_lower_end) & (RET_DS.frequency_orig.values < g_upper_end)))[0]

		TB_orig = RET_DS.brightness_temperatures_orig
		TB_dsba = copy.deepcopy(TB_orig)
		

		if g_freq.size > 0: # G band within frequencies
			g_low = np.where((RET_DS.frequency_orig.values <= 183.31) & (RET_DS.frequency_orig.values >= g_lower_end))[0]
			g_high = np.where((RET_DS.frequency_orig.values >= 183.31) & (RET_DS.frequency_orig.values <= g_upper_end))[0]

			assert len(g_low) == len(g_high)
			idx = g_low[0]
			for jj in range(len(g_high)):
				TB_dsba[:,:,:,jj] = (TB_orig[:,:,:,g_low[-1-jj]] + TB_orig[:,:,:,g_high[jj]])/2

		# remove all TBs that are outside the G double side band and append the non_g_freq lateron:
		# finally, rename the dimension so that it can be put back into the RET_DS dataset without
		# messing up the other variables' dimensions.

		TB_dsba = TB_dsba[:,:,:,:len(g_low)]
		TB_dsba = xr.concat((TB_dsba, TB_orig[:,:,:,non_g_freq]), dim='n_frequency_orig')
		TB_dsba = TB_dsba.rename({'n_frequency_orig': 'n_frequency'})

		# create new frequency array and rename the dimension:
		freq_dsba = RET_DS.frequency_orig[g_high]
		freq_dsba = xr.concat((freq_dsba, RET_DS.frequency_orig[non_g_freq]), dim='n_frequency_orig')
		freq_dsba = freq_dsba.rename({'n_frequency_orig': 'n_frequency'})
		
		# save the edited TBs and frequencies to the dataset:
		RET_DS['brightness_temperatures'] = TB_dsba
		RET_DS['frequency'] = freq_dsba
		
		# write to file:
		outfile = os.path.basename(ret_file)

		# Encoding:
		encoding = {kk: {"_FillValue": None} for kk in RET_DS.variables}

		RET_DS.to_netcdf(path_fwd_sim + "new_rt_nya/" + outfile, mode='w', format='NETCDF3_CLASSIC', encoding=encoding)

		RET_DS.close()
		continue



# plot histogram:
fs = 20
# colours:
c_fwd = (1,0.435,0)
c_iwv = (0.067,0.29,0.769)

if create_IWV_histogram:
	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(12,9)

	x_lim = [0, 30]
	lowest_valid_IWV = np.amin(IWV_all[IWV_all >=0])
	highes_valid_IWV = np.amax(IWV_all[IWV_all >=0])
		
	if plot_density:
		if include_RS:
			ax1.hist(sonde_dict['iwv'], bins=np.arange(x_lim[0], x_lim[1]), density=True, color=c_fwd,
						alpha=0.5, label='MOSAiC Radiosondes')
		ax1.hist(IWV_all, bins=np.arange(x_lim[0],x_lim[1]), density=True, color=c_iwv,
					alpha=0.5, label='Training, %s'%(which_training.replace("_", "-")))
	else:
		ax1.hist(IWV_all, bins=np.arange(x_lim[0],x_lim[1]), density=False, color=c_iwv,
				alpha=0.5)

	ax1.text(0.98, 0.96, "Min Training IWV = %.2f\n Max Training IWV = %.2f\n Min MOSAiC IWV = %.2f\nMax MOSAiC IWV = %.2f"%(
				lowest_valid_IWV, highes_valid_IWV, np.amin(sonde_dict['iwv'][sonde_dict['iwv']>0]), np.amax(sonde_dict['iwv'])),
				ha='right', va='top', transform=ax1.transAxes, fontsize=fs-4,
				bbox=dict(boxstyle='round', ec=(0,0,0), fc=(1,1,1), alpha=0.5))

	ax1.xaxis.set_minor_locator(AutoMinorLocator())
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1.set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0)
	if plot_density:
		ax1.set_ylabel("Frequency occurrence", fontsize = fs)
	else:
		ax1.set_ylabel("Count", fontsize = fs)

	ax1.tick_params(axis='both', labelsize=fs-2)

	if include_RS:
		ax1.set_title("Integrated Water Vapour (IWV)", fontsize=fs, pad=0)
		hann, labb = ax1.get_legend_handles_labels()
		ax1.legend(handles=hann, labels=labb, loc='center right', fontsize=fs-4)
	else:
		ax1.set_title("Integrated Water Vapour (IWV) training data - %s"%RET_DS.location, fontsize=fs, pad=0)

	ax1_pos = ax1.get_position().bounds
	ax1.set_position([ax1_pos[0], ax1_pos[1] + 0.05*ax1_pos[3], ax1_pos[2], ax1_pos[3]*0.95])


	# pdb.set_trace()
	if plot_density:
		if include_RS:
			fig1.savefig(path_plots + "IWV_hist_RS_MOSAiC_training_%s_density.png"%(which_training.replace("_", "")), dpi=400)
		else:
			fig1.savefig(path_plots + "IWV_hist_training_%s_density.png"%(which_training.replace("_", "")), dpi=400)
	else:
		fig1.savefig(path_plots + "IWV_hist_training_%s.png"%(which_training.replace("_", "")), dpi=400)

	plt.show()