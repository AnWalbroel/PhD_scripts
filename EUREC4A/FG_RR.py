import numpy as np
import datetime as dt
import glob
import pdb
import os
import copy
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl

import sys
sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')
from data_tools import datetime_to_epochtime
from met_tools import ZR_rain_rate, compute_LWC_from_Z
from my_classes_eurec4a import *


"""
	This script ...
	... imports HALO's HAMP (MWR and radar) and WALES data from the EUREC4A campaign,
	... eventually checks temporal sampling of the data and unifies them if necessary,
	... identifies clouds with existing cloud masks (but there must be at least 60 sec
		of clear sky between clouds) via WALES and radar; if less than 60 sec is clear
		sky it is considered as the same cloudy period,
	... computes TB in clear sky scenes before each cloudy period,
	... for each cloudy period TB_cloudy - TB_clear_sky is computed and radar refl. or
		rain rate noted,
	... plot (TB_cloudy - TB_clear_sky) x (rain rate / radar refl.), and eventually
		save the identified idx, time stamps, TBs, rain rates, radar refls.
"""




t_spacing = 60		# required clear sky time in seconds that separates two cloudy periods.
					# also affects the range over which TB_clear_sky is averaged.
entire_period = False		# if True: all indices from beginning to end of cloudy period are
							# considered cloudy (even if radar refl. pretty low) and therefore
							# will appear in the plot; if False: only true cloudy indices are
							# used (recommended)
TB_diff_plus = False		# if True: TB_diff (cloudy - clear) will be enhanced by taking an
							# emission signal (with pos. TB difference) difference into account 
							# as well (but in many cases the emission signal might then dominate)
cases_manually = False		# if False: daily plots are created; if True: multi-day cases
							# are appended and plotted together
case_type = 'liq'			# select liquid or ice G band signal cases (only relevant if cases_manually == True)
							# options: 'liq', 'ice'
compute_LWC = True			# if True, Z-LWC relations are used to compute LWC
save_figures = True		# if True, figures will be saved


dates = {'all': ['20200119',
				# '20200122', # no radar
				'20200124',
				'20200126',
				'20200128',
				'20200130',
				'20200131',
				'20200202',
				'20200205',
				'20200207',
				'20200209',
				'20200211',
				'20200213',
				#'20200215',
				'20200218'],
		'ice': 	[
				'20200119', 
				'20200209'
				],
		'liq':	[
				#####################'20200124',
				'20200202'
				]}

if cases_manually:
	dates = dates[case_type]	
else:
	dates = dates['all']		# default selection


unified_version = 'v0.9'
# WALES_version = 'V1'
HAMP_CM_version = 'v0.9.1'

path_cloudmask_mwr_radar = "/net/blanc/awalbroe/Data/EUREC4A/HAMP_cloud_mask/"
path_unified = "/data/obs/campaigns/eurec4a/HALO/unified/v0.9/"
path_plots = "/net/blanc/awalbroe/Plots/EUREC4A/FG_RR_output/"

# check if plot folder exists. If it doesn't, create it.
path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
	os.makedirs(path_plots_dir)

for hh, which_date in enumerate(['20200202']):
	print(which_date)

	path_WALES = f"/data/obs/campaigns/eurec4a/HALO/{which_date}/WALES-LIDAR/"

	# Import HALO HAMP, lidar, and cloud mask data:
	HAMP_MWR = MWR(path_unified, version=unified_version, which_date=which_date)
	HAMP_radar = radar(path_unified, version=unified_version, which_date=which_date)
	CM_MWR = cloudmask(path_cloudmask_mwr_radar, instrument='mwr', version=HAMP_CM_version, which_date=which_date,
						return_DS=True)
	CM_radar = cloudmask(path_cloudmask_mwr_radar, instrument='radar', version=HAMP_CM_version, which_date=which_date,
						return_DS=True)


	if compute_LWC:
		# filter cloud types:
		time_found = dict()

		# heavy drizzle (HD):
		# >10 indices with dBZ > 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask_HD = HAMP_radar.dBZ > 15
		time_found['HD'] = np.where(np.count_nonzero(search_mask_HD[:,height_idx], axis=1) > 10)[0]
		HAMP_radar.LWC_HD = compute_LWC_from_Z(HAMP_radar.Z[time_found['HD'],:], 'heavy_drizzle')

		# light drizzle (LD):
		# >10 indices with dBZ >= -15 but <= 15 at altitudes below 1000 m
		height_idx = np.where(HAMP_radar.height < 1000)[0]
		search_mask_LD = ((HAMP_radar.dBZ >= -15) & (HAMP_radar.dBZ <= 15))
		time_found['LD'] = np.where(np.count_nonzero(search_mask_LD[:, height_idx], axis=1) > 10)[0]
		HAMP_radar.LWC_LD = compute_LWC_from_Z(HAMP_radar.Z[time_found['LD'],:], 'light_drizzle')

		# no drizzle (ND):
		# no drizzle threshold of -15 was chosen because of Fig. 1 and 6 in Khain et al. 2008,
		# Fig. 9 of Fox and Illingworth 1997, an because of Liu et al. 2008.
		# And dBZ must be < -38 in the lowest 300 m
		height_idx = np.where(HAMP_radar.height < 300)[0]
		search_mask1 = HAMP_radar.dBZ >= -15
		search_mask2 = HAMP_radar.dBZ[:,height_idx] >= -38
		time_found['ND'] = np.where((np.count_nonzero(search_mask1, axis=1) == 0) &
								(np.count_nonzero(search_mask2, axis=1) == 0))[0]

		HAMP_radar.LWC_ND = compute_LWC_from_Z(HAMP_radar.Z[time_found['ND'],:], 'no_drizzle', 
												algorithm='Fox_and_Illingworth_1997_ii')

		# Integrate LWC and bring it back to the normal unified time axis:
		# Gives LWP or RWP in g m^-2
		n_time_ND = len(time_found['ND'])
		n_time_LD = len(time_found['LD'])
		n_time_HD = len(time_found['HD'])
		n_height_radar = len(HAMP_radar.height)
		LWP_ND = np.zeros((n_time_ND,))
		LWP_LD = np.zeros((n_time_LD,))
		LWP_HD = np.zeros((n_time_HD,))

		for ttt in range(n_time_ND):
			for kkk in range(n_height_radar-1):
				if ~np.isnan(HAMP_radar.LWC_ND[ttt,kkk]):
					LWP_ND[ttt] += HAMP_radar.LWC_ND[ttt,kkk] * (HAMP_radar.height[kkk+1] - HAMP_radar.height[kkk])

		for ttt in range(n_time_LD):
			for kkk in range(n_height_radar-1):
				if ~np.isnan(HAMP_radar.LWC_LD[ttt,kkk]):
					LWP_LD[ttt] += HAMP_radar.LWC_LD[ttt,kkk] * (HAMP_radar.height[kkk+1] - HAMP_radar.height[kkk])

		for ttt in range(n_time_HD):
			for kkk in range(n_height_radar-1):
				if ~np.isnan(HAMP_radar.LWC_HD[ttt,kkk]):
					LWP_HD[ttt] += HAMP_radar.LWC_HD[ttt,kkk] * (HAMP_radar.height[kkk+1] - HAMP_radar.height[kkk])

		# Put it back to unified time grid (don't interpolate ...just put it on the grid):
		n_time = len(HAMP_radar.time)
		HAMP_radar.LWP_ND = np.zeros((n_time,))
		HAMP_radar.LWP_LD = np.zeros((n_time,))
		HAMP_radar.LWP_HD = np.zeros((n_time,))
		HAMP_radar.LWP_ND[time_found['ND']] = LWP_ND
		HAMP_radar.LWP_LD[time_found['LD']] = LWP_LD
		HAMP_radar.LWP_HD[time_found['HD']] = LWP_HD

		# a variable saving all cases; when time indices are overlapping, the stronger precip type will be taken
		HAMP_radar.LWP_combined = copy.deepcopy(HAMP_radar.LWP_ND)
		HAMP_radar.LWP_combined[time_found['LD']] = LWP_LD
		HAMP_radar.LWP_combined[time_found['HD']] = LWP_HD

	else:
		HAMP_radar.LWP_ND = np.zeros((n_time,))
		HAMP_radar.LWP_LD = np.zeros((n_time,))
		HAMP_radar.LWP_HD = np.zeros((n_time,))
		HAMP_radar.LWP_combined = np.zeros((n_time,))


	# Make sure HAMP_mwr, HAMP_radar, CM_MWR, CM_radar are on the same time axis:
	# Because the unified HAMP products and HAMP cloud masks originate from the same unified data set time axis,
	# it is possible to resample the CM products to the filtered time axis of HAMP_MWR and HAMP_radar as it is
	# merely a subset. No interpolation errors will occur.
	CM_MWR.DS = CM_MWR.DS.sel(time=HAMP_MWR.time)
	CM_MWR.time = CM_MWR.DS.time.values
	CM_MWR.cloud_mask = CM_MWR.DS.cloud_mask.values
	CM_radar.DS = CM_radar.DS.sel(time=HAMP_MWR.time)
	CM_radar.time = CM_radar.DS.time.values
	CM_radar.cloud_mask = CM_radar.DS.cloud_mask.values
	assert (np.all(CM_MWR.time == CM_radar.time) and np.all(CM_MWR.time == HAMP_radar.time)
			and np.all(HAMP_radar.time == HAMP_MWR.time))


	if cases_manually:
		# Truncate time for G band ice signal cases: define time limits of the cases
		if (case_type == 'ice') and (which_date in ['20200119', '20200209']):
			time_lims = [
						[datetime_to_epochtime(dt.datetime(2020,1,19,12,30)), datetime_to_epochtime(dt.datetime(2020,1,19,14,30))],
						[datetime_to_epochtime(dt.datetime(2020,2,9,14,15)), datetime_to_epochtime(dt.datetime(2020,2,9,14,45))],
						[datetime_to_epochtime(dt.datetime(2020,2,9,15,15)), datetime_to_epochtime(dt.datetime(2020,2,9,15,45))],
						[datetime_to_epochtime(dt.datetime(2020,2,9,16,15)), datetime_to_epochtime(dt.datetime(2020,2,9,16,45))]
						]

		# G band liq. signal cases:
		elif (case_type == 'liq') and (which_date in ['20200124', '20200202']):
			time_lims = [
						[datetime_to_epochtime(dt.datetime(2020,1,24,14,45)), datetime_to_epochtime(dt.datetime(2020,1,24,15,15))],
						[datetime_to_epochtime(dt.datetime(2020,1,24,15,45)), datetime_to_epochtime(dt.datetime(2020,1,24,16,15))],
						[datetime_to_epochtime(dt.datetime(2020,2,2,16,30)), datetime_to_epochtime(dt.datetime(2020,2,2,17,00))],
						[datetime_to_epochtime(dt.datetime(2020,2,2,17,30)), datetime_to_epochtime(dt.datetime(2020,2,2,18,00))],
						[datetime_to_epochtime(dt.datetime(2020,2,2,18,30)), datetime_to_epochtime(dt.datetime(2020,2,2,19,00))]
						]

		time_idx = list()
		for tl in time_lims:
			time_idx_temp = np.where((HAMP_MWR.time >= tl[0]) & (HAMP_MWR.time <= tl[1]))[0]
			for kkk in time_idx_temp:
				time_idx.append(kkk)

		# truncate
		HAMP_MWR.time = HAMP_MWR.time[time_idx]
		HAMP_MWR.TB = HAMP_MWR.TB[time_idx,:]

		HAMP_radar.time = HAMP_radar.time[time_idx]
		HAMP_radar.dBZ = HAMP_radar.dBZ[time_idx,:]
		HAMP_radar.Z = HAMP_radar.Z[time_idx,:]

		CM_MWR.time = CM_MWR.time[time_idx]
		CM_MWR.cloud_mask = CM_MWR.cloud_mask[time_idx]

		CM_radar.time = CM_radar.time[time_idx]
		CM_radar.cloud_mask = CM_radar.cloud_mask[time_idx]
	


	# Identify cloudy periods (with at least t_spacing seconds of clear sky between two cloudy periods):
	# Use MWR and radar cloud mask; try with 0 cloudy idx allowed in those t_spacing sec; if too little periods
	# are identified, increase the permitted cloudy idx to 1 or 2. idx_cloudy is a 2d list: 
	# (n_cloudy_periods, n_idx_for_nth_cloudy_period)
	CM_MWR.find_cloudy_periods(t_spacing)
	CM_radar.find_cloudy_periods(t_spacing)


	# Compute TB clear sky for each cloudy period (average over one minute before and after the cloudy
	# period): for the nth cloudy period, this would be something like:
	# 0.5*(np.nanmean(TB[idx_cloudy[n,0] - t_spacing : idx_cloudy[n,0],:], axis=0) + np.nanmean(
	# 		TB[idx_cloudy[n,-1]:idx_cloudy[n,-1]+t_spacing,:], axis=0))
	# Eventually, if the averaging should really be t_spacing seconds, the correct time indices have to be
	# found: np.where(CM_radar.time == CM_radar.time[CM_radar.cloudy_idx[n][0]] - t_spacing)[0] and
	# np.where(CM_radar.time == CM_radar.time[CM_radar.cloudy_idx[n][-1]] + t_spacing)[0]

	# Loop through all cloudy periods and compute the clear sky TBs, and also the difference
	# of TBs inside the cloud. And note the reflectivities.
	n_cloudy_periods = len(CM_radar.cloudy_idx)
	n_freq = len(HAMP_MWR.freq)
	n_height = len(HAMP_radar.height)
	if entire_period:
		n_max_cloudy = 0		# will be the maximum number of cloudy pixels
		for k in CM_radar.cloudy_idx: n_max_cloudy += (k[-1]+1 - k[0])
	else:
		n_max_cloudy = 0		# will be the number of cloudy pixels
		for k in CM_radar.cloudy_idx: n_max_cloudy += len(k)

	HAMP_MWR.TB_clear_sky = np.zeros((n_cloudy_periods, n_freq))
	HAMP_MWR.TB_diff = np.full((n_max_cloudy, n_freq), np.nan)
	HAMP_radar.dBZ_incloud = np.full((n_max_cloudy, n_height), np.nan)
	HAMP_radar.Z_incloud = np.full((n_max_cloudy, n_height), np.nan)
	HAMP_radar.LWP_ND_incloud = np.full((n_max_cloudy,), 0.0)
	HAMP_radar.LWP_LD_incloud = np.full((n_max_cloudy,), 0.0)
	HAMP_radar.LWP_HD_incloud = np.full((n_max_cloudy,), 0.0)
	HAMP_radar.LWP_combined_incloud = np.full((n_max_cloudy,), 0.0)
	faily_periods = list()
	nn = 0

	# which freq. index is closest to the freq. for the 'positive' emission signal?
	frq_idx_e = np.argmin(np.abs(HAMP_MWR.freq - 52.8))

	if entire_period:
		for k, cloudy_idx in enumerate(CM_radar.cloudy_idx):

			n_cl_pix = cloudy_idx[-1]+1 - cloudy_idx[0]

			# time indices t_spacing sec before begin and after end of cloudy period:
			time_idx_before = np.where(CM_radar.time == CM_radar.time[cloudy_idx[0]] - t_spacing)[0]
			time_idx_after = np.where(CM_radar.time == CM_radar.time[cloudy_idx[-1]] + t_spacing)[0]

			if len(time_idx_before) > 0 and len(time_idx_after) > 0:
				time_idx_before = time_idx_before[0]
				time_idx_after = time_idx_after[0]

				# TBs have to be nonnan: count nan values before and after cloudy period:
				# It's preferred to have both before and after for the computation of TB_clear_sky,
				# but I think it's also fine to just consider the one minute average of either before
				# or after.
				n_nanTBs_before = np.count_nonzero(np.isnan(HAMP_MWR.TB[time_idx_before:cloudy_idx[0],0]))
				n_nanTBs_after = np.count_nonzero(np.isnan(HAMP_MWR.TB[cloudy_idx[-1]+1:time_idx_after+1,0]))
					# --> + 1 needed because of python indexing to make sure that the last cloudy pixel is 
					# excluded and still keep t_spacing seconds respected

				if n_nanTBs_before + n_nanTBs_after <= t_spacing:
					n_nonnans = 2*t_spacing - n_nanTBs_before - n_nanTBs_after
					weights_before = (t_spacing - n_nanTBs_before)/n_nonnans
					weights_after = (t_spacing - n_nanTBs_after)/n_nonnans
					HAMP_MWR.TB_clear_sky[k,:] = (weights_before*np.nanmean(HAMP_MWR.TB[time_idx_before:cloudy_idx[0],:], axis=0) + 
												weights_after*np.nanmean(HAMP_MWR.TB[cloudy_idx[-1]+1:time_idx_after+1,:], axis=0))

					# compute TB difference and note radar reflectivity in cloud (and LWP or RWP):
					HAMP_MWR.TB_diff[nn:nn+n_cl_pix,:] = HAMP_MWR.TB[cloudy_idx[0]:cloudy_idx[-1]+1,:] - HAMP_MWR.TB_clear_sky[k,:]
					HAMP_radar.dBZ_incloud[nn:nn+n_cl_pix,:] = HAMP_radar.dBZ[cloudy_idx[0]:cloudy_idx[-1]+1,:]
					HAMP_radar.Z_incloud[nn:nn+n_cl_pix,:] = HAMP_radar.Z[cloudy_idx[0]:cloudy_idx[-1]+1,:]
					HAMP_radar.LWP_ND_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_ND[cloudy_idx[0]:cloudy_idx[-1]+1]
					HAMP_radar.LWP_LD_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_LD[cloudy_idx[0]:cloudy_idx[-1]+1]
					HAMP_radar.LWP_HD_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_HD[cloudy_idx[0]:cloudy_idx[-1]+1]
					HAMP_radar.LWP_combined_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_combined[cloudy_idx[0]:cloudy_idx[-1]+1]
					
				# else: no non nan values found --> TB_clear_sky cannot be computed
					# faily_periods.append(k)

			nn += n_cl_pix

	else:
		for k, cloudy_idx in enumerate(CM_radar.cloudy_idx):

			n_cl_pix = len(cloudy_idx)

			# time indices t_spacing sec before begin and after end of cloudy period:
			time_idx_before = np.where(CM_radar.time == CM_radar.time[cloudy_idx[0]] - t_spacing)[0]
			time_idx_after = np.where(CM_radar.time == CM_radar.time[cloudy_idx[-1]] + t_spacing)[0]

			if len(time_idx_before) > 0 and len(time_idx_after) > 0:
				time_idx_before = time_idx_before[0]
				time_idx_after = time_idx_after[0]

				# TBs have to be nonnan: count nan values before and after cloudy period:
				# It's preferred to have both before and after for the computation of TB_clear_sky,
				# but I think it's also fine to just consider the one minute average of either before
				# or after.
				n_nanTBs_before = np.count_nonzero(np.isnan(HAMP_MWR.TB[time_idx_before:cloudy_idx[0],0]))
				n_nanTBs_after = np.count_nonzero(np.isnan(HAMP_MWR.TB[cloudy_idx[-1]+1:time_idx_after+1,0]))
					# --> + 1 needed because of python indexing to make sure that the last cloudy pixel is 
					# excluded and still keep t_spacing seconds respected

				if n_nanTBs_before + n_nanTBs_after <= t_spacing:
					n_nonnans = 2*t_spacing - n_nanTBs_before - n_nanTBs_after
					weights_before = (t_spacing - n_nanTBs_before)/n_nonnans
					weights_after = (t_spacing - n_nanTBs_after)/n_nonnans
					HAMP_MWR.TB_clear_sky[k,:] = (weights_before*np.nanmean(HAMP_MWR.TB[time_idx_before:cloudy_idx[0],:], axis=0) + 
												weights_after*np.nanmean(HAMP_MWR.TB[cloudy_idx[-1]+1:time_idx_after+1,:], axis=0))

					# compute TB difference and note radar reflectivity in cloud:
					HAMP_MWR.TB_diff[nn:nn+n_cl_pix,:] = HAMP_MWR.TB[cloudy_idx,:] - HAMP_MWR.TB_clear_sky[k,:]
					HAMP_radar.dBZ_incloud[nn:nn+n_cl_pix,:] = HAMP_radar.dBZ[cloudy_idx,:]
					HAMP_radar.Z_incloud[nn:nn+n_cl_pix,:] = HAMP_radar.Z[cloudy_idx,:]
					HAMP_radar.LWP_ND_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_ND[cloudy_idx]
					HAMP_radar.LWP_LD_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_LD[cloudy_idx]
					HAMP_radar.LWP_HD_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_HD[cloudy_idx]
					HAMP_radar.LWP_combined_incloud[nn:nn+n_cl_pix] = HAMP_radar.LWP_combined[cloudy_idx]

				# else: no non nan values found --> TB_clear_sky cannot be computed
					# faily_periods.append(k)

			nn += n_cl_pix

	HAMP_MWR.TB_diff_e = HAMP_MWR.TB_diff[:, frq_idx_e]

	# reflectivity and TBs have to be reduced to 1D arrays to compute
	# typical scatterplot statistics and to plot them easily:
	# For now, choose max reflectivity of a column and G band TBs:
	idx_G = np.where((HAMP_MWR.freq > 170) & (HAMP_MWR.freq < 200))[0]

	# instead of max. radar refl. factor (i), try the lowest (iii) or average over
	# the lowest 200 m (ii): Three options:
	# (i):
	HAMP_radar.dBZ_plot = np.nanmax(HAMP_radar.dBZ_incloud, axis=1)
	HAMP_radar.Z_plot = np.nanmax(HAMP_radar.Z_incloud, axis=1)

	# (ii):
	# HAMP_radar.dBZ_plot = np.nanmean(HAMP_radar.dBZ_incloud[:,:7], axis=1)
	# HAMP_radar.Z_plot = np.nanmean(HAMP_radar.Z_incloud[:,:7], axis=1)

	# # (iii):
	# HAMP_radar.dBZ_plot = np.full((n_max_cloudy,), np.nan)
	# HAMP_radar.Z_plot = np.full((n_max_cloudy,), np.nan)
	# for hihi in range(n_max_cloudy):
		# lowest_Z_cloud_idx = np.where(~np.isnan(HAMP_radar.Z_incloud[hihi,:]))[0]
		# if len(lowest_Z_cloud_idx) > 0:
			# if HAMP_radar.height[lowest_Z_cloud_idx[0]] < 500:
				# HAMP_radar.dBZ_plot[hihi] = HAMP_radar.dBZ_incloud[hihi,lowest_Z_cloud_idx[0]]
				# HAMP_radar.Z_plot[hihi] = HAMP_radar.Z_incloud[hihi,lowest_Z_cloud_idx[0]]

	if cases_manually:
		# save data for icey or liq. G band signal cases:
		if hh == 0:
			dBZ_plot = HAMP_radar.dBZ_plot
			Z_plot = HAMP_radar.Z_plot
			if not TB_diff_plus:
				TB_diff = HAMP_MWR.TB_diff
		else:
			dBZ_plot = np.concatenate((dBZ_plot, HAMP_radar.dBZ_plot), axis=0)
			Z_plot = np.concatenate((Z_plot, HAMP_radar.Z_plot), axis=0)
			if not TB_diff_plus:
				TB_diff = np.concatenate((TB_diff, HAMP_MWR.TB_diff), axis=0)
		

# Compute rain rate (with Marshall Palmer drop size distribution):
if cases_manually:
	HAMP_radar.RR_plot = ZR_rain_rate(Z_plot)
	HAMP_radar.dBZ_plot = dBZ_plot			# overwrite the daily dBZ_plot
	HAMP_MWR.TB_diff = TB_diff				# same here
else:
	HAMP_radar.RR_plot = ZR_rain_rate(HAMP_radar.Z_plot)

# Plot results:
fs = 19

fig1, ax1 = plt.subplots(2,3)
fig1.set_size_inches(16,10)
ax1 = ax1.flatten()


for k, G_idx in enumerate(idx_G):

	# Compute statistics:
	# x_stuff = HAMP_radar.RR_plot				# RAIN RATE
	x_stuff = HAMP_radar.LWP_ND_incloud	# Integrated LWC
	# x_stuff = HAMP_radar.dBZ_plot				# default
	if TB_diff_plus:
		y_stuff = HAMP_MWR.TB_diff_e - HAMP_MWR.TB_diff[:,G_idx]
	else:
		y_stuff = HAMP_MWR.TB_diff[:,G_idx]		# default

	# filter nans:
	where_nonnan = np.argwhere(~np.isnan(y_stuff) & ~np.isnan(x_stuff)).flatten()
	x_stuff = x_stuff[where_nonnan]
	y_stuff = y_stuff[where_nonnan]
	stat_dict = compute_retrieval_statistics(x_stuff, y_stuff)

	# Concatenate (stack) x and y data to create a coloured density scatter plot
	if len(x_stuff) > 0 and len(y_stuff) > 0:
		xy = np.vstack([x_stuff, y_stuff])
		z = gaussian_kde(xy)(xy)

		# Sort the points by density, so that the densest points are plotted last
		idx = z.argsort()
		x_stuff, y_stuff, z = x_stuff[idx], y_stuff[idx], z[idx]

		# ax1[k].plot(x_stuff, y_stuff, linestyle='none', marker='.', color=(0,0,0), markersize=2.0)
		ax1[k].scatter(x_stuff, y_stuff, c=z, s=6)#, linestyle='none', marker='.', color=(0,0,0), markersize=2.0)

		# plot a line for orientation which would represent a perfect fit:
		# ax1[k].plot(axlim[k], axlim[k], color=(0,0,0), linewidth=0.75, alpha=0.5, label="Theoretical perfect fit")


		# set axis limits and labels:
		# ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		# ax1[k].set_xlim(left=-50, right=60)
		# # # # # # # # # # # # # # # # # # # # ax1[k].set_xlim(left=-0.5, right=5)
		# ax1[k].set_xlim(left=-2, right=160)
		ax1[k].set_xlim(left=0, right=300)

		ax1[k].tick_params(axis='both', labelsize=fs-6)
		ax1[k].set_aspect(1.0/ax1[k].get_data_ratio(), 'box')

		ax1[k].minorticks_on()
		ax1[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		# ax1[k].set_ylim(bottom=axlim[k][0], top=axlim[k][1])
		# ax1[k].set_xlim(left=-50, right=60)
		# # # # # # # # # # # # # # # # # # # # # ax1[k].set_xlim(left=-0.5, right=5)
		# ax1[k].set_xlim(left=-2, right=160)
		ax1[k].set_xlim(left=0, right=300)

		if k%3 == 0: ax1[k].set_ylabel("TB$_{\mathrm{cloudy}}$ - TB$_{\mathrm{clear}}$ (K)", fontsize=fs-2)
		# if k >= 3: ax1[k].set_xlabel("Radar refl. (dBZ)", fontsize=fs-2)
		if k >= 3: ax1[k].set_xlabel("Integrated LWC ($\mathrm{g}\,\mathrm{m^{-2}}$)", fontsize=fs-2)

		ax1[k].set_title("%.2f GHz"%(HAMP_MWR.freq[G_idx]), fontsize=fs-4, fontweight='bold')

# fig1.suptitle("Relationship between radar reflectivity and TB difference", fontsize=fs)
# fig1.suptitle("Relationship between rain rate and TB diff - %s cases"%(case_type), fontsize=fs)
fig1.suptitle("Relationship between LWP (RWP) and TB diff - %s cases"%(case_type), fontsize=fs)
# fig1.suptitle("Relationship between radar reflectivity and TB diff", fontsize=fs)


if save_figures:
	if TB_diff_plus:
		# name_base = "EUREC4A_HALO_HAMP_TBdiffplus%02i_maxrefl_%s"%(frq_idx_e, which_date)
		# name_base = "EUREC4A_HALO_HAMP_TBdiffplus%02i_rainrate_%s"%(frq_idx_e, which_date)
		name_base = "EUREC4A_HALO_HAMP_TBdiffplus%02i_LWPRWP_%s"%(frq_idx_e, which_date)
	else:
		# name_base = "EUREC4A_HALO_HAMP_TBdiff_maxrefl_%s"%(which_date)
		# name_base = "EUREC4A_HALO_HAMP_TBdiff_rainrate_%s"%(which_date)
		name_base = "EUREC4A_HALO_HAMP_TBdiff_LWPRWP_%s"%(which_date)

	# if cases_manually: name_base = "EUREC4A_HALO_HAMP_TBdiff_rainrate_%scases"%(case_type)
	if cases_manually: name_base = "EUREC4A_HALO_HAMP_TBdiff_LWPRWP_%scases"%(case_type)
	fig1.savefig(path_plots + name_base + ".png", dpi=400)

else:
	plt.show()



print("Done....")
