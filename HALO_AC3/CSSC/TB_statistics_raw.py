from __future__ import print_function, division
import numpy as np
import os
import datetime
from general_importer import *
import glob
import pdb
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import xarray as xr



def run_TB_statistics_raw(
	path_mwr,
	path_pam_ds,
	out_path,
	plot_path,
	scatterplot_name,
	bias_ev_plotname,
	output_filename,
	obs_height='BAHAMAS',
	path_BAH_data=None,
	path_RADAR_data=None,
):
	"""
	Parameters
	----------
	path_mwr : str
		Path of HAMP microwave radiometer date (concatenated).
	path_pam_ds : str
		Path of simulated TBs from dropsonde measurements.
	out_path : str
		Path where the clear sky sonde correction NETCDF file is saved to.
	plot_path : str
		Path where the generated plots are saved to.
	scatterplot_name : str
		File name of the scatterplot.
	bias_ev_plotname : str
		File name of the plot showing the evolution of the biases.
	output_filename : str
		File name of the clear sky sonde correction NETCDF file. Saved into out_path.
	obs_height : number or str
		If a number is given use this as assumed aircraft altitude.
		If 'BAHAMAS' is given get altitude from BAHAMAS files.
			BAHAMAS files have to be in unified netCDF format and `path_BAH_data' has to be set.
	path_BAH_data : str, optional
		Path of BAHAMAS data in unified netCDF files. Required if obs_height == 'BAHAMAS'.
	path_RADAR_data : str, optional
		Path of RADAR data in unified netCDF files. Is set use radar as additional cloud detector.

	this program shall find the mean + stddev TB from HAMP measurements from - to + 10 seconds of each dropsonde launch.
	Will be named: 'tb_mean' and 'tb_std'. It will also contain a variable 'tb_N' noting the number of HAMP mwr measurements
	in this 20 sec. time period. The measured TBs will be compared with TBs simulated from dropsondes using PAMTRA; the
	simulated TBs will be called 'tb_sonde'. The dropsonde release time will serve as timestamp and is saved in
	'time'. Other supplemental information: 'frequency', string date(time) with description: "Date of corresponding Flight" and
	in units "YYYYMMDD", sonde index 'sondenumber'.
	"""

	# Check if the sonde comparison output and plot path exist:
	out_path_dir = os.path.dirname(out_path)
	if not os.path.exists(out_path_dir):
		os.makedirs(out_path_dir)
	plot_path_dir = os.path.dirname(plot_path)
	if not os.path.exists(plot_path_dir):
		os.makedirs(plot_path_dir)

	# Import HAMP mwr data, simulated dropsonde files, ...:

	MWR_ncfiles = sorted(glob.glob(path_mwr + "*.nc"))
	PAM_ds_ncfiles = sorted(glob.glob(path_pam_ds + "*.nc"))

	if len(PAM_ds_ncfiles) == 0:
		raise RuntimeError("Could not find any dropsonde files in `%s'*.nc"  % (PAM_ds_ncfiles))

	if isinstance(obs_height, str):
		if obs_height == 'BAHAMAS':
			if not isinstance(path_BAH_data, str):
				raise ValueError("path_BAH_data is required as string argument when obs_height == 'BAHAMAS'")
			BAH_files_NC = sorted(glob.glob(path_BAH_data + "bahamas*.nc"))
			if len(BAH_files_NC) == 0:
				raise RuntimeError("Could not find any BAHAMAS data in `%s'"  % (path_BAH_data + "bahamas*.nc"))
		else:
			raise ValueError("Unknown obs_height `%s'" % obs_height)
	else:
		BAH_files_NC = []
		obs_height_value = np.asarray(obs_height).flatten() # we need a 1D array, which is used for all dropsondes.


	# cycle through all files (days) where a dropsonde (ds) could be pamtra simulated
	file_index = 0		# can also be used to address the correct files if the number of files is the same for all 3 options (DS, MWR, PAM_DS)
	sonde_number_temp = 0
	for pam_ds_file in PAM_ds_ncfiles:

		pam_ds_dict = import_DSpam_nc(pam_ds_file, '', True, True)	# imports all keys and performs double side band averaging

		work_date_temp = datetime.datetime.utcfromtimestamp(pam_ds_dict['datatime'][0]).strftime("%Y%m%d")

		print("Date: " + work_date_temp)

		# auxiliary date which will be used to identify if the current file still belongs to the same day:
		if file_index == 0:
			date_aux = copy.deepcopy(work_date_temp)
			day_idx_temp = 0		# identifies the n-th sonde of the current day
		else:
			day_idx_temp = day_idx_temp + 1

		if work_date_temp != date_aux:		# overwrite date aux and reset the day_idx counter
			date_aux = copy.deepcopy(work_date_temp)
			day_idx_temp = 0


		# Import the correct files (of same day):
		# finding the right file via scanning with for: ...
		mwr_file = [mwrfile for mwrfile in MWR_ncfiles if work_date_temp[2:] in mwrfile][0]


		bah_keys = ['time', 'altitude']		# keys for BAHAMAS data import
		bah_filenames = [bah_file for bah_file in BAH_files_NC if work_date_temp in bah_file]
		if bah_filenames:
			bah_dict = import_BAHAMAS_unified(bah_filenames[0], bah_keys)


		mwr_dict = import_mwr_nc(mwr_file)		# imports all keys of the mwr file (v01 - concatenated mwr files)

		nlaunches = len(pam_ds_dict['datatime'])			# should be == 1 for _raw (bc. each sonde file considered seperately)

		# find mwr time indices when they match the dropsonde launch time: Should yield about 4*nlaunches indices bc. of 4 Hz measurement rate.
		# must be reshaped because python create an unnecessary array dimensions:
		# expand the indices for each launch to 10 seconds before and 10 seconds after the dropsonde launch:
		idx = np.argwhere((mwr_dict['time'] >= pam_ds_dict['datatime'] - 10) & (mwr_dict['time'] < pam_ds_dict['datatime'] + 10))
		idx = np.reshape(idx, (idx.shape[0],))


		# mean and std_dev of mwr TB at these indices:
		tb_mean_temp = np.asarray([np.nanmean(mwr_dict['TBs'][idx,:], axis=0)])
		tb_std_temp = np.asarray([np.nanstd(mwr_dict['TBs'][idx,:], axis=0, dtype=np.float64)])
		frequency = np.asarray([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.4, 50.3, 51.76, 52.8, 53.75, 54.94, 56.66,
			58., 90., 120.15, 121.05, 122.95, 127.25, 183.91, 184.81, 185.81, 186.81, 188.31, 190.81, 195.81])

		# max_tb_stddev_temp will contain the maximum std deviation among all channels for each sonde launch
		max_tb_stddev_temp = np.reshape(np.repeat(np.nanmax(tb_std_temp), len(frequency)), tb_std_temp.shape)


		# number of included measurements per launch:
		tb_N = np.asarray([len(idx)])

		# find the correct aircraft altitude for each launch:
		if isinstance(obs_height, str) and obs_height == 'BAHAMAS':
			bah_dict['time'] = np.rint(bah_dict['time']).astype(float)
			t_idx = np.argwhere(bah_dict['time'] == pam_ds_dict['datatime']).flatten()
			drop_alt = np.floor(np.asarray([np.mean(bah_dict['altitude'][i-10:i+10]) for i in t_idx])/100)*100		# drop altitude for each sonde (floored)
			obs_height_value = drop_alt

		# select correct pamtra outlevel:
		outlevel_idx = np.asarray([np.argwhere(pam_ds_dict['outlevels'][0] == alt) for alt in obs_height_value]).flatten()
		outlevel_idx = 0 # pamtra simulations should be made under the consideration of the correct flight altitude
		if not abs(pam_ds_dict['outlevels'][0, outlevel_idx] - obs_height_value[0]) <= 100:
			raise ValueError("pam_ds_dict['outlevels'] and obs_height_value missmatch:\n%s\n%s"
				% (pam_ds_dict['outlevels'], obs_height_value)
			)

		# positional information:
		sonde_lon_temp = pam_ds_dict['longitude']
		sonde_lat_temp = pam_ds_dict['latitude']


		# pamtra simulated TBs and information about launch time:
		tb_sonde_temp = np.asarray([pam_ds_dict['tb'][0, outlevel_idx, :]])			# shall be a (nlaunches x frequencies) array
		time_temp = pam_ds_dict['datatime']
		date_temp = datetime.datetime.utcfromtimestamp(time_temp[0]).strftime("%Y%m%d")


		# find if std dev threshold is surpassed for any sonde launch:
		stddev_threshold = 1		# in Kelvin
		tb_used_temp = np.ones((1, len(frequency)), dtype=bool)
		# for G band: np.any and for the remaining channels it also suffices for np.any channel to surpass the threshold.
		# For the 6 (or 7) G band entries we consider the std. of all 25 channels. In case it's greater than the
		# threshold the 6 (or 7) G band entries are assigned False entries. The remaining 19 channels (K,V,W,F)
		# are assigned True if the std. in all of these channels is less than the threshold. In case any non G band
		# channel shows a greater std. the 19 channels (K,V,W,F) receive a False.
		cloudy_G = [np.any(tb_std_temp[k,:26] > stddev_threshold) for k in range(nlaunches)]
		# -> if this is true for a sonde launch and cloudy_rest is NOT: only the G band channels receive a FALSE in tb_used

		cloudy_rest = [np.any(tb_std_temp[k,:19] > stddev_threshold) for k in range(nlaunches)]
		# -> for the launch when this is true: tb_used gets a FALSE for frequency entries 0:19

		sondenumber_temp = np.asarray([day_idx_temp])


		# set cloudy launches to false:
		if cloudy_G[0]:
			tb_used_temp[0,19:] = False
		if cloudy_rest[0]:
			tb_used_temp[0,:19] = False

		if path_RADAR_data is not None:
			# set tb_used_temp to false, if radar sees anything within window
			if not is_radar_clear_sky(path_RADAR_data, work_date_temp, pam_ds_dict['datatime']):
				print('>>>> Do not use sond', work_date_temp, 'because the radar sees something')
				tb_used_temp[0, :] = False

		# remove cases with extreme bias: considered cloudy if the bias_threshold is exceeded
		bias_threshold = 30
		for k in range(len(frequency)):	tb_used_temp[np.abs(tb_sonde_temp[:,k] - tb_mean_temp[:,k]) > bias_threshold, k] = False


		# Concatenate the temporary arrays for each day:
		if file_index == 0:
			tb_mean = np.reshape(tb_mean_temp, (1, len(frequency)))
			tb_std = np.reshape(tb_std_temp, (1, len(frequency)))
			tb_sonde = np.reshape(tb_sonde_temp, (1, len(frequency)))
			max_tb_stddev = np.reshape(max_tb_stddev_temp, (1, len(frequency)))
			tb_used = np.reshape(tb_used_temp, (1, len(frequency)))
			work_date = [work_date_temp]
			time = time_temp
			obsheight_save = obs_height_value
			sondenumber = sondenumber_temp
			sonde_lon = sonde_lon_temp
			sonde_lat = sonde_lat_temp

			day_index = [day_idx_temp]	# for the first day just indicates the first day's sonde launches


		else: # then the non temporary arrays have already been initialized and we can use np.concatenate
			tb_mean = np.concatenate((tb_mean, tb_mean_temp), axis=0)
			tb_std = np.concatenate((tb_std, tb_std_temp), axis=0)
			tb_sonde = np.concatenate((tb_sonde, tb_sonde_temp), axis=0)
			max_tb_stddev = np.concatenate((max_tb_stddev, max_tb_stddev_temp), axis=0)
			tb_used = np.concatenate((tb_used, tb_used_temp), axis=0)
			work_date.append(work_date_temp)
			time = np.concatenate((time, time_temp), axis=0)
			obsheight_save = np.concatenate((obsheight_save, obs_height_value), axis=0)
			sondenumber = np.concatenate((sondenumber, sondenumber_temp), axis=0)
			sonde_lon = np.concatenate((sonde_lon, sonde_lon_temp), axis=0)
			sonde_lat = np.concatenate((sonde_lat, sonde_lat_temp), axis=0)

			day_index.append(day_idx_temp)


		file_index = file_index + 1



	day_index = np.asarray(day_index)
	nsondes = len(day_index)		# equals the number of considered sondes


	# Create dataset:

	# convert time to datetime object:
	time = np.asarray([datetime.datetime.utcfromtimestamp(k) for k in time])
	date = [u'%s' % ttt.strftime('%Y%m%d') for ttt in time] # "u'%s' %" is a workaround, that should give a unicode string in python 2 and python 3

	TB_stat_DS = xr.Dataset({
		'tb_mean': 			(['time', 'frequency'], tb_mean,
			{'description': "Average measured brightness temperature within -comparison_window_before/+comparison_window time around sonde release.",
			'units': "K"}),
		'tb_std':  			(['time', 'frequency'], tb_std,
			{'description': "Standard deviation of measured brightness temperature within -comparison_window_before/+comparison_window time around sonde release.",
			'units': "K"}),
		'tb_sonde': 		(['time', 'frequency'], tb_sonde, {'description': "Simulated clear sky brightness temperature", 'units': "K"}),
		'max_tb_std_any':	(['time', 'frequency'], max_tb_stddev,
			{'description': "Maximum standard deviation of measured brightness temperature of any* channel. ''any*'' means that for G band all channels " +
							"are checked whether any HAMP channel exceeds tb_cloud_threshold, while for the other channels it suffices to be considered " +
							"a cloudy case, if any HAMP channel (except the G band) exceeds tb_cloud_threshold.",
			'units': "K"}),
		'tb_used': 			(['time', 'frequency'], tb_used,
			{'criteria': "none of: tb_cloud_threshold, count_reflective_bins_threshold, or max_bias_threshold exceeded or tb_sonde is invalid",
			'description': "Flag that is True, if sonde shall be used in bias calculation", 'type': "bool"}),
		'date': 			(['time'], date, {'description': "Date of corresponding Flight", 'units': "YYYYMMDD"}),
		'obsheight':		(['time'], obsheight_save, {'description': "Altitude of simulated brightness temperature", 'units': "m"}),
		'i_sonde':			(['time'], sondenumber),
		'tb_N': 			(tb_N),
		'longitude':		(['time'], sonde_lon, {'description': "Longitude of sonde launch", 'units': "degree east"}),
		'latitude':			(['time'], sonde_lat, {'description': "Latitude of sonde launch", 'units': "degree north"}),
		},
		coords={'time': (['time'], time, {'description': "timestamp or seconds since 1970-01-01 00:00:00"}),
			'frequency': (['frequency'], frequency, {'description': "channel frequency", 'units': "GHz"})})



	# Set global attributes to dataset:
	TB_stat_DS.attrs['description'] = ("HAMP MWR vs. pamtra simulated dropsondes: " +
		"Includes 20 seconds averaged TBs (and std deviation) from MWR and the respective pamtra simulated dropsonde TBs.")
	TB_stat_DS.attrs['history'] = "Created: " + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
	TB_stat_DS.attrs['author'] = "Andreas Walbroel (Mail: a.walbroel@uni-koeln.de)"
	TB_stat_DS.attrs['comparison_window'] = 10
	TB_stat_DS.attrs['comparison_window_description'] = "Time delta (seconds) after the drop release that is used for comparison of each sonde."
	TB_stat_DS.attrs['comparison_window_before'] = 10
	TB_stat_DS.attrs['comparison_window_before_description'] = "Time delta (seconds) before the drop release that is used for comparison of each sonde."
	TB_stat_DS.attrs['tb_cloud_threshold'] = stddev_threshold
	TB_stat_DS.attrs['tb_cloud_threshold_description'] = ("A TB measurement is regarded cloudy if any* tb_std exceeded this value. " +
		"''any*'' means that for G band all channels " +
		"are checked whether any HAMP channel exceeds tb_cloud_threshold, while for the other channels it suffices to be considered " +
		"a cloudy case, if any HAMP channel (except the G band) exceeds tb_cloud_threshold.")
	TB_stat_DS.attrs['max_bias_threshold'] = bias_threshold
	TB_stat_DS.attrs['max_bias_threshold_description'] = "A TB measurement is regarded cloudy if the absolute difference between tb_sonde and tb_mean is larger than this value."

	# encoding + compressing all variables:
	encoding = {k: {'zlib': True, 'fletcher32': True} for k in TB_stat_DS.keys()}
	encoding = {'time': dict()}

	encoding['time']['dtype'] = 'float64'
	encoding['time']['units'] = 'seconds since 1970-01-01 00:00:00'

	nfreq = len(frequency)
	bias = np.zeros((nfreq,))		# will contain the bias for each frequency
	rmse = np.zeros((nfreq,))		# will contain the rmse for each frequency
	R = np.zeros((nfreq,))
	N = np.zeros((nfreq,))

	# Define cloudfree data:
	tb_sonde_cf = TB_stat_DS.tb_sonde.where(TB_stat_DS.tb_used).values	# sets the cloudy conditions to nan
	tb_mean_cf = TB_stat_DS.tb_mean.where(TB_stat_DS.tb_used).values
	tb_std_cf = TB_stat_DS.tb_std.where(TB_stat_DS.tb_used).values

	for k in range(nfreq):
		# for each frequency: compute bias as: TB_mwr - TB_ds (mean over all sonde launches):
		bias[k] = np.nanmean(tb_mean_cf[:,k] - tb_sonde_cf[:,k])
		rmse[k] = np.sqrt(np.nanmean((np.abs(tb_sonde_cf[:,k] - tb_mean_cf[:,k]))**2))
		tb_sonde_nonnan = tb_sonde_cf[np.argwhere(~np.isnan(tb_mean_cf[:,k])).flatten(),k]	# consider only nonnan cases
		N[k] = len(tb_sonde_nonnan)
		tb_mean_nonnan = tb_mean_cf[np.argwhere(~np.isnan(tb_mean_cf[:,k])).flatten(),k]		# consider only nonnan cases (tb_mean contains most nans) -> reference!
		R[k] = np.corrcoef(tb_sonde_nonnan, tb_mean_nonnan)[0,1]


	# Add bias, rmse, ... to Dataset and save it as netcdf file:
	TB_stat_DS['bias'] = xr.DataArray(bias, dims=('frequency'), coords={'frequency': TB_stat_DS.frequency})
	TB_stat_DS['rmse'] = xr.DataArray(rmse, dims=('frequency'), coords={'frequency': TB_stat_DS.frequency})
	TB_stat_DS['R'] = xr.DataArray(R, 		dims=('frequency'), coords={'frequency': TB_stat_DS.frequency})

	TB_stat_DS.to_netcdf(out_path + output_filename + ".nc", mode='w', format='NETCDF4', encoding=encoding)


	# Closing the Dataset
	TB_stat_DS.close()


	# Convert time back to seconds since 1970-01-01 00:00:00 UTC for plotting:
	time = np.asarray([(ttt - datetime.datetime(1970,1,1,0,0,0)).total_seconds() for ttt in time])



	# creating a figure showing the fit over all flights + correlation + RMSE + bias:

	fig2, ax2 = plt.subplots(7,4)
	fig2.set_size_inches(6.5,10.0)


	ax2 = ax2.flatten()
	ax2[26].axis('off')
	ax2[27].axis('off')


	for k in range(nfreq):

		# x_error for frequencies: 0.5 Kelvin for K & V and 1 Kelvin for the higher frq:
		if frequency[k] < 90:
			xerror = 0.5
		else:
			xerror = 1.0

		# axis limits: min and max TB of all:
		limits = np.asarray([np.nanmin(np.concatenate((tb_sonde_cf[:,k], tb_mean_cf[:,k]), axis=0))-2,
			np.nanmax(np.concatenate((tb_sonde_cf[:,k], tb_mean_cf[:,k]), axis=0))+2])

		if np.isnan(limits[0]):
			limits = np.array([0., 1.])

		eb = ax2[k].errorbar(tb_sonde_cf[:,k], tb_mean_cf[:,k], xerr=xerror, yerr=tb_std_cf[:,k], ecolor=(0,0,0), elinewidth=0.4, linestyle='none', \
		color=(0,0,0), label="All considered flights")

		# generate a linear fit with least squares approach: notes, p.2:
		nocloud_case = np.argwhere(~np.isnan(tb_mean_cf[:,k])).flatten()	# must be done to avoid nans being selected for the fit
		y_fit = tb_mean_cf[nocloud_case,k]
		x_fit = tb_sonde_cf[nocloud_case,k]


		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T		# must be transposed because of python's strange conventions
			m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)	# least squares solution
			a = m_fit[0]
			b = m_fit[1]

			ds_fit = ax2[k].plot(limits, a*limits + b, color=(0,0,0), linewidth=0.4)	# DS, a*DS + b

		# plot a line for orientation which would represent a perfect fit:
		ax2[k].plot(limits, limits, color=(0.6,0.6,0.6), linewidth=0.4)

		# fitting the other way round:
		y_fit_vv = tb_sonde_cf[nocloud_case,k]
		x_fit_vv = tb_mean_cf[nocloud_case,k]
		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit_vv) > 1) and (len(x_fit_vv) > 1):
			G_fit_vv = np.array([x_fit_vv, np.ones((len(x_fit_vv),))]).T		# must be transposed because of python's strange conventions
			m_fit_vv = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit_vv.T, G_fit_vv)), G_fit_vv.T), y_fit_vv)	# least squares solution
			a_vv = m_fit_vv[0]
			b_vv = m_fit_vv[1]

			# mwr_fit = ax2[k].plot(a_vv*limits + b_vv, limits, color=(0,0,0), linewidth=0.4)	# a*MWR + b, MWR

		ax2[k].text(0.01, 0.99, """\
bias = %.2f K
rmse = %.2f K
R = %.3f
N = %d"""%(bias[k], rmse[k], R[k], N[k]),
		horizontalalignment='left', verticalalignment='top', transform=ax2[k].transAxes, fontsize=4.5)

		ax2[k].set_ylim(ymin=limits[0], ymax=limits[1])
		ax2[k].set_xlim(xmin=limits[0], xmax=limits[1])
		ax2[k].tick_params(axis='both', which='major', labelsize=4.5, width=0.4, length=2, pad=2)

		ax2[k].set_aspect('equal')
		ax2[k].set_title(str(frequency[k]) + r"$\,$GHz", pad=2.0, fontsize=5)

		# reduce the width of the axis lines:
		for axlines in ['top', 'bottom', 'left', 'right']:
			ax2[k].spines[axlines].set_linewidth(0.4)

		# placing x and y axis labels on the outer plots
		if k == 24 or k == 25:
			ax2[k].set_xlabel(r"TB PAMTRA DS [K]", fontsize=4.5, labelpad=1.0)
		if k%4 == 0:
			ax2[k].set_ylabel(r"TB MWR [K]", fontsize=4.5, labelpad=1.0)


	# add two auxiliary line plots for pseudo legend entries:
	aux_ds = ax2[25].plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=0.4, label="x,y = DS, a*DS + b")
	aux_perfect_fit = ax2[25].plot([np.nan, np.nan], [np.nan, np.nan], color=(0.6,0.6,0.6), linewidth=0.4, label="theoretical perfect fit")

	# some axis positions for positioning of the legend & description
	frq25_posi = ax2[25].get_position()	# need y coordinates of this one
	frq22_posi = ax2[22].get_position()	# need x coordinates of this one


	# add legend to the "26th" subplot: get the handles from just one axis (it suffices because each axis at least theoretically includes a plot) ->
	# enough to create a legend: choose ax2[25] because it also contains the auxiliary labels created a few lines above:
	leg_handles, leg_labels = ax2[25].get_legend_handles_labels()
	i_am = ax2[26].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=4.0, title="Year-MM-DD, #sondes")
	# --> loc works like horiz.alignment: on this ANCHOR, the positioning is with respect to THIS point
	i_am.get_title().set_fontsize(4.0)		# must be done to set the title fontsize because title_fontsize doesnt work


	fig2.tight_layout()
	fig2.subplots_adjust(left=0.04, right=0.96, bottom=0.03, top=0.97, wspace=0.2, hspace=0.22)

	# is positioned after tight layout to avoid being shifted by it
	ax27_pos = ax2[27].get_position()
	posnew = [ax27_pos.x0, ax27_pos.y0, 0.8*ax27_pos.width, ax27_pos.height]
	ax2[27].set_position(posnew)
	ax2[27].text(0.02, 0.96, "Error bars are the std. of all \n TB measurements from 10 s \n before "\
	+ "the drop to 10 s \n after the drop. \n \n  The bias is calculated from all \n drops for which the std. deviation \n of the corresponding radiometer \n"\
	"measurements of all channels \n (except 183 bank) is below 1 K. \n  \n " + r"bias = TB$_{MWR}$ - TB$_{DS}$", fontsize=4.5, \
	horizontalalignment='left', verticalalignment='top', wrap=True, transform=ax2[27].transAxes)


	fig2.savefig(plot_path + scatterplot_name + ".png", dpi=400, bbox_inches='tight')
	fig2.savefig(plot_path + scatterplot_name + ".pdf", orientation='portrait')



	###
	# creating another figure showing the evolution of the bias over all flights:
	#

	if len(np.unique(TB_stat_DS.date)) <= 1:
		return # Plotting an evolution of 1 time step does not make sense

	fig3, ax3 = plt.subplots(7,4)
	fig3.set_size_inches(6.5,10.0)


	ax3 = ax3.flatten()
	ax3[26].axis('off')
	ax3[27].axis('off')
	ax3[25].axis('off')		# because the outermost G band channel no longer exists anyway ..produces nan only
	for k in range(nfreq-1):		# remove the last freq (last G band channel because it isn't active anyway)

		# daily bias:
		# find the indices in the time coordinate axis that indicate a new day (e.g. when day_index is 0 again):
		when_new_day = np.argwhere(day_index == 0).flatten()
		day_range = []
		for uwu in range(0, len(when_new_day)-1):
			day_range.append([when_new_day[uwu], when_new_day[uwu+1]-1])
		# append the indices of the last day:
		day_range.append([when_new_day[-1], len(day_index)-1])	# until len(...)-1 because of python indexing
		# day_range[0] now tells you the first index of that day (day_range[0][0]) and the last index of that day (day_range[0][1])
		# of course, for python you have to address day_range[0][1]+1 to actually catch the whole day

		dt_date = [datetime.datetime.strptime(work_date[dd], "%Y%m%d") for dd in when_new_day]		# datetime axis for each day (found via when_new_day)
		TB_bias_date = np.asarray([np.nanmean(tb_mean_cf[d_i[0]:d_i[1]+1,k] - tb_sonde_cf[d_i[0]:d_i[1]+1,k]) for d_i in day_range])


		# mark those days with a red dot when calibration has been performed:
		calibration_days = ['20200126', '20200128', '20200130', '20200131', '20200202', '20200205', '20200207', '20200209', '20200211', '20200213', '20200215', '20200218']
		calibration_days = [datetime.datetime.strptime(c_days, "%Y%m%d") for c_days in calibration_days]

		biasplot = ax3[k].plot(dt_date, TB_bias_date, color=(0,0,0), linewidth=0.4, marker='o', \
		markerfacecolor=(0,0,0), markersize=0.6)			# add - bias[k] for a relative bias evolution

		# marking the calibration days red
		# find indices when dt_date and calibration_days fit:
		calibration_days = [calib_day for calib_day in calibration_days if calib_day in dt_date]
		fittingtime = [dt_date.index(calib_day) for calib_day in calibration_days]
		extraplot = ax3[k].plot(calibration_days, TB_bias_date[fittingtime], color=(1,0,0), linestyle='none', marker='o', \
		markerfacecolor=(1,0,0), markersize=0.6, label='calibration')		# add - bias[k] for a relative bias evolution

		# auxiliary line at 0 bias:
		ax3[k].axhline(0, linewidth=0.25, color='black', linestyle=':')

		if np.all(np.isnan(TB_bias_date)):		# need to catch this frequency band because it produces nan only
			ax3[k].set_ylim(ymin=-1, ymax=1)
		else:
			y_limits = np.nanmax(np.abs(TB_bias_date))		# add - bias[k] for a relative bias evolution
			ax3[k].set_ylim(ymin=-y_limits, ymax=y_limits)

		ax3[k].tick_params(axis='both', which='major', labelsize=4.5, width=0.4, length=2, pad=2)

		# correct x ticks:
		ax3[k].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
		ax3[k].xaxis.set_major_locator(mdates.DayLocator(interval=7))
		ax3[k].xaxis.set_minor_locator(mdates.DayLocator(interval=1))	# so that we can read each day
		ax3[k].grid(True, which='both', axis='x', linewidth=0.25, linestyle=':')

		ax3[k].set_title(str(frequency[k]) + r"$\,$GHz", pad=2.0, fontsize=5)

		# reduce the width of the axis lines:
		for axlines in ['top', 'bottom', 'left', 'right']:
			ax3[k].spines[axlines].set_linewidth(0.4)

		# placing x and y axis labels on the outer plots
		if k >= 21:
			ax3[k].set_xlabel(r"2020", fontsize=4.5, labelpad=1.0)
		if k%4 == 0:
			ax3[k].set_ylabel(r"Bias$_{\mathrm{daily}}$ [K]", fontsize=4.5, labelpad=1.0)		# add  - Bias for relative bias

	# small legend @ ax3[25] location:
	# add legend to the "25th" subplot: get the handles from just one axis (it suffices because each axis at least theoretically includes a plot) ->
	# enough to create a legend: choose ax3[25] because it also contains the auxiliary labels created a few lines above:
	leg_handles, leg_labels = ax3[24].get_legend_handles_labels()
	i_am = ax3[25].legend(handles=leg_handles, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=4.0)
	# --> loc works like horiz.alignment: on this ANCHOR, the positioning is with respect to THIS point


	fig3.tight_layout()
	fig3.subplots_adjust(left=0.06, right=0.96, bottom=0.03, top=0.97, wspace=0.2, hspace=0.22)

	fig3.savefig(plot_path + bias_ev_plotname + ".png", dpi=400, bbox_inches='tight')
	fig3.savefig(plot_path + bias_ev_plotname + ".pdf", orientation='portrait')


def is_radar_clear_sky(path_RADAR_data, work_date_temp, datatime):
	# radar was not working on one day:
	if work_date_temp == '20200122': return True

	radar_ncfiles = sorted(glob.glob(path_RADAR_data + "*.nc"))
	radar_file = [radarfile for radarfile in radar_ncfiles if work_date_temp in radarfile][0]

	radar_dict = import_radar_nc(radar_file)

	idx = np.argwhere((datatime - 10 <= radar_dict['time']) & (radar_dict['time'] < datatime + 10))
	idx = np.reshape(idx, (idx.shape[0],))

	# time-averaged number of reflective bins
	count_reflective_bins = np.sum(radar_dict['dBZ'][idx, :] > -40) / len(idx)

	# it is considered clear sky, if the mean nurmber of reflective bins ins smaller than 2
	return count_reflective_bins < 2
