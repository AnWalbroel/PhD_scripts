import numpy as np
import xarray as xr
from import_data import import_mirac_BRT_RPG_daterange
import matplotlib.pyplot as plt
import datetime as dt
import glob
import pdb


"""
	Investigate MiRAC-P noise strength during MOSAiC: special time series in clear sky: march and 
	clear sky in summer 
	Entire MOSAiC period: 1 minute averages (not moving avg) with 1 minute std dev.
			---> then plot histogram of std dev for each frq.
"""

path_data_obs = "/data/obs/campaigns/mosaic/mirac-p/l1/"

qucik_time_eries = False
create_nc = False

aux_info = dict()
aux_info['considered_period'] = "user"
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'user': ["2020-03-05", "2020-03-05"],
					'mosaic': ["2019-09-20", "2020-10-12"]}
aux_info['date_start'] = daterange_options[aux_info['considered_period']][0]	# def: "2019-09-30"
aux_info['date_end'] = daterange_options[aux_info['considered_period']][1]		# def: "2020-10-02"

if qucik_time_eries or create_nc:
	mwr_dict = import_mirac_BRT_RPG_daterange(path_data_obs, aux_info['date_start'], aux_info['date_end'],
												verbose=1)


if qucik_time_eries:
	mwr_dict['time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in mwr_dict['time']])
	fs = 15

	f1 = plt.figure(figsize=(12,10))
	a1 = plt.subplot2grid((3,1), (0,0), rowspan=1)
	a2 = plt.subplot2grid((3,1), (1,0), rowspan=1)
	a3 = plt.subplot2grid((3,1), (2,0), rowspan=1)


	a1.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,0], color='r', label=str(mwr_dict['Freq'][0]))
	a1.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,1], color='g', label=str(mwr_dict['Freq'][1]))
	a1.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,2], color='b', label=str(mwr_dict['Freq'][2]))

	a2.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,3], color='r', label=str(mwr_dict['Freq'][3]))
	a2.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,4], color='g', label=str(mwr_dict['Freq'][4]))
	a2.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,5], color='b', label=str(mwr_dict['Freq'][5]))

	a3.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,6], color='r', label=str(mwr_dict['Freq'][6]))
	a3.plot(mwr_dict['time_dt'], mwr_dict['TBs'][:,7], color='g', label=str(mwr_dict['Freq'][7]))


	a1.legend()
	a2.legend()
	a3.legend()

	a3.set_xlabel("Time")
	a2.set_ylabel("TB (K)")
	a1.set_title(f"Time series: {dt.datetime.strftime(mwr_dict['time_dt'][0], '%Y-%m-%d')}")

	plt.show()

# one minute averages and std dev:
if create_nc:

	# create a time axis that isn't fully broken:
	mwr_dict['time'] = mwr_dict['time'].astype(np.int64)
	time = np.arange(mwr_dict['time'][0], mwr_dict['time'][-1]+1)

	# interpolate on working time grid:
	mwr_dict['tb'] = np.zeros((len(time), len(mwr_dict['Freq'])))
	for k in range(len(mwr_dict['Freq'])): mwr_dict['tb'][:,k] = np.interp(time, mwr_dict['time'], mwr_dict['TBs'][:,k])
	mwr_dict['RF'] = np.interp(time, mwr_dict['time'], mwr_dict['RF'])

	pdb.set_trace()

	DS = xr.Dataset({'tb':			(['time', 'freq'], mwr_dict['tb']),
					'flag':			(['time'], mwr_dict['RF'])},
					coords=			{'time': (['time'], time.astype(np.int64),
												{'units': "seconds since 1970-01-01 00:00:00"}),
									'freq':	(['freq'], mwr_dict['Freq'])})

	DS['time'].encoding['units'] = "seconds since 1970-01-01 00:00:00"
	DS['time'].encoding['dtype'] = "int64"

	DS.to_netcdf("/net/blanc/awalbroe/Plots/quick_analysis_miracp_noise/MOSAiC_mirac-p_TB.nc", mode='w', format="NETCDF4")
	1/0


if (not create_nc) and (not qucik_time_eries):
	# load netcdf:
	DS = xr.open_dataset("/net/blanc/awalbroe/Plots/quick_analysis_miracp_noise/MOSAiC_mirac-p_TB.nc")

	aux_info['resample'] = "300"			# resampled to '...' seconds
	DS_std = DS.resample(time=f"{aux_info['resample']}S").std()


	# plt.plot(DS.time.sel(time="2020-07-26"), DS.tb.sel(time="2020-07-26").values[:,0])
	# plt.plot(DS_mean.time.sel(time="2020-07-26"), DS_mean.tb.sel(time="2020-07-26").values[:,0])


	f1 = plt.figure(figsize=(15,15))
	a1 = plt.subplot2grid((3,3), (0,0))
	a2 = plt.subplot2grid((3,3), (0,1))
	a3 = plt.subplot2grid((3,3), (0,2))
	a4 = plt.subplot2grid((3,3), (1,0))
	a5 = plt.subplot2grid((3,3), (1,1))
	a6 = plt.subplot2grid((3,3), (1,2))
	a7 = plt.subplot2grid((3,3), (2,0))
	a8 = plt.subplot2grid((3,3), (2,1))

	n_time = float(len(DS_std.time.values))
	n_time_int = int(n_time)
	weights_tb = np.ones((n_time_int,)) / n_time
	c_fill = (0,148/255,1)

	x_lim = [0, 1.0]
	a1.hist(DS_std.tb.values[:,0], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a2.hist(DS_std.tb.values[:,1], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a3.hist(DS_std.tb.values[:,2], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a4.hist(DS_std.tb.values[:,3], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a5.hist(DS_std.tb.values[:,4], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a6.hist(DS_std.tb.values[:,5], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a7.hist(DS_std.tb.values[:,6], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))
	a8.hist(DS_std.tb.values[:,7], bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.01), weights=weights_tb, 
						color=c_fill, ec=(0,0,0))

	a1.set_title(f"{DS.freq.values[0]:.2f}", pad=0.0)
	a2.set_title(f"{DS.freq.values[1]:.2f}", pad=0.0)
	a3.set_title(f"{DS.freq.values[2]:.2f}", pad=0.0)
	a4.set_title(f"{DS.freq.values[3]:.2f}", pad=0.0)
	a5.set_title(f"{DS.freq.values[4]:.2f}", pad=0.0)
	a6.set_title(f"{DS.freq.values[5]:.2f}", pad=0.0)
	a7.set_title(f"{DS.freq.values[6]:.2f}", pad=0.0)
	a8.set_title(f"{DS.freq.values[7]:.2f}", pad=0.0)

	a6.set_xlabel("$\sigma_{\mathrm{TB}}$ (K)")
	a7.set_xlabel("$\sigma_{\mathrm{TB}}$ (K)")
	a8.set_xlabel("$\sigma_{\mathrm{TB}}$ (K)")

	f1.savefig(f"/net/blanc/awalbroe/Plots/quick_analysis_miracp_noise/MOSAiC_mirac-p_TB_stddev_{aux_info['resample']}s_hist.png", dpi=300)
pdb.set_trace()