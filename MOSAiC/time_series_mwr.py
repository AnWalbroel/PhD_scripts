import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import glob
import gc
from import_data import import_mirac_level2a_daterange, import_hatpro_level2a_daterange, import_hatpro_level2b, import_hatpro_level2c
import pdb


path_data = {'mirac-p': "/data/obs/campaigns/mosaic/mirac-p/l2/",
				'hatpro': "/data/obs/campaigns/mosaic/hatpro/l2/"}
path_plots = "/net/blanc/awalbroe/Plots/time_series_mwr_level_2/"

aux_info = dict()
aux_info['considered_period'] = 'mosaic'	#f"leg{aux_info['mosaic_leg']}"	# specify which period shall be plotted or computed:
									# DEFAULT: 'mwr_range': 2019-09-30 - 2020-10-02
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'leg1': 2019-09-20 - 2019-12-13
									# 'leg2': 2019-12-13 - 2020-02-24
									# 'leg3': 2020-02-24 - 2020-06-04
									# 'leg4': 2020-06-04 - 2020-08-12
									# 'leg5': 2020-08-12 - 2020-10-12
									# ("leg%i"%(aux_info['mosaic_leg']))
									# 'user': user defined
daterange_options = {'mwr_range': ["2019-09-30", "2020-10-02"],
					'mosaic': ["2019-09-20", "2020-10-12"],
					'leg1': ["2019-09-20", "2019-12-12"],
					'leg2': ["2019-12-13", "2020-02-23"],
					'leg3': ["2020-02-24", "2020-06-03"],
					'leg4': ["2020-06-04", "2020-08-11"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-01-01", "2020-01-10"]}
aux_info['date_start'] = daterange_options[aux_info['considered_period']][0]	# def: "2019-09-30"
aux_info['date_end'] = daterange_options[aux_info['considered_period']][1]		# def: "2020-10-02"


# mirac_dict = import_mirac_level2a_daterange(path_data['mirac-p'], aux_info['date_start'], aux_info['date_end'], 
											# which_retrieval='iwv', vers='v01', verbose=1)
# mirac_dict['time64'] = mirac_dict['time'].astype("datetime64[s]")

# hatpro_dict_2a = import_hatpro_level2a_daterange(path_data['hatpro'], aux_info['date_start'], aux_info['date_end'], 
											# which_retrieval='lwp', vers='v01', verbose=1)
# hatpro_dict_2a['time64'] = hatpro_dict_2a['time'].astype("datetime64[s]")

# # build another importer for HATPRO level 2b data because the existing one is (for a good reason) filtering out 
# # lot of data per day to avoid a memory overload. So, I probably have to go through each 5 days, 
# # plot temperature of 10 heights in one subplot, do 4 subplots per figure. Need to repeat that also for hum profiles
# now_date = dt.datetime.strptime(aux_info['date_start'], "%Y-%m-%d")
# date_end_dt = dt.datetime.strptime(aux_info['date_end'], "%Y-%m-%d")
# n_days_range = 10
# quantity_2b = 'hua'			# must be 'ta' or 'hua' (for temperature or hum. profiles, respectively)
# n_height = 43				# number of height levels (investigated manually from inspecting the netcdf files)
# while now_date <= date_end_dt:
	# # select days where data will be imported:
	# sel_dates = [now_date + dt.timedelta(days=k) for k in range(n_days_range)]

	# # import data: loop over 5 days to concatenate stuff:
	# n_files = 0
	# for k, sel_date in enumerate(sel_dates):
		# yyyy = sel_date.year
		# mm = sel_date.month
		# dd = sel_date.day
		# path_date = f"{yyyy:4}/{mm:02}/{dd:02}/"

		# # search for files:
		# file_list = glob.glob(path_data['hatpro'] + path_date + f"ioppol_tro_mwr00_l2_{quantity_2b}_v01_*.nc")
		# if len(file_list) == 0:
			# continue

		# if k == 0:
			# hatpro_dict_2b = import_hatpro_level2b(file_list[0], keys='basic', minute_avg=False)
			# n_files += 1

		# else:
			# hd2b = import_hatpro_level2b(file_list[0], keys='basic', minute_avg=False)
			# n_files += 1
			# hatpro_dict_2b[quantity_2b] = np.concatenate((hatpro_dict_2b[quantity_2b], hd2b[quantity_2b]), axis=0)
			# hatpro_dict_2b['time'] = np.concatenate((hatpro_dict_2b['time'], hd2b['time']))
			# hatpro_dict_2b['flag'] = np.concatenate((hatpro_dict_2b['flag'], hd2b['flag']))

			# del hd2b

	# # plot only if data was found in this sel_dates range:
	# if n_files > 0:
		
		# # plot data time series:
		# f1, (a1, a2, a3, a4) = plt.subplots(4,1)
		# f1.set_size_inches(15,8)
		# hatpro_dict_2b['time64'] = hatpro_dict_2b['time'].astype("datetime64[s]")
		# for hh in range(0,10):	# lowest 10 height levels
			# a1.plot(hatpro_dict_2b['time64'], hatpro_dict_2b[quantity_2b][:,hh], label=str(hh))
		# a1.legend(loc='upper right', fontsize=8, ncol=4)

		# for hh in range(10,20):	# lowest 10 height levels
			# a2.plot(hatpro_dict_2b['time64'], hatpro_dict_2b[quantity_2b][:,hh], label=str(hh))
		# a2.legend(loc='upper right', fontsize=8, ncol=4)

		# for hh in range(20,30):	# lowest 10 height levels
			# a3.plot(hatpro_dict_2b['time64'], hatpro_dict_2b[quantity_2b][:,hh], label=str(hh))
		# a3.legend(loc='upper right', fontsize=8, ncol=4)

		# for hh in range(30,n_height):	# lowest 10 height levels
			# a4.plot(hatpro_dict_2b['time64'], hatpro_dict_2b[quantity_2b][:,hh], label=str(hh))
		# a4.legend(loc='upper right', fontsize=8, ncol=4)

		# f1.savefig((path_plots + f"MOSAiC_hatpro_l2_{quantity_2b}_all_{dt.datetime.strftime(sel_dates[0], '%Y%m%d')}-" +
					# f"{dt.datetime.strftime(sel_dates[-1], '%Y%m%d')}.png"), dpi=350)
		# f1.clf()
		# plt.close()
		# gc.collect()

		# del hatpro_dict_2b
	# now_date += dt.timedelta(days=n_days_range)




# # Plot simple time series:
# f1, (a1, a2) = plt.subplots(2,1)

# a1.plot(mirac_dict['time64'][mirac_dict['flag'] == 16], mirac_dict['prw'][mirac_dict['flag'] == 16], color=(0,0,1,0.5), label="mirac-p")
# a1.plot(hatpro_dict_2a['time64'][hatpro_dict_2a['flag']==0], hatpro_dict_2a['prw'][hatpro_dict_2a['flag']==0], color=(1,0,0,0.5), label="hatpro")
# a2.plot(mirac_dict['time64'], mirac_dict['flag'], color=(0,0,0))

# a1.legend()
# plt.show()
# # f1.savefig(path_plots + "MOSAiC_mirac-p_prw_time_series.png", dpi=350)


# f1, (a1, a2) = plt.subplots(2,1)

# a1.plot(hatpro_dict_2a['time64'], hatpro_dict_2a['clwvi'], color=(0,0,0))
# a2.plot(hatpro_dict_2a['time64'], hatpro_dict_2a['flag'], color=(0,0,0))

# plt.show()
# f1.savefig(path_plots + "MOSAiC_hatpro_prw_time_series.png", dpi=350)


# build another importer for HATPRO level 2c data
now_date = dt.datetime.strptime(aux_info['date_start'], "%Y-%m-%d")
date_end_dt = dt.datetime.strptime(aux_info['date_end'], "%Y-%m-%d")
n_days_range = 10
quantity_2c = 'ta'			# must be 'ta'
n_height = 43				# number of height levels (investigated manually from inspecting the netcdf files)
while now_date <= date_end_dt:
	# select days where data will be imported:
	sel_dates = [now_date + dt.timedelta(days=k) for k in range(n_days_range)]

	# import data: loop over 5 days to concatenate stuff:
	n_files = 0
	for k, sel_date in enumerate(sel_dates):
		yyyy = sel_date.year
		mm = sel_date.month
		dd = sel_date.day
		path_date = f"{yyyy:4}/{mm:02}/{dd:02}/"

		# search for files:
		file_list = glob.glob(path_data['hatpro'] + path_date + f"ioppol_tro_mwrBL00_l2_{quantity_2c}_v01_*.nc")
		if len(file_list) == 0:
			continue

		if k == 0:
			hatpro_dict_2b = import_hatpro_level2c(file_list[0], keys='basic')
			n_files += 1

		else:
			hd2b = import_hatpro_level2c(file_list[0], keys='basic')
			n_files += 1
			hatpro_dict_2b[quantity_2c] = np.concatenate((hatpro_dict_2b[quantity_2c], hd2b[quantity_2c]), axis=0)
			hatpro_dict_2b['time'] = np.concatenate((hatpro_dict_2b['time'], hd2b['time']))
			hatpro_dict_2b['flag'] = np.concatenate((hatpro_dict_2b['flag'], hd2b['flag']))

			del hd2b

	# plot only if data was found in this sel_dates range:
	if n_files > 0:
		
		# plot data time series:
		f1, (a1, a2, a3, a4) = plt.subplots(4,1)
		f1.set_size_inches(15,8)
		hatpro_dict_2b['time64'] = hatpro_dict_2b['time'].astype("datetime64[s]")
		for hh in range(0,10):	# lowest 10 height levels
			a1.plot(hatpro_dict_2b['time64'][hatpro_dict_2b['flag'] == 0], hatpro_dict_2b[quantity_2c][hatpro_dict_2b['flag'] == 0,hh], label=str(hh))
		a1.legend(loc='upper right', fontsize=8, ncol=4)

		for hh in range(10,20):	# lowest 10 height levels
			a2.plot(hatpro_dict_2b['time64'][hatpro_dict_2b['flag'] == 0], hatpro_dict_2b[quantity_2c][hatpro_dict_2b['flag'] == 0,hh], label=str(hh))
		a2.legend(loc='upper right', fontsize=8, ncol=4)

		for hh in range(20,30):	# lowest 10 height levels
			a3.plot(hatpro_dict_2b['time64'][hatpro_dict_2b['flag'] == 0], hatpro_dict_2b[quantity_2c][hatpro_dict_2b['flag'] == 0,hh], label=str(hh))
		a3.legend(loc='upper right', fontsize=8, ncol=4)

		for hh in range(30,n_height):	# lowest 10 height levels
			a4.plot(hatpro_dict_2b['time64'][hatpro_dict_2b['flag'] == 0], hatpro_dict_2b[quantity_2c][hatpro_dict_2b['flag'] == 0,hh], label=str(hh))
		a4.legend(loc='upper right', fontsize=8, ncol=4)

		f1.savefig((path_plots + f"MOSAiC_hatpro_l2_BL_{quantity_2c}_nonflagged_{dt.datetime.strftime(sel_dates[0], '%Y%m%d')}-" +
					f"{dt.datetime.strftime(sel_dates[-1], '%Y%m%d')}.png"), dpi=350)
		f1.clf()
		plt.close()
		gc.collect()

		del hatpro_dict_2b
	now_date += dt.timedelta(days=n_days_range)


print("Done....")