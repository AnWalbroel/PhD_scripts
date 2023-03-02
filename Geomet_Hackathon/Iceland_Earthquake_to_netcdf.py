import numpy as np
import datetime as dt
import pdb
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def import_Iceland_met_office_earthquake_table(filename):

	"""
	Importing a .txt file of darthquake data taken from
	https://en.vedur.is/earthquakes-and-volcanism/earthquakes/#view=table
	and filtering out unnecessary data. It returns datetime (in seconds
	since 1970-01-01 00:00:00 UTC, lat, lon, depth and magnitude information.

	Parameters:
	-----------
	filename : str
		Includes the filename and path of the all_iwv.txt.
	"""

	reftime = dt.datetime(1970,1,1)

	headersize = 1
	file_handler = open(filename, 'r')
	list_of_lines = list()

	n_data_optimistic = 10000
	earthquake_dict = {'time': np.full((n_data_optimistic,), np.nan),
					'lat': np.full((n_data_optimistic,), np.nan),
					'lon': np.full((n_data_optimistic,), np.nan),
					'depth': np.full((n_data_optimistic,), np.nan),		# in km
					'magnitude': np.full((n_data_optimistic,), np.nan)}

	m = 0		# used as index to cycle through earthquake_dict data
	for k, line in enumerate(file_handler):

		k_even = k%2 == 0
		if k >= headersize and k_even:	# skip header
			current_line = line.strip().split()	# split by tabs

			# convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
			datestring_now = current_line[0] + " " + current_line[1]
			earthquake_dict['time'][m] = (dt.datetime.strptime(datestring_now, "%d.%m.%Y %H:%M:%S") - reftime).total_seconds()

			# extract geoinfo, magnitude and depth:
			earthquake_dict['lat'][m] = float(current_line[2])		# in deg N
			earthquake_dict['lon'][m] = float(current_line[3])		# in deg E
			earthquake_dict['depth'][m] = float(current_line[4])		# in km
			earthquake_dict['magnitude'][m] = float(current_line[6])
			m = m + 1

		elif k >= headersize and (not k_even): # only contains the weekday -> irrelevant
			continue

		elif k == headersize-1: continue

		else:
			raise ValueError("Some unexpected line occurred in %s."%filename)

	# truncate redundant lines:
	last_nonnan = np.where(~np.isnan(earthquake_dict['time']))[0][-1] + 1 	# + 1 because of python indexing
	for key in earthquake_dict.keys(): earthquake_dict[key] = earthquake_dict[key][:last_nonnan]

	# sort by ascending order:
	s_idx = np.argsort(earthquake_dict['time'])
	for key in earthquake_dict.keys(): earthquake_dict[key] = earthquake_dict[key][s_idx]

	return earthquake_dict


def plot_simple(earthquake_dict, save_fig):
	fs = 16			# fontsize

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')		# 111: one row, one column and at the position "1"
	fig.set_size_inches(10,10)

	ax.plot(earthquake_dict['lon'], earthquake_dict['lat'], -earthquake_dict['depth'], linestyle='None',
			marker='o', mec=(0,0,0), mfc=(0,0,0), ms=1.5)

	ax.set_xlim(left=-24.5, right=-19.5)
	ax.set_ylim(bottom=63.5, top=64.5)
	ax.set_zlim(0, -15)

	ax.set_xlabel("Longitude ($^{\circ}\mathrm{E}$)", fontsize=fs)
	ax.set_ylabel("Latitude ($^{\circ}\mathrm{N}$)", fontsize=fs)
	ax.set_zlabel("Depth (km)", fontsize=fs)
	dt_range_string = (dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][0]), "%Y-%m-%d %H:%M:%S") + " - " +
					dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][-1]), "%Y-%m-%d %H:%M:%S"))
	ax.set_title("Earthquake depths 3D; " + dt_range_string,
					fontsize=fs)
	ax.invert_zaxis()

	if save_fig: fig.savefig(path_data + "EQ_depth_" + dt_range_string.replace(":","") + ".png", dpi=400)
	else: plt.show()


def plot_time_coloured(earthquake_dict, save_fig):
	fs = 16			# fontsize

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')		# 111: one row, one column and at the position "1"
	fig.set_size_inches(10,10)

	n_data = len(earthquake_dict['time'])
	c_map = plt.cm.get_cmap('jet', n_data)
	for k in range(n_data):
		c_now = c_map(k)
		ax.plot([earthquake_dict['lon'][k], earthquake_dict['lon'][k]], 
				[earthquake_dict['lat'][k], earthquake_dict['lat'][k]],
				[-earthquake_dict['depth'][k], -earthquake_dict['depth'][k]],
				linestyle='None',
				marker='o', mec=c_now, mfc=c_now, ms=1.5)

	ax.set_xlim(left=-24.5, right=-19.5)
	ax.set_ylim(bottom=63.5, top=64.5)
	ax.set_zlim(0, -15)

	# # dummy contour:
	# ax2 = fig.add_subplot(122)
	# # ax2.axis('off')
	# CFp = ax2.contour(np.full_like(earthquake_dict['time'], np.nan), np.full_like(earthquake_dict['time'], np.nan), np.repeat(earthquake_dict['time'], n_data, axis=1),
				# cmap = plt.cm.jet)
	# plt.colorbar(mappable=CFp, ax=ax2)

	ax.set_xlabel("Longitude ($^{\circ}\mathrm{E}$)", fontsize=fs)
	ax.set_ylabel("Latitude ($^{\circ}\mathrm{N}$)", fontsize=fs)
	ax.set_zlabel("Depth (km)", fontsize=fs)
	dt_range_string = (dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][0]), "%Y-%m-%d %H:%M:%S") + " - " +
					dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][-1]), "%Y-%m-%d %H:%M:%S"))
	ax.set_title("Earthquake depths 3D, time coloured; \n" + dt_range_string,
					fontsize=fs)
	ax.invert_zaxis()

	if save_fig: fig.savefig(path_data + "EQ_depth_" + dt_range_string.replace(":","") + "_time.png", dpi=400)
	else: plt.show()


def plot_magnitude_coloured(earthquake_dict, save_fig):
	fs = 16			# fontsize

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')		# 111: one row, one column and at the position "1"
	fig.set_size_inches(10,10)

	n_data = len(earthquake_dict['time'])
	c_map = plt.cm.get_cmap('jet', n_data)

	# need to sort for magnitudes occurring:
	mag_sort_idx = np.argsort(earthquake_dict['magnitude'])

	for k in range(n_data):
		c_now = c_map(mag_sort_idx[k])
		ax.plot([earthquake_dict['lon'][k], earthquake_dict['lon'][k]], 
				[earthquake_dict['lat'][k], earthquake_dict['lat'][k]],
				[-earthquake_dict['depth'][k], -earthquake_dict['depth'][k]],
				linestyle='None',
				marker='o', mec=c_now, mfc=c_now, ms=1.5)

	ax.set_xlim(left=-24.5, right=-19.5)
	ax.set_ylim(bottom=63.5, top=64.5)
	ax.set_zlim(0, -15)

	ax.set_xlabel("Longitude ($^{\circ}\mathrm{E}$)", fontsize=fs)
	ax.set_ylabel("Latitude ($^{\circ}\mathrm{N}$)", fontsize=fs)
	ax.set_zlabel("Depth (km)", fontsize=fs)
	dt_range_string = (dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][0]), "%Y-%m-%d %H:%M:%S") + " - " +
					dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][-1]), "%Y-%m-%d %H:%M:%S"))
	ax.set_title("Earthquake depths 3D, Magnitude coloured; \n" + dt_range_string,
					fontsize=fs)
	ax.invert_zaxis()

	if save_fig: fig.savefig(path_data + "EQ_depth_" + dt_range_string.replace(":","") + "_magnitude.png", dpi=400)
	else: plt.show()



#################################################
#################################################


path_data = "/mnt/d/alanw64/Downloads/Iceland_Earthquakes/"
eq_data = "Erd_Quack_dada9.txt"


# Import earthquake data:
earthquake_dict = import_Iceland_met_office_earthquake_table(path_data + eq_data)

# # Export as netcdf: Convert to xarray Dataset:
# EQ_DS = xr.Dataset({
					# 'lat':			(['time'], earthquake_dict['lat'],
									# {'description': "Latitude",
									# 'units': "deg N"}),
					# 'lon':			(['time'], earthquake_dict['lon'],
									# {'description': "Longitude",
									# 'units': "deg E"}),
					# 'depth':		(['time'], earthquake_dict['depth'],
									# {'description': "Depth of earthquake epicentre",
									# 'units': "km"}),
					# 'magnitude':	(['time'], earthquake_dict['magnitude'],
									# {'description': "Earthquake magnitude according to Richter Scale",
									# 'units': ""})},
					# coords =		{'time': 	(['time'], earthquake_dict['time'],
												# {'description': "Time",
												# 'units': "seconds since 1970-01-01 00:00:00 UTC"})})

# # Global attributes:
# EQ_DS.attrs['description'] = "Iceland earthquake data"
# EQ_DS.attrs['daterange'] = (dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][0]), "%Y-%m-%d %H:%M:%S") + " - " +
							# dt.datetime.strftime(dt.datetime.utcfromtimestamp(earthquake_dict['time'][-1]), "%Y-%m-%d %H:%M:%S"))
# EQ_DS.attrs['data_source'] = "https://en.vedur.is/earthquakes-and-volcanism/earthquakes/#view=table"
# EQ_DS.attrs['author'] = "Andreas Walbroel"
# EQ_DS.attrs['history'] = "Created on: " + dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# EQ_DS.to_netcdf(path_data + eq_data.replace(".txt", ".nc"), mode='w', format="NETCDF4")
# EQ_DS.close()


# Plotting attempts:
save_fig = True
# plot_simple(earthquake_dict, save_fig)

plot_time_coloured(earthquake_dict, save_fig)

# plot_magnitude_coloured(earthquake_dict, save_fig)
