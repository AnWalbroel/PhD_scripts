# import matplotlib.pyplot as plt
import numpy as np
import pdb
import datetime as dt
import sys
from geopy import distance

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")
from data_tools import *


def run_check_halo_track():

	path_data = "/net/blanc/awalbroe/Data/HALO_AC3/"
	file = "HALO_track_2022-03-13.txt"


	with open(path_data + file, "r") as f:
		lines = f.readlines()

	# split by comma; extract time and coordinates:
	lat = list()
	lon = list()
	time = list()
	for line in lines:
		line_split = line.split(",")
		lat.append(line_split[2])
		lon.append(line_split[3])
		time.append(dt.datetime.strptime(line_split[1], "%Y-%m-%d %H:%M:%S"))

	lon = np.array(lon).astype("float")
	lat = np.array(lat).astype("float")
	time_dt = np.array(time)
	time = datetime_to_epochtime(time_dt)

	# Legs manually:

	# # Identified legs: RF04
	# leg_coords = {'1': [[67.71, 20.17], [77.7103, -10.7665]],
					# '2': [[77.672, -10.958], [77.6621, 9.100]],
					# '3': [[77.607, 9.665], [80.028, -6.748]],
					# '4': [[80.000, -6.724], [79.995, 9.283]],
					# '5': [[79.944, 9.167], [82.963, -3.268]],
					# '6': [[82.990, -3.785], [83.505, 22.080]],
					# '7': [[83.433, 22.304], [85.003, -4.032]],
					# '8': [[85.120, -3.468], [87.514, 12.280]],
					# '9': [[87.413, 11.580], [75.280, 2.349]],
					# '10': [[75.104, 2.577], [68.887, 17.826]]}



	# identify respective times:
	leg_time = dict()
	n_data = len(lon)
	for key in leg_coords.keys():
		geod_dist_start = np.ones(lat.shape)*(-999.0)
		geod_dist_end = np.ones(lat.shape)*(-999.0)

		for idx in range(n_data):
			# distance of start and end point of leg
			geod_dist_start[idx] = (distance.distance((lat[idx], lon[idx]), (leg_coords[key][0][0], leg_coords[key][0][1])).km)
			geod_dist_end[idx] = (distance.distance((lat[idx], lon[idx]), (leg_coords[key][1][0], leg_coords[key][1][1])).km)

		idx_min_start = np.argmin(geod_dist_start)
		idx_min_end = np.argmin(geod_dist_end)
		leg_time[key] = [time_dt[idx_min_start], time_dt[idx_min_end]]


	return leg_time