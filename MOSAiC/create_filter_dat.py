import numpy as np
import datetime as dt
# import pandas as pd
import copy
import pdb
import os
import glob
import warnings
from data_tools import *


"""
	This program aims to write the filter.dat file requried for MiRAC-P processing with
	the mwr_pro tool.
"""


def convert_outlier_datetime_to_dat_string(outliers_dt):

	"""
	Convert each outlier time to a string with yymmdd n hh.hh hh.hh (decimal hours!).

	Parameters:
	-----------
	outliers_dt : list of datetime objects
		List that includes the outlier start time (with minute precision) and end time
		for each outlier. E.g. outliers_dt[0][0] is the first outlier start datetime object,
		outliers_dt[0][1] its respective end time. outliers_dt[1][0] is the start time of
		the second outlier, and so on.
	"""

	# Run through the entire list:
	# First outlier of a day includes the DATE, TIME and NUMBER of outliers of that day.
	# Second outlier only includes TIME.
	n_outliers = len(outliers_dt)
	n_out = 0		# number of outliers for a day; will be set to 1 for each new day
	previous_outl_date = dt.datetime(1970,1,1)	# dummy first date
	# outlier_strings = np.chararray((n_outliers,))
	outlier_strings = list()
	for idx, outliers in enumerate(outliers_dt):
		current_outl_date = outliers[0].date()

		# start gathering the outliers from the current date
		if previous_outl_date != current_outl_date and n_out == 0: # then it's the first ever outlier:
			n_out = 1
			outliers_today = [outliers]

		elif previous_outl_date != current_outl_date and n_out > 0:	# then reset daily outlier counter n_out
			
			outliers_yesterday = outliers_today
			n_outliers_day = len(outliers_yesterday)		# number of outliers of 'yesterday'
			for nn, out_yes in enumerate(outliers_yesterday):
				if nn == 0:	# longer string required with DATE, TIME and NUMBER:
					outlier_strings.append(dt.datetime.strftime(out_yes[0], "%y%m%d") +
											" %2i"%(n_outliers_day) +
											" %02i.%02i"%(out_yes[0].hour, np.floor(100*out_yes[0].minute/60)) +
											" %02i.%02i"%(out_yes[1].hour, np.ceil(100*out_yes[1].minute/60)) +
											" 1 0 0")

				else:
					outlier_strings.append("          %02i.%02i %02i.%02i 1 0 0"%(out_yes[0].hour,
																				np.floor(100*out_yes[0].minute/60),
																				out_yes[1].hour,
																				np.ceil(100*out_yes[1].minute/60)))

			# now set the true outliers from current_outl_date to outliers_today:
			outliers_today = [outliers]

		else: # previous_outl_date equals current_outl_date -> increment n_out
			n_out = n_out + 1
			outliers_today.append(outliers)


		

		previous_outl_date = current_outl_date

	return outlier_strings


def write_filter_dat(
	filename,
	outlier_strings):

	"""
	Write outliers to filename.dat file

	Parameters:
	-----------
	filename : str
		Filename (including path) of the file (.dat) that contains
		the outliers according to the norm required for the mwr_pro tool.
	outlier_strings : list of str
		List of strings of each identified outlier according to the norm and formatting
		that is required for the mwr_pro tool.
	"""

	header = ['#This file contains manually set quality control flags for MWR data.',
				'#Faulty data times can be manually set and will be flagged in quicklooks and final netcdf products.',
				'#Possible reasons: disturbances on radome, radio-frequency interference, mis-calibration, ...',
				'#Note: TBs and products will still be available during the specified times.',
				'#1st column contains date of faulty data following format specification',
				'#2cnd column contains number of faulty intervals on one day',
				'#3rd and 4th column: start time and end time in decimal(!) hours - note, e.g. 19:30=19.50 ',
				'#5th column: set to 1 if Band1 channels are subject to error',
				'#6th column: set to 1 if Band2 (if existing) channels are subject to error',
				'#7th column: set to 1 if Band3 (if existing) channels are subject to error',
				'#The second to last line must always contain the string "date of last change"',
				'#The last row must contain the actual date of last change.',
				'#You must adhere to formats given in the example below!',
				'#BEGIN OF EXAMPLE',
				'yymmdd nn hh.hh hh.hh 1 2 3',
				'110117  1 19.00 21.00 1 0 0',
				'110118  2 10.00 11.00 1 0 0',
				'          12.50 14.50 1 0 0',
				'date of last change',
				'130710',
				'#END OF EXAMPLE',
				'160422  1 12.00 15.00 1 0 0']

	footer = ['date of last change', dt.datetime.utcnow().strftime("%y%m%d")]

	all_dat_lines = header + outlier_strings + footer

	# write stuff into file:
	file_handler = open(filename, 'w')

	for line in all_dat_lines:
		file_handler.write(line + "\n")

	file_handler.close()


path_mout = "/net/blanc/awalbroe/Codes/MOSAiC/"
outlier_file = path_mout + "MiRAC-P_outliers.txt"
path_out = "/net/blanc/awalbroe/Data/MOSAiC_radiometers/"
outlier_dat_filename = "filter_mosaic_mirac-p.dat"

# import outlier text file
outliers_dt = import_MiRAC_outliers(outlier_file)

# convert each outlier time to a string with yymmdd n hh.hh hh.hh:
outlier_strings = convert_outlier_datetime_to_dat_string(outliers_dt)

# save to .dat file:
write_filter_dat(path_out + outlier_dat_filename, outlier_strings)
