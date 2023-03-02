import numpy as np
import datetime as dt
import pdb
import sys
sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')    # so that functions and modules from MOSAiC can be used
from data_tools import *


def import_mrr_ave(
	filename,
	keys='all'):

	"""
	Imports ave-6-0-0-6.txt(.gz) data and returns it as dictionary.

	Parameters:
	-----------
	filename : str
		Path and filename of the mrr_ave data.
	keys : str or list of str
		Specify variables to be imported as list of str. Alternatively, set keys='all' to
		import all variables. Default: 'all'.
	"""

	n_l_expected = 3 + 7*31 + 1		# expected length of a line (manually inspected)

	with open(filename, mode='r') as f_handler:
		list_of_lines = list()

		if keys == 'all':
			n_rows = 25000				# assumed max. number of rows in the data file ('time')
			n_data = 31					# number of data entries in a line (inspected manually) ('height bins')
			n_spectral_reflectivity = 64		# number of FFT lines for spectral reflectivity
			n_DSD = 64					# number of drop size distribution (DSD) bins
			shape_time_height = (n_rows, n_data)
			shape_height = (n_data,)
			shape_time = (n_rows,)
			data_dict = {
				'time': np.full(shape_time, -9999.0),				# time in sec since 1970-01-01 00:00:00 UTC
				'height': np.full(shape_height, -9999.0),				# height in m above surface around MRR
				'spec_ref': np.full((n_rows, n_data, n_spectral_reflectivity), -9999.0),	# spectral reflectivity in dB
				'DSD': np.full((n_rows, n_data, n_DSD), -9999.0),	# drop size distribution in m^-3 m^-1 (per volume and class size)
				'Z': np.full(shape_time_height, -9999.0),			# radar reflectivity factor in dBZ ?? (no information on units given)
				'z': np.full(shape_time_height, -9999.0),			# radar reflectivity factor, but smaller (no info given)
				'RR': np.full(shape_time_height, -9999.0),			# rain rate in mm h^-1
				'LWC': np.full(shape_time_height, -9999.0),			# liquid water content in g m^-3
				'W': np.full(shape_time_height, -9999.0)}			# mean vertical (fall) velocity (first moment of doppler spectrum) in m s-^1?

			curr_time = -1			# used to indicate to which time the current line belongs to
			count_faulty_lines = 0
			for k, line in enumerate(f_handler):

				if line[:3] == 'MRR':		# then, next time step is found
					current_line = line.strip().split()
					curr_time += 1

					# convert time string to sec since 1970-01-01 00:00:00 UTC:
					curr_time_str = current_line[1]		# time string in yymmddHHMMSS
					curr_time_epoch = datetime_to_epochtime(dt.datetime.strptime(curr_time_str, "%y%m%d%H%M%S"))
					data_dict['time'][curr_time] = curr_time_epoch

				elif line[:3] == 'H  ':
					current_line = line.strip().split()
					height = np.asarray([float(hh) for hh in current_line[1:]])
					data_dict['height'] = height

				elif line[:3] == 'TF ': 		# skip (no idea, what this is)
					continue

				else:	# split columns of current line manually because the file has got horribly many empty spaces:
					current_line = list()
					current_line.append(line[:3])
					n_l = len(line)

					if n_l > n_l_expected:
						count_faulty_lines += 1
						continue			# then this line has got

					for j in range(3, n_l, 7): current_line.append(line[j:j+7])

					if current_line[0][0] == 'F':		# spectral reflectivity
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						spec_ref_bin = int(current_line[0][1:])
						data_dict['spec_ref'][curr_time, :, spec_ref_bin] = line_data

					elif current_line[0][0] == 'N':		# drop size distribution
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						dsd_bin = int(current_line[0][1:])
						data_dict['DSD'][curr_time, :, dsd_bin] = line_data

					elif current_line[0][0] == 'z':		# small reflectivity factor
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['z'][curr_time,:] = line_data

					elif current_line[0][0] == 'Z':		# reflectivity factor
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['Z'][curr_time,:] = line_data

					elif current_line[0] == 'RR ':		# rain rate
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['RR'][curr_time,:] = line_data

					elif current_line[0] == 'LWC':		# small reflectivity factor
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['LWC'][curr_time,:] = line_data

					elif current_line[0][0] == 'W':		# small reflectivity factor
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['W'][curr_time,:] = line_data


	if count_faulty_lines/k > 0.33:
		raise RuntimeError("The file '%s' seems to have many unexpectedly long lines. It is considered faulty."%(filename))

	# truncate unused dimensions:		
	for key in data_dict.keys():
		if key == 'time':
			data_dict[key] = data_dict[key][:curr_time+1]
		elif data_dict[key].shape == shape_time_height:
			data_dict[key] = data_dict[key][:curr_time+1,:]
		elif key in ['spec_ref', 'DSD']:
			data_dict[key] = data_dict[key][:curr_time+1,:,:]

	return data_dict


def import_mrr_raw(
	filename,
	keys='all'):

	"""
	Imports _mrr_raw.txt(.gz) data and returns it as dictionary.

	Parameters:
	-----------
	filename : str
		Path and filename of the mrr_ave data.
	keys : str or list of str
		Specify variables to be imported as list of str. Alternatively, set keys='all' to
		import all variables. Default: 'all'.
	"""

	n_l_expected = 3 + 9*32 + 1		# expected length of a line (manually inspected)

	with open(filename, mode='r', encoding='utf-8', errors='ignore') as f_handler:
		list_of_lines = list()

		if keys == 'all':
			n_rows = 50000				# assumed max. number of rows in the data file ('time')
			n_data = 32					# number of data entries in a line (inspected manually) ('height bins')
			n_spectral_reflectivity = 64		# number of FFT lines of spectral reflectivity
			shape_time_height = (n_rows, n_data)
			shape_height = (n_data,)
			shape_time = (n_rows,)
			data_dict = {
				'time': np.full(shape_time, -9999.0),				# time in sec since 1970-01-01 00:00:00 UTC
				'height': np.full(shape_height, -9999.0),			# height in m above surface around MRR
				'spec_ref': np.full((n_rows, n_data, n_spectral_reflectivity), -9999.0),	# spectral reflectivity in dB
				'TF': np.full(shape_time_height, -9999.0)}			# TF ..whatever that is

			curr_time = -1			# used to indicate to which time the current line belongs to
			count_faulty_lines = 0
			for k, line in enumerate(f_handler):

				if line[:3] == 'MRR':		# then, next time step is found
					current_line = line.strip().split()
					curr_time += 1

					# convert time string to sec since 1970-01-01 00:00:00 UTC:
					curr_time_str = current_line[1]		# time string in yymmddHHMMSS
					curr_time_epoch = datetime_to_epochtime(dt.datetime.strptime(curr_time_str, "%y%m%d%H%M%S"))
					data_dict['time'][curr_time] = curr_time_epoch

				elif line[:3] == 'H  ':
					current_line = line.strip().split()
					height = np.asarray([float(hh) for hh in current_line[1:]])
					data_dict['height'] = height

				else:	# split columns of current line manually because the file has got horribly many empty spaces:
					current_line = list()
					current_line.append(line[:3])
					n_l = len(line)

					if n_l > n_l_expected:
						count_faulty_lines += 1
						continue			# then this line has got

					for j in range(3, n_l, 9): current_line.append(line[j:j+9])

					if current_line[0][0] == 'F':		# spectral reflectivity
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						spec_ref_bin = int(current_line[0][1:])
						data_dict['spec_ref'][curr_time, :, spec_ref_bin] = line_data

					elif current_line[0] == 'TF ':		# no idea about this one either
						line_data = np.full((n_data,), -9999.0)
						for j, ld in enumerate(current_line[1:-1]):
							if not ld.isspace():			# then the entry contains spaces only
								line_data[j] = float(ld)

						data_dict['TF'][curr_time,:] = line_data


	if count_faulty_lines/k > 0.33:
		raise RuntimeError("The file '%s' seems to have many unexpectedly long lines. It is considered faulty."%(filename))

	# truncate unused dimensions:		
	for key in data_dict.keys():
		if key == 'time':
			data_dict[key] = data_dict[key][:curr_time+1]
		elif data_dict[key].shape == shape_time_height:
			data_dict[key] = data_dict[key][:curr_time+1,:]
		elif key in ['spec_ref']:
			data_dict[key] = data_dict[key][:curr_time+1,:,:]


	# Now, other data products need to be computed from this...


	return data_dict