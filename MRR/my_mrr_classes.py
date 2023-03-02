import numpy as np
import datetime as dt
from private_importer import *
import pdb


class MRR:
	"""
		Micro-Rain-Radar (MRR):

		For initialisation, we need:
		path_r : str
			Path and filename of MRR data (.txt).

		**kwargs:
		with_datetime : bool
			If true, a time_dt variable will be created, saving the time also as an array
			of datetime objects. If false or not specified, this won't be done.
	"""

	def __init__(self, path_r, **kwargs):

		if "_mrr_ave-6-0-0-6" in path_r: 
			version = 'ave'
		elif "_mrr_raw" in path_r:
			version = 'raw'
		else:
			raise ValueError("Unknown MRR version. Make sure the data is '_mrr_ave-6-0-0-6' or '_mrr_raw'.")

		if version == 'ave':
			mrr_dict = import_mrr_ave(path_r, keys="all")
		elif version == 'raw':
			mrr_dict = import_mrr_raw(path_r, keys="all")


		self.time = mrr_dict['time']		# time in sec since 1970-01-01 00:00:00 UTC
		self.height = mrr_dict['height']	# height in m above surface around MRR
		self.Z = mrr_dict['Z']		# radar reflectivity factor in dBZ ?? (no information on units given)
		self.z = mrr_dict['z']		# radar reflectivity factor, but smaller (no info given)
		self.RR = mrr_dict['RR']	# rain rate in mm h^-1
		self.LWC = mrr_dict['LWC']	# liquid water content in g m^-3
		self.W = mrr_dict['W']		# mean vertical (fall) velocity (first moment of doppler spectrum) in m s-^1?
		
		if kwargs['with_datetime']:
			self.time_dt = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in self.time])