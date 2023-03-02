import pdb
import numpy as np
import xarray as xr
import glob
import datetime as dt
import matplotlib.pyplot as plt
from import_data import *
from met_tools import *
from data_tools import *
from my_classes import cloudnet, radiosondes, radiometers


class Neural_Network_v02:
	"""
		Simple Neural Network that follows the instructions and computations in
		https://realpython.com/python-ai-neural-network/.

		For initialisation, we need:
		input_vector : array of floats
			Input vector of the Neural Network training data (predictors, e.g. all MiRAC-P freq. TBs).
		target : array of floats
			Target of a prediction for the Neural Network of the training data (predictands, 
			e.g. IWV).
		learning_rate : float
			Learning rate of the Neural Network. Usually a small number (0.00001 - 0.1) to
			avoid skipping the 0 error point. Default: 0.0001
		scaling_factor : float
			Scaling factor for layer 1 to keep the sigmoid function from saturating (becoming either
			0 or 1 for all values outside a certain range). Default: 0.0001
	"""

	def __init__(self, input_vector, target, learning_rate=0.0001, scaling_factor=0.0001):
		self.input_vector = input_vector
		self.target = target
		self.learning_rate = learning_rate
		self.scaling_factor = scaling_factor

		# initialise bias and weights (for level 1):
		if self.input_vector.ndim == 1:
			self.weights = np.random.rand(self.input_vector.shape[0])
			self.weights_2 = np.random.rand(self.input_vector.shape[0])
		elif self.input_vector.ndim == 2:
			self.weights = np.random.rand(self.input_vector.shape[1])
			self.weights_2 = np.random.rand(self.input_vector.shape[1])
		self.bias = np.random.rand(1)

		# If target not in [-1, 1], translate it to those limits:
		# t_min and t_max define the range of possible predictand values
		if (np.nanmin(target) < 0) or (np.nanmax(target) > 1):
			t_min = -5		# here: min possible IWV
			t_max = 25		# here: max possible IWV in kg m^-2

			# translation coeffs; see notes, p.58
			self.a = 1 / (t_max - t_min)
			self.b = -self.a*t_min
			self.target = self.translate_target_fwd(self.target)

	def make_prediction(self, input_vector):
		self.layer_1 = self.scaling_factor*(np.dot(input_vector, self.weights) + np.dot(input_vector*input_vector, self.weights_2) + self.bias)
		self.layer_2 = sigmoid(self.layer_1)
		self.prediction = self.layer_2

	def translate_target_fwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target = self.a*target + self.b

		return target

	def translate_target_bwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target =  (target - self.b) / self.a

		return target

	def update_parameters(self, input, target):
		# compute error
		x = np.log(self.prediction + 1) - np.log(target + 1)
		# self.error = x**2; not needed
		self.error_deriv = 2*x/(self.prediction+1)

		derror_dlayer1 = self.error_deriv * sigmoid(self.layer_1) * (1 - sigmoid(self.layer_1))

		# update weights:
		derror_dweights = derror_dlayer1*self.scaling_factor*input
		self.weights = self.weights - self.learning_rate*derror_dweights

		derror_dweights2 = derror_dlayer1*self.scaling_factor*input*input
		self.weights_2 = self.weights_2 - self.learning_rate*derror_dweights2

		# update bias:
		derror_dbias = derror_dlayer1*self.scaling_factor
		self.bias = self.bias - self.learning_rate*derror_dbias

	def train_NN(self, iterations=100000):

		"""
		Parameters.
		-----------
		iterations : int
			Number of iterations to train the retrieval.
		"""

		if self.input_vector.ndim == 2: n_cases = self.input_vector.shape[0]

		# save mean square error over entire data set (all cases) for every
		# 100th iteration:
		MSE = list()

		for itt in range(iterations):
			# Pick training data case:

			# if itt % 10 == 0: print(itt)
			square_error = np.zeros((n_cases,))			# square error for current iteration
			for case in range(n_cases):
				# predict and update weights
				self.make_prediction(self.input_vector[case,:])	# sets attribute 'prediction' to NN
				self.update_parameters(input_vector[case,:], self.target[case])

				square_error[case] = (self.prediction - self.target[case])**2

			MSE.append(np.mean(square_error))

		return MSE

	def test_NN(self, input, target):

		"""
		Parameters:
		-----------
		input : array of floats
			Input (predictors).
		target : array of floats
			Target (predictand).
		"""

		target_target = self.translate_target_fwd(target)	# target in target space ([0, 1])

		n_cases = len(target)
		IWV_predicted = np.zeros((n_cases,))
		predicted = np.zeros((n_cases,))
		square_error_target = np.zeros((n_cases,))
		square_error_iwv = np.zeros((n_cases,))
		for case in range(n_cases):
			self.make_prediction(input[case,:])		# sets attribute 'prediction' to NN

			predicted[case] = self.prediction
			square_error_target[case] = (self.prediction - target_target[case])**2
			IWV_predicted[case] = self.translate_target_bwd(self.prediction)
			square_error_iwv[case] = (IWV_predicted[case] - target[case])**2
		
		test_dict = {	'square_error_target': square_error_target,
						'square_error_iwv': square_error_iwv,
						'prediction': predicted,
						'IWV_target': target,
						'IWV_pred': IWV_predicted}
		return test_dict

	def scatter_plot_retrieval_output(self, x_stuff, y_stuff):
		fs = 18

		fig1 = plt.figure(figsize=(22,10))
		ax0 = plt.axes()

		# Compute statistics for scatterplot:
		stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

		sc_N = stats_dict['N']
		sc_bias = stats_dict['bias']
		sc_rmse = stats_dict['rmse']
		sc_R = stats_dict['R']

			
		# ------------------------------------- #


		ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.',
					color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


		# diagonal line:
		xlims = np.asarray([0, np.nanmax(np.array([y_stuff,x_stuff]))+3])
		ylims = xlims
		ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		x_fit = x_stuff
		y_fit = y_stuff

		mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

		y_fit = y_fit[mask]
		x_fit = x_fit[mask]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			slope, offset = np.polyfit(x_fit, y_fit, 1)
			ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

		ax0.set_xlim(left=xlims[0], right=xlims[1])
		ax0.set_ylim(bottom=ylims[0], top=ylims[1])

				# add statistics:
		ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f \nLR=%s \nSF=%s"%(sc_N, 
				np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R, str(self.learning_rate),
				str(self.scaling_factor)),
				horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

		leg_handles, leg_labels = ax0.get_legend_handles_labels()
		ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

		ax0.set_aspect('equal', 'box')

		ax0.set_title("Retrieved IWV (pred) vs. observed IWV (obs)", fontsize=fs, pad=0)
		ax0.set_xlabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax0.minorticks_on()
		ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax0.tick_params(axis='both', labelsize=fs-4)

		plt.show()


class Neural_Network_v01:
	"""
		Simple Neural Network that follows the instructions and computations in
		https://realpython.com/python-ai-neural-network/.

		Some changes regarding the input vector and the handling of its scales have been made
		since Neural_Network_v00 (see notes, p.60 f).

		For initialisation, we need:
		input_vector : array of floats
			Input vector of the Neural Network training data (predictors, e.g. all MiRAC-P freq. TBs).
		target : array of floats
			Target of a prediction for the Neural Network of the training data (predictands, 
			e.g. IWV).
		learning_rate : float
			Learning rate of the Neural Network. Usually a small number (0.00001 - 0.1) to
			avoid skipping the 0 error point. Default: 0.0001
		scaling_factor : float
			Scaling factor for layer 1 to keep the sigmoid function from saturating (becoming either
			0 or 1 for all values outside a certain range). Default: 0.0001
	"""

	def __init__(self, input_vector, target, learning_rate=0.0001, scaling_factor=0.0001):
		self.input_vector = input_vector
		self.target = target
		self.learning_rate = learning_rate
		self.scaling_factor = scaling_factor

		# initialise bias and weights (for level 1):
		# We need to constrain the weights to certain values so that the output of layer_1 is within
		# [-2, 2] because other values than that cause sigm(x) to be rather saturated.
		weight_TB = 0.8
		weight_DOY = 0.1
		weight_bias = 0.1

		# -> these weights can be adjusted but must add up to 1 !
		if weight_TB + weight_DOY + weight_bias != 1:
			raise ValueError("weight_TB + weight_DOY + weight_bias must equal 1!")

		# compute scaling parameters for the input vector components (for linear and quadratic
		# order as both orders are used in layer_1):
		idx_TB = np.arange(8)
		idx_DOY = np.arange(8,10)
		n_TB = len(idx_TB)
		n_DOY = len(idx_DOY)
		self.sf_TB_1 = weight_TB / np.nanmax(np.sum(input_vector[:,idx_TB], axis=1))
		self.sf_TB_2 = weight_TB / np.nanmax(np.sum((input_vector*input_vector)[:,idx_TB], axis=1))
		self.sf_DOY_1 = weight_DOY / np.nanmax(np.sum(input_vector[:,idx_DOY], axis=1))
		self.sf_DOY_2 = weight_DOY / np.nanmax(np.sum((input_vector*input_vector)[:,idx_DOY], axis=1))
		self.sf_bias = weight_bias*2			# *2 because interval is [-2, 2]

		# scaling factors
		self.sf_weights = np.concatenate((np.repeat(self.sf_TB_1, n_TB), np.repeat(self.sf_DOY_1, n_DOY)), axis=0)
		self.sf_weights_2 = np.concatenate((np.repeat(self.sf_TB_2, n_TB), np.repeat(self.sf_DOY_2, n_DOY)), axis=0)


		# set the weights of the input vector components with their respective scaling factors:
		# also set the "bias" with scaled rng:
		weights_TB_1 = np.random.rand(n_TB)
		weights_DOY_1 = np.random.rand(n_DOY)
		weights_TB_2 = np.random.rand(n_TB)
		weights_DOY_2 = np.random.rand(n_DOY)

		self.weights = np.concatenate((weights_TB_1, weights_DOY_1), axis=0)
		self.weights_2 = np.concatenate((weights_TB_2, weights_DOY_2), axis=0)

		self.bias = np.random.rand(1)

		# If target not in [-1, 1], translate it to those limits:
		# t_min and t_max define the range of possible predictand values
		if (np.nanmin(target) < 0) or (np.nanmax(target) > 1):
			t_min = -5		# here: min possible IWV
			t_max = 25		# here: max possible IWV in kg m^-2

			# translation coeffs; see notes, p.58
			self.a = 1 / (t_max - t_min)
			self.b = -self.a*t_min
			self.target = self.translate_target_fwd(self.target)

	def make_prediction(self, input_vector):
		self.layer_1 = (np.dot(input_vector, (self.weights*self.sf_weights)) + 
						np.dot(input_vector*input_vector, (self.weights_2*self.sf_weights_2)) + self.bias*self.sf_bias)
		if np.abs(self.layer_1) > 5: pdb.set_trace()
		self.layer_2 = sigmoid(self.layer_1)
		self.prediction = self.layer_2

	def translate_target_fwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target = self.a*target + self.b

		return target

	def translate_target_bwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target =  (target - self.b) / self.a

		return target

	def update_parameters(self, input, target):
		# compute error
		x = self.prediction - target
		# self.error = x**2; not needed
		self.error_deriv = 2*x

		derror_dlayer1 = self.error_deriv * sigmoid(self.layer_1) * (1 - sigmoid(self.layer_1))

		# update weights:
		derror_dweights = derror_dlayer1*input*self.sf_weights
		self.weights = self.weights - self.learning_rate*derror_dweights

		derror_dweights2 = derror_dlayer1*input*input*self.sf_weights_2
		self.weights_2 = self.weights_2 - self.learning_rate*derror_dweights2

		# update bias:
		derror_dbias = derror_dlayer1*self.sf_bias
		self.bias = self.bias - self.learning_rate*derror_dbias

	def train_NN(self, iterations=100000):

		"""
		Parameters.
		-----------
		iterations : int
			Number of iterations to train the retrieval.
		"""

		if self.input_vector.ndim == 2: n_cases = self.input_vector.shape[0]

		# save mean square error over entire data set (all cases) for every
		# 100th iteration:
		MSE = list()

		for itt in range(iterations):
			# Pick training data case:

			# if itt % 10 == 0: print(itt)
			square_error = np.zeros((n_cases,))			# square error for current iteration
			for case in range(n_cases):
				# predict and update weights
				self.make_prediction(self.input_vector[case,:])	# sets attribute 'prediction' to NN
				self.update_parameters(input_vector[case,:], self.target[case])

				square_error[case] = (self.prediction - self.target[case])**2

			MSE.append(np.mean(square_error))

		return MSE

	def test_NN(self, input, target):

		"""
		Parameters:
		-----------
		input : array of floats
			Input (predictors).
		target : array of floats
			Target (predictand).
		"""

		target_target = self.translate_target_fwd(target)	# target in target space ([0, 1])

		n_cases = len(target)
		IWV_predicted = np.zeros((n_cases,))
		predicted = np.zeros((n_cases,))
		square_error_target = np.zeros((n_cases,))
		square_error_iwv = np.zeros((n_cases,))
		for case in range(n_cases):
			self.make_prediction(input[case,:])		# sets attribute 'prediction' to NN

			predicted[case] = self.prediction
			square_error_target[case] = (self.prediction - target_target[case])**2
			IWV_predicted[case] = self.translate_target_bwd(self.prediction)
			square_error_iwv[case] = (IWV_predicted[case] - target[case])**2
		
		test_dict = {	'square_error_target': square_error_target,
						'square_error_iwv': square_error_iwv,
						'prediction': predicted,
						'IWV_target': target,
						'IWV_pred': IWV_predicted}
		return test_dict

	def scatter_plot_retrieval_output(self, x_stuff, y_stuff):
		fs = 18

		fig1 = plt.figure(figsize=(22,10))
		ax0 = plt.axes()

		# Compute statistics for scatterplot:
		stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

		sc_N = stats_dict['N']
		sc_bias = stats_dict['bias']
		sc_rmse = stats_dict['rmse']
		sc_R = stats_dict['R']

			
		# ------------------------------------- #


		ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.',
					color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


		# diagonal line:
		xlims = np.asarray([0, np.nanmax(np.array([y_stuff,x_stuff]))+3])
		ylims = xlims
		ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		x_fit = x_stuff
		y_fit = y_stuff

		mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

		y_fit = y_fit[mask]
		x_fit = x_fit[mask]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			slope, offset = np.polyfit(x_fit, y_fit, 1)
			ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

		ax0.set_xlim(left=xlims[0], right=xlims[1])
		ax0.set_ylim(bottom=ylims[0], top=ylims[1])

				# add statistics:
		ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f \nLR=%s \nSF=%s"%(sc_N, 
				np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R, str(self.learning_rate),
				str(self.scaling_factor)),
				horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

		leg_handles, leg_labels = ax0.get_legend_handles_labels()
		ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

		ax0.set_aspect('equal', 'box')

		ax0.set_title("Retrieved IWV (pred) vs. observed IWV (obs)", fontsize=fs, pad=0)
		ax0.set_xlabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax0.minorticks_on()
		ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax0.tick_params(axis='both', labelsize=fs-4)

		plt.show()


class Neural_Network_v00:
	"""
		Simple Neural Network that follows the instructions and computations in
		https://realpython.com/python-ai-neural-network/.

		For initialisation, we need:
		input_vector : array of floats
			Input vector of the Neural Network training data (predictors, e.g. all MiRAC-P freq. TBs).
		target : array of floats
			Target of a prediction for the Neural Network of the training data (predictands, 
			e.g. IWV).
		learning_rate : float
			Learning rate of the Neural Network. Usually a small number (0.00001 - 0.1) to
			avoid skipping the 0 error point. Default: 0.0001
		scaling_factor : float
			Scaling factor for layer 1 to keep the sigmoid function from saturating (becoming either
			0 or 1 for all values outside a certain range). Default: 0.0001
	"""

	def __init__(self, input_vector, target, learning_rate=0.0001, scaling_factor=0.0001):
		self.input_vector = input_vector
		self.target = target
		self.learning_rate = learning_rate
		self.scaling_factor = scaling_factor

		# initialise bias and weights (for level 1):
		if self.input_vector.ndim == 1:
			self.weights = np.random.rand(self.input_vector.shape[0])
			self.weights_2 = np.random.rand(self.input_vector.shape[0])
		elif self.input_vector.ndim == 2:
			self.weights = np.random.rand(self.input_vector.shape[1])
			self.weights_2 = np.random.rand(self.input_vector.shape[1])
		self.bias = np.random.rand(1)

		# If target not in [-1, 1], translate it to those limits:
		# t_min and t_max define the range of possible predictand values
		if (np.nanmin(target) < 0) or (np.nanmax(target) > 1):
			t_min = -5		# here: min possible IWV
			t_max = 25		# here: max possible IWV in kg m^-2

			# translation coeffs; see notes, p.58
			self.a = 1 / (t_max - t_min)
			self.b = -self.a*t_min
			self.target = self.translate_target_fwd(self.target)

	def make_prediction(self, input_vector):
		self.layer_1 = self.scaling_factor*(np.dot(input_vector, self.weights) + np.dot(input_vector*input_vector, self.weights_2) + self.bias)
		self.layer_2 = sigmoid(self.layer_1)
		self.prediction = self.layer_2

	def translate_target_fwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target = self.a*target + self.b

		return target

	def translate_target_bwd(self, target):

		"""
		Parameters:
		-----------
		target : float or array of floats
			Predictand (most likely the target or prediction).
		"""

		target =  (target - self.b) / self.a

		return target

	def update_parameters(self, input, target):
		# compute error
		x = self.prediction - target
		# self.error = x**2; not needed
		self.error_deriv = 2*x

		derror_dlayer1 = self.error_deriv * sigmoid(self.layer_1) * (1 - sigmoid(self.layer_1))

		# update weights:
		derror_dweights = derror_dlayer1*self.scaling_factor*input
		self.weights = self.weights - self.learning_rate*derror_dweights

		derror_dweights2 = derror_dlayer1*self.scaling_factor*input*input
		self.weights_2 = self.weights_2 - self.learning_rate*derror_dweights2

		# update bias:
		derror_dbias = derror_dlayer1*self.scaling_factor
		self.bias = self.bias - self.learning_rate*derror_dbias

	def train_NN(self, iterations=100000):

		"""
		Parameters.
		-----------
		iterations : int
			Number of iterations to train the retrieval.
		"""

		if self.input_vector.ndim == 2: n_cases = self.input_vector.shape[0]

		# save mean square error over entire data set (all cases) for every
		# 100th iteration:
		MSE = list()

		for itt in range(iterations):
			# Pick training data case:

			# if itt % 10 == 0: print(itt)
			square_error = np.zeros((n_cases,))			# square error for current iteration
			for case in range(n_cases):
				# predict and update weights
				self.make_prediction(self.input_vector[case,:])	# sets attribute 'prediction' to NN
				self.update_parameters(input_vector[case,:], self.target[case])

				square_error[case] = (self.prediction - self.target[case])**2

			MSE.append(np.mean(square_error))

		return MSE

	def test_NN(self, input, target):

		"""
		Parameters:
		-----------
		input : array of floats
			Input (predictors).
		target : array of floats
			Target (predictand).
		"""

		target_target = self.translate_target_fwd(target)	# target in target space ([0, 1])

		n_cases = len(target)
		IWV_predicted = np.zeros((n_cases,))
		predicted = np.zeros((n_cases,))
		square_error_target = np.zeros((n_cases,))
		square_error_iwv = np.zeros((n_cases,))
		for case in range(n_cases):
			self.make_prediction(input[case,:])		# sets attribute 'prediction' to NN

			predicted[case] = self.prediction
			square_error_target[case] = (self.prediction - target_target[case])**2
			IWV_predicted[case] = self.translate_target_bwd(self.prediction)
			square_error_iwv[case] = (IWV_predicted[case] - target[case])**2
		
		test_dict = {	'square_error_target': square_error_target,
						'square_error_iwv': square_error_iwv,
						'prediction': predicted,
						'IWV_target': target,
						'IWV_pred': IWV_predicted}
		return test_dict

	def scatter_plot_retrieval_output(self, x_stuff, y_stuff):
		fs = 18

		fig1 = plt.figure(figsize=(22,10))
		ax0 = plt.axes()

		# Compute statistics for scatterplot:
		stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

		sc_N = stats_dict['N']
		sc_bias = stats_dict['bias']
		sc_rmse = stats_dict['rmse']
		sc_R = stats_dict['R']

			
		# ------------------------------------- #


		ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.',
					color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


		# diagonal line:
		xlims = np.asarray([0, np.nanmax(np.array([y_stuff,x_stuff]))+3])
		ylims = xlims
		ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


		# generate a linear fit with least squares approach: notes, p.2:
		# filter nan values:
		x_fit = x_stuff
		y_fit = y_stuff

		mask = np.isfinite(x_fit + y_fit)		# check for nans and inf.

		y_fit = y_fit[mask]
		x_fit = x_fit[mask]

		# there must be at least 2 measurements to create a linear fit:
		if (len(y_fit) > 1) and (len(x_fit) > 1):
			slope, offset = np.polyfit(x_fit, y_fit, 1)
			ds_fit = ax0.plot(xlims, slope*xlims + offset, color=(0.1,0.1,0.1), linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

		ax0.set_xlim(left=xlims[0], right=xlims[1])
		ax0.set_ylim(bottom=ylims[0], top=ylims[1])

				# add statistics:
		ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f \nLR=%s \nSF=%s"%(sc_N, 
				np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R, str(self.learning_rate),
				str(self.scaling_factor)),
				horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

		leg_handles, leg_labels = ax0.get_legend_handles_labels()
		ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

		ax0.set_aspect('equal', 'box')

		ax0.set_title("Retrieved IWV (pred) vs. observed IWV (obs)", fontsize=fs, pad=0)
		ax0.set_xlabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
		ax0.set_ylabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax0.minorticks_on()
		ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax0.tick_params(axis='both', labelsize=fs-4)

		plt.show()
					

###############################################################


"""
	This script aims to be a test of a Neural Network based retrieval of IWV
	from synthetic G band microwave radiometer TBs (MiRAC-P).

	This test follows the instructions given in https://realpython.com/python-ai-neural-network/.

	Ny Alesund radiosondes (and simulated TBs) will serve as training and test
	data. Years 2006 - 2013 will be training data, 2014 - 2017 will be test data.
"""

# Save some statistics of the model evaluation to a text file:
save_stats_file = "/net/blanc/awalbroe/Plots/NN_test/NN_test_output.txt"
with open(save_stats_file, 'w') as f:

	# Training and test years:
	seeed = 10			# seed for rng
	np.random.seed(seed=seeed)
	yrs = ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]
	n_yrs = len(yrs)
	n_training = round(0.667*n_yrs)
	n_test = n_yrs - n_training

	output_lines = list()
	for rerere in range(10):
		yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
		yrs_idx_training = yrs_idx_rng[:n_training]
		yrs_idx_test = yrs_idx_rng[n_training:]

		site = 'nya'
		rs_version = 'mwr_pro'

		output_0 = "Years Training: %s"%(np.asarray(yrs)[yrs_idx_training])
		output_1 = "Years Testing: %s"%(np.asarray(yrs)[yrs_idx_test])
		output_lines.append(output_0)
		output_lines.append(output_1)
		
		print(output_0)
		print(output_1)

		# First of all, load training and test data:
		path_data = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/"}
		path_data = path_data[site]

		# split training and test data:
		data_files_training = list()
		data_files_test = list()
		for yyyy in yrs_idx_training:
			data_files_training.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])
		for yyyy in yrs_idx_test:
			data_files_test.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])


		# Load TB data (predictor):
		TB_syn_training = radiometers(data_files_training, instrument='synthetic', site=site)
		TB_syn_test = radiometers(data_files_test, instrument='synthetic', site=site)

		# Load IWV data (predictand):
		IWV_training = radiosondes(data_files_training, s_version=rs_version, site=site)
		IWV_test = radiosondes(data_files_test, s_version=rs_version, site=site)

		# because there are some stupid IWV values (< -80000 kg m^-2), I replace those values:
		# also need to repair training TBs at the same spot:
		iwv_broken_training = np.argwhere(IWV_training.iwv < 0).flatten()
		iwv_broken_test = np.argwhere(IWV_test.iwv < 0).flatten()
		if iwv_broken_training.size > 0:
			IWV_training.iwv[iwv_broken_training] = np.asarray([(IWV_training.iwv[ib-1] + IWV_training.iwv[ib+1]) / 2 for ib in iwv_broken_training])
			TB_syn_training.TB[iwv_broken_training,:] = np.asarray([(TB_syn_training.TB[ib-1,:] + TB_syn_training.TB[ib+1,:]) / 2 for ib in iwv_broken_training])

		if iwv_broken_test.size > 0:
			IWV_test.iwv[iwv_broken_test] = np.asarray([(IWV_test.iwv[ib-1] + IWV_test.iwv[ib+1]) / 2 for ib in iwv_broken_test])
			TB_syn_test.TB[iwv_broken_test,:] = np.asarray([(TB_syn_test.TB[ib-1,:] + TB_syn_test.TB[ib+1,:]) / 2 for ib in iwv_broken_test])


		# # Compute Day of Year (which shall also be used in input vector):
		# TB_syn_training.DOY = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in TB_syn_training.time])
		# TB_syn_training.DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days for ttt in TB_syn_training.DOY])

		# TB_syn_test.DOY = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in TB_syn_test.time])
		# TB_syn_test.DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days for ttt in TB_syn_test.DOY])

		n_time = len(TB_syn_training.time)
		n_time_test = len(TB_syn_test.time)
		# Input vector: all TBs (G band + 243 and 340 GHz) for one radiosonde launch; 
		# target: IWV of radiosonde launch;
		# Layer 2 (last): Sigmoid activation function
		# Layer 1: dot_product(input, weight) + bias_vector

		# for a first test, select case 0:
		for seeed in range(25):
			np.random.seed(seed=seeed)
			# DOY_factor = 2*np.pi/365
			# input_vector = np.concatenate((TB_syn_training.TB, 
											# np.reshape(np.cos(DOY_factor*TB_syn_training.DOY), (n_time,1)), 
											# np.reshape(np.sin(DOY_factor*TB_syn_training.DOY), (n_time,1))), axis=1)
			input_vector = TB_syn_training.TB
			target = IWV_training.iwv
			learning_rate = 0.01
			scaling_factor = 0.0001

			# initialise NN and make prediction. Then compute error of prediction:
			NN = Neural_Network_v02(input_vector, target, learning_rate, scaling_factor)
			MSE = NN.train_NN(iterations=600)


			# The NN has been trained to work with target data in [0, 1]. Test the
			# retrieval. It will first predict in target space before target is then
			# converted to IWV.
			# input_vector_test = np.concatenate((TB_syn_test.TB, 
												# np.reshape(np.cos(DOY_factor*TB_syn_test.DOY), (n_time_test,1)),
												# np.reshape(np.sin(DOY_factor*TB_syn_test.DOY), (n_time_test,1))), axis=1)
			input_vector_test = TB_syn_test.TB
			test_dict = NN.test_NN(input_vector_test, IWV_test.iwv)
			IWV_pred = test_dict['IWV_pred']
			square_error_target = test_dict['square_error_target']
			square_error_iwv = test_dict['square_error_iwv']


			# plt.plot(MSE)
			# plt.show()

			# plt.plot(IWV_test.iwv, IWV_pred, linestyle='none', marker='.')
			# plt.show()


			# Plot IWV predicted and test:
			# NN.scatter_plot_retrieval_output(IWV_pred, IWV_test.iwv)
			stat_dict = compute_retrieval_statistics(IWV_pred, IWV_test.iwv)

			output_2 = "N: %s, Bias: %s, RMSE: %s, R: %s"%(str(stat_dict['N']), str(stat_dict['bias']), 
													str(stat_dict['rmse']), str(stat_dict['R']))
			pdb.set_trace()
			print(output_2)
			output_lines.append(output_2)

		output_3 = "\n"
		output_lines.append(output_3)

	f.writelines('\n'.join(output_lines))	# write output to txt file
