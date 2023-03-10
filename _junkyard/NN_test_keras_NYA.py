import pdb
import glob
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from import_data import import_mirac_BRT_RPG_daterange, import_radiosonde_daterange, import_mirac_IWV_LWP_RPG_daterange, import_mirac_MET_RPG_daterange
from my_classes import radiosondes, radiometers
from data_tools import compute_retrieval_statistics

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow


def quick_plot_prediction(prediction, target):
	
	plt.plot(prediction, label='prediction')
	plt.plot(predictand_test.iwv, color=(0,0,0), label='target')
	plt.legend()
	plt.show()


def plot_DOY_IWV(predictor_training, predictand_training, save_figures=True):

	x_stuff_total = np.concatenate((predictor_training.DOY_1, predictor_training.DOY_2), axis=1)
	y_stuff = predictand_training.iwv

	fs = 19

	fig1, ax0 = plt.subplots(1,2)
	fig1.set_size_inches(20,10)
	ax0 = ax0.flatten()


	for k in range(2):

		x_stuff = x_stuff_total[:,k]

			
		# ------------------------------------- #

		ax0[k].plot(x_stuff, y_stuff, linestyle='none', marker='.',
					color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


		# diagonal line:
		xlims = np.asarray([-1, 1])
		ylims = np.asarray([0, 35])
		ax0[k].plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")

		ax0[k].set_xlim(left=xlims[0], right=xlims[1])
		ax0[k].set_ylim(bottom=ylims[0], top=ylims[1])

		leg_handles, leg_labels = ax0[k].get_legend_handles_labels()
		ax0[k].legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

		ax0[k].set_aspect('equal', 'box')
		ax0[k].set_aspect(1.0/ax0[k].get_data_ratio(), 'box')

		ax0[k].set_title("Day of the Year (DOY) vs. IWV", fontsize=fs, pad=0.1)
		if k == 0:
			ax0[k].set_xlabel("cos(DOY)", fontsize=fs, labelpad=0.5)
		elif k == 1:
			ax0[k].set_xlabel("sin(DOY)", fontsize=fs, labelpad=0.5)
		ax0[k].set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

		ax0[k].minorticks_on()
		ax0[k].grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

		ax0[k].tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		fig1.savefig(path + "Scatter_DOY_vs_IWV_training.png", dpi=400)
	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_pres_sfc_IWV(predictor_training, predictand_training, save_figures=True):

	x_stuff = predictor_training.pres_sfc
	y_stuff = predictand_training.iwv

	fs = 19

	fig1 = plt.figure(figsize=(14.6,6.6))
	ax0 = plt.axes()

		
	# ------------------------------------- #

	ax0.plot(x_stuff, y_stuff, linestyle='none', marker='.',
				color=(0,0,0), markeredgecolor=(0,0,0), markersize=5.0, alpha=0.65)


	# diagonal line:
	xlims = np.asarray([95000, 107000])
	ylims = np.asarray([0, 35])

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	ax0.set_aspect('equal', 'box')
	ax0.set_aspect(1.0/ax0.get_data_ratio(), 'box')

	ax0.set_title("Surface pressure (pres$_{\mathrm{sfc}}$) vs. IWV", fontsize=fs, pad=0.1)
	ax0.set_xlabel("pres$_{\mathrm{sfc}}$ (Pa)", fontsize=fs, labelpad=0.5)
	ax0.set_ylabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		fig1.savefig(path + "Scatter_pres_sfc_vs_IWV_training.png", dpi=400)
	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_iwv_comparison(prediction_obs, mwr_dict, mirac_dict, sonde_dict, 
						aux_info, save_figures=True):

	fs = 19

	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(16,10)


	ax1.plot(mirac_dict['time_dt'][mirac_dict['RF'] == 0], mirac_dict['IWV'][mirac_dict['RF'] == 0], 
				linewidth=0.8, color=(0.2,0.4,0.9))
	# dummy plot:
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=(0.2,0.4,0.9), label='RPG')

	ax1.plot(mwr_dict['time_dt'], prediction_obs, linewidth=0.9, color=(0,0,0))
	# dummy plot:
	ax1.plot([np.nan, np.nan], [np.nan, np.nan], linewidth=2.0, color=(0,0,0), label='keras')

	ax1.plot(sonde_dict['launch_time_dt'], sonde_dict['iwv'], linestyle='none', marker='.', color=(1,0,0), markersize=6.0, label='sonde')

	lh0, ll0 = ax1.get_legend_handles_labels()
	ax1.legend(handles=lh0, labels=ll0, loc="upper right", fontsize=fs-6)

	# diagonal line and axis limits:
	if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
		ylims_dict = {	'leg1': np.asarray([0,10]),
						'leg2': np.asarray([0,8]),
						'leg3': np.asarray([0,20]),
						'leg4': np.asarray([0,35]),
						'leg5': np.asarray([0,25])}
		ylims = ylims_dict[aux_info['considered_period']]

	else:
		ylims = np.asarray([-0.5, 15])
	
	ax1.set_ylim(bottom=ylims[0], top=ylims[1])


	ax1.set_title("Batch_size: %i, Epochs: %i"%(aux_info['batch_size'], aux_info['epochs']), fontsize=fs, pad=0)
	ax1.set_ylabel("IWV (mm)", fontsize=fs, labelpad=1.0)
	ax1.tick_params(axis='both', labelsize=fs-5)
	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Timeseries_BS%i_EPOCHS%i"%(aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i.png"%(aux_info['mosaic_leg'])

		plotname = plotname  + "_nya"

		fig1.savefig(path + plotname + ".png", dpi=400)


	else:
		plt.show()

	# plt.close(fig1)
	plt.clf()


def plot_bias_histogram(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mwr_dict['time'] >= lt) & 
						(mwr_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(prediction_obs[hat])


	### this block is for MiRAC-P retrieved IWV done by RPG ###
	# # find temporal overlaps sonde with MiRAC-P obs:
	# # no absolute value because we want the closest mwr time after radiosonde launch!
	# launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	# hatson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) & 
						# (mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	# n_sondes = len(sonde_dict['launch_time'])

	# y_stuff = np.full((n_sondes,), np.nan)
	# for kk, hat in enumerate(hatson_idx):
		# y_stuff[kk] = np.nanmean(mirac_dict['IWV'][hat])
	###_____________________________________________________###


	# eventually separate into three IWV categories: IWV in [0,5), [5,10), [10,inf)
	mask_bot = ((sonde_dict['iwv'] >= 0) & (sonde_dict['iwv'] < 5))
	mask_mid = ((sonde_dict['iwv'] >= 5) & (sonde_dict['iwv'] < 10))
	mask_top = ((sonde_dict['iwv'] >= 10) & (sonde_dict['iwv'] < 100))
	bias_bot = y_stuff[mask_bot] - sonde_dict['iwv'][mask_bot]
	bias_mid = y_stuff[mask_mid] - sonde_dict['iwv'][mask_mid]
	bias_top = y_stuff[mask_top] - sonde_dict['iwv'][mask_top]
	bias_categorized = [bias_bot, bias_mid, bias_top]
	bias = y_stuff - sonde_dict['iwv']

	n_bias = float(len(bias))
	# weights_bias = np.ones_like(bias) / n_bias
	weights_bias_categorized = [np.ones_like(bias_bot) / n_bias, 
								np.ones_like(bias_mid) / n_bias, 
								np.ones_like(bias_top) / n_bias]


	fs = 19
	c_M = (0,0.729,0.675)

	x_lim = [-6.25, 6.25]			# bias in mm or kg m^-2

	fig1, ax1 = plt.subplots(1,1)
	fig1.set_size_inches(16,8)

	# ax1.hist(bias, bins=np.arange(-6.25, 6.26, 0.5), weights=weights_bias, color=c_M, ec=(0,0,0))
	# ax1.hist(bias_categorized, bins=np.arange(-6.25, 6.26, 0.5), density=True, 
				# color=['blue', 'white', 'red'], ec=(0,0,0), stacked=True)
	if aux_info['shall_be_stacked']:
		ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
					color=['blue', 'white', 'red'], ec=(0,0,0), stacked=True,
					label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])
	else:
		ax1.hist(bias_categorized, bins=np.arange(x_lim[0], x_lim[1]+0.01, 0.5), weights=weights_bias_categorized, 
					color=['blue', 'white', 'red'], ec=(0,0,0),
					label=['IWV in [0,5)', 'IWV in [5,10)', 'IWV in [10,inf)'])

	ax1.text(0.98, 0.96, "Min = %.2f\n Max = %.2f\n Mean = %.2f\n Median = %.2f"%(np.nanmin(bias),
				np.nanmax(bias), np.nanmean(bias), np.nanmedian(bias)), ha='right', va='top', 
				transform=ax1.transAxes, fontsize=fs-4, bbox=dict(boxstyle='round', ec=(0,0,0),
				fc=(1,1,1), alpha=0.5))

	# legend:
	leg_handles, leg_labels = ax1.get_legend_handles_labels()
	ax1.legend(handles=leg_handles, labels=leg_labels, loc='upper left', fontsize=fs,
					framealpha=1.0)

	ax1.set_ylim(bottom=0.0, top=0.75)
	ax1.set_xlim(left=x_lim[0], right=x_lim[1])

	ax1.minorticks_on()
	ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax1.set_xlabel("IWV difference ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0)
	ax1.set_ylabel("Frequency occurrence", fontsize = fs)
	ax1.tick_params(axis='both', labelsize=fs-2)

	ax1.set_title("IWV difference (retrieved - radiosonde)", fontsize=fs, pad=0.4)


	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Bias_hist_BS%i_EPOCHS%i"%(aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i"%(aux_info['mosaic_leg'])

		if not aux_info['shall_be_stacked']:
			plotname = plotname + "_notstacked"

		if aux_info['include_pres_sfc']:
			plotname = plotname + "_psfc"

		plotname = plotname + "_seed%02i"%aux_info['seeed']
		plotname = plotname  + "_nya"

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()


def scatterplot_comparison(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mwr_dict['time'] >= lt) & 
						(mwr_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	y_std = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(prediction_obs[hat])
		y_std[kk] = np.nanstd(prediction_obs[hat])


	x_stuff = sonde_dict['iwv']


	fs = 19
	c_M = (0,0.729,0.675)

	fig1 = plt.figure(figsize=(14.6,6.6))
	ax0 = plt.axes()

	# Compute statistics for scatterplot:
	stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

	sc_N = stats_dict['N']
	sc_bias = stats_dict['bias']
	sc_rmse = stats_dict['rmse']
	sc_R = stats_dict['R']


	# ------------------------------------- #

	ax0.errorbar(x_stuff, y_stuff, yerr=y_std, ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P')


	# diagonal line and axis limits:
	if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
		xlims_dict = {	'leg1': np.asarray([0,10]),
						'leg2': np.asarray([0,8]),
						'leg3': np.asarray([0,20]),
						'leg4': np.asarray([0,35]),
						'leg5': np.asarray([0,25])}
		xlims = xlims_dict[aux_info['considered_period']]

	else:
		# xlims = np.asarray([0, 10])
		xlims = np.asarray([0, 30])
	ylims = xlims
	ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	mask = np.isfinite(x_stuff + y_stuff)		# check for nans and inf.

	y_fit = y_stuff[mask]
	x_fit = x_stuff[mask]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		slope, offset = np.polyfit(x_fit, y_fit, 1)
		ds_fit = ax0.plot(xlims, slope*xlims + offset, color=c_M, linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	# add statistics:
	ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
			np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R),
			horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

	leg_handles, leg_labels = ax0.get_legend_handles_labels()
	ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

	ax0.set_aspect('equal', 'box')

	ax0.set_title("Retrieved IWV (pred) vs. observed (radiosonde) IWV (obs)", fontsize=fs, pad=0)
	ax0.set_xlabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
	ax0.set_ylabel("IWV$_{\mathrm{pred}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Scatter_BS%i_EPOCHS%i"%(aux_info['batch_size'], aux_info['epochs'])

		plotname = plotname + "_%s"%(aux_info['activation']) + "_%sto%s"%(str(aux_info['feature_range'][0]), 
																		str(aux_info['feature_range'][1]))

		if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
			plotname = plotname + "_LEG%i"%(aux_info['mosaic_leg'])

		if aux_info['include_pres_sfc']:
			plotname = plotname + "_psfc"

		plotname = plotname + "_seed%02i"%aux_info['seeed']
		plotname = plotname  + "_nya"

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()


def scatterplot_reference(mirac_dict, mwr_dict, sonde_dict, aux_info, save_figures=True):

	# find temporal overlaps sonde with MiRAC-P obs:
	# no absolute value because we want the closest mwr time after radiosonde launch!
	launch_window = 900		# duration (in sec) added to radiosonde launch time in which MWR data should be averaged			#
	hatson_idx = np.asarray([np.argwhere((mirac_dict['time'] >= lt) & 
						(mirac_dict['time'] <= lt+launch_window)).flatten() for lt in sonde_dict['launch_time']])

	n_sondes = len(sonde_dict['launch_time'])

	y_stuff = np.full((n_sondes,), np.nan)
	y_std = np.full((n_sondes,), np.nan)
	for kk, hat in enumerate(hatson_idx):
		y_stuff[kk] = np.nanmean(mirac_dict['IWV'][hat])
		y_std[kk] = np.nanstd(mirac_dict['IWV'][hat])


	x_stuff = sonde_dict['iwv']


	fs = 19
	c_M = (0,0.729,0.675)

	fig1 = plt.figure(figsize=(14.6,6.6))
	ax0 = plt.axes()

	# Compute statistics for scatterplot:
	stats_dict = compute_retrieval_statistics(x_stuff, y_stuff)

	sc_N = stats_dict['N']
	sc_bias = stats_dict['bias']
	sc_rmse = stats_dict['rmse']
	sc_R = stats_dict['R']


	# ------------------------------------- #

	ax0.errorbar(x_stuff, y_stuff, yerr=y_std, ecolor=c_M, elinewidth=1.2, capsize=3, markerfacecolor=c_M, markeredgecolor=(0,0,0),
						linestyle='none', marker='.', linewidth=0.75, capthick=1.2, label='MiRAC-P RPG')


	# diagonal line and axis limits:
	if aux_info['considered_period'] == "leg%i"%(aux_info['mosaic_leg']):
		xlims_dict = {	'leg1': np.asarray([0,10]),
						'leg2': np.asarray([0,8]),
						'leg3': np.asarray([0,20]),
						'leg4': np.asarray([0,35]),
						'leg5': np.asarray([0,25])}
		xlims = xlims_dict[aux_info['considered_period']]

	else:
		# xlims = np.asarray([0, 10])
		xlims = np.asarray([0, 30])
	ylims = xlims
	ax0.plot(xlims, ylims, linewidth=1.0, color=(0.65,0.65,0.65), label="Theoretical best fit")


	# generate a linear fit with least squares approach: notes, p.2:
	# filter nan values:
	mask = np.isfinite(x_stuff + y_stuff)		# check for nans and inf.

	y_fit = y_stuff[mask]
	x_fit = x_stuff[mask]

	# there must be at least 2 measurements to create a linear fit:
	if (len(y_fit) > 1) and (len(x_fit) > 1):
		slope, offset = np.polyfit(x_fit, y_fit, 1)
		ds_fit = ax0.plot(xlims, slope*xlims + offset, color=c_M, linewidth=0.75, label="Best fit: y = %.2fx + %.2f"%(slope,offset))

	ax0.set_xlim(left=xlims[0], right=xlims[1])
	ax0.set_ylim(bottom=ylims[0], top=ylims[1])

	# add statistics:
	ax0.text(0.99, 0.01, "N = %i \nMean = %.2f \nbias = %.2f \nrmse = %.2f \nR = %.3f"%(sc_N, 
			np.nanmean(np.concatenate((y_stuff, x_stuff), axis=0)), sc_bias, sc_rmse, sc_R),
			horizontalalignment='right', verticalalignment='bottom', transform=ax0.transAxes, fontsize=fs-6)

	leg_handles, leg_labels = ax0.get_legend_handles_labels()
	ax0.legend(handles=leg_handles, labels=leg_labels, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=fs-6)

	ax0.set_aspect('equal', 'box')

	ax0.set_title("Retrieved IWV (RPG) vs. observed (radiosonde) IWV (obs)", fontsize=fs, pad=0)
	ax0.set_xlabel("IWV$_{\mathrm{obs}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=0.5)
	ax0.set_ylabel("IWV$_{\mathrm{RPG}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs, labelpad=1.0)

	ax0.minorticks_on()
	ax0.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

	ax0.tick_params(axis='both', labelsize=fs-4)

	if save_figures:
		path = "/net/blanc/awalbroe/Plots/NN_test/keras_test/"
		plotname = "Scatter_MiRAC-P_RPG"

		plotname = plotname + "_%s-%s"%(aux_info['date_start'].replace('-',''), 
										aux_info['date_end'].replace('-',''))

		fig1.savefig(path + plotname + ".png", dpi=400)

	else:
		plt.show()

	plt.clf()
	1/0


def NN_retrieval(predictor_training, predictand_training, predictor_test,
					predictand_test, mwr_dict, mirac_dict, sonde_dict, aux_info,
					save_figures=True):

	tensorflow.random.set_seed(10)

	input_shape = predictor_training.input.shape
	model = Sequential()
	model.add(Dense(8, input_dim=input_shape[1], activation=aux_info['activation']))
	model.add(Dense(1, activation='linear'))
	# opti = keras.optimizers.Adam()			# default learning_rate = 0.001

	model.compile(loss='mse', optimizer='adam', 
					# metrics=["mse"],
					)

	# train the NN:
	history = model.fit(predictor_training.input_scaled, predictand_training.iwv, batch_size=aux_info['batch_size'],
				epochs=aux_info['epochs'], verbose=1,
				# validation_data=(predictor_test.input_scaled, predictand_test.iwv),
				# callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
				)

	print("Test loss: ", model.evaluate(predictor_test.input_scaled, predictand_test.iwv, verbose=0))

	# plt.plot(history.history['loss'], label='train')
	# plt.legend()
	# plt.show()
	# prediction = model.predict(predictor_test.input_scaled)
	# quick_plot_prediction(prediction, predictand_test.iwv)

	prediction_obs = model.predict(mwr_dict['input_scaled'])
	# prediction_obs = np.ones((len(mwr_dict['time']),))

	# plot_iwv_comparison(prediction_obs, mwr_dict, mirac_dict, sonde_dict, aux_info, save_figures=True)
	scatterplot_comparison(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True)
	plot_bias_histogram(prediction_obs, mwr_dict, sonde_dict, aux_info, save_figures=True)



###################################################################################################
###################################################################################################


"""
	In this script, Tensorflow.Keras will be tested for a retrieval of IWV from
	ground-based microwave radiometer (MWR) TB measurements of the MiRAC-P. Following
	steps are executed:
	- Importing training and test data (i.e., ERA-Interim provided by E. Orlandi);
		split into training and test data sets
	- quality control of the data (see RPG_MWR_STD_Software_Manual G5_2021.pdf p. 128);
		rescaling (to [0, 1]?)
	- define and build Neural Network model (try different settings)
	- compile model: choose loss function and optimiser
	- fit model (training): try various subsets of the entire data as training; try
		different batch sizes and learning rates; validate with test data
	- evaluate model (with test data)
	- predict unknown output from new data (application on MiRAC-P obs during MOSAiC)
"""


site = 'nya'		# options of training and test data: 'nya' for Ny Alesudn radiosondes
					# 'pol': ERA-Interim grid points north of 84.5 deg N
rs_version = 'mwr_pro'			# radiosonde type: 'mwr_pro' means that the structure is built
								# so that mwr_pro retrieval can read it


yrs = {'pol': ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011",
				"2012", "2013", "2014", "2015", "2016", "2017"],
		'nya': ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
				"2015", "2016", "2017"]}		# available years of data
yrs = yrs[site]
n_yrs = len(yrs)
n_training = round(0.94*n_yrs)			# number of training years; default: 0.94
n_test = n_yrs - n_training
aux_info = dict()


path_data = {'nya': "/net/blanc/awalbroe/Data/mir_fwd_sim/new_rt_nya/",
			'pol': "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/combined/"}
path_data = path_data[site]
path_data_obs = "/data/obs/campaigns/mosaic/mirac-p/l1/"
path_radiosondes = "/data/radiosondes/Polarstern/PS122_MOSAiC_upper_air_soundings/Level_2/"
path_mirac_level1 = "/data/obs/campaigns/mosaic/mirac-p/l1/"

aux_info['shall_be_stacked'] = False		# specifies if bias histogram shall be stacked or 
											# plotted side by side per bin
aux_info['include_pres_sfc'] = False			# specify if surface pressure is included as predictor or not
aux_info['mosaic_leg'] = 5
aux_info['considered_period'] = 'user'	# specify which period shall be plotted or computed:
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
					'leg1': ["2019-09-20", "2019-12-13"],
					'leg2': ["2019-12-13", "2020-02-24"],
					'leg3': ["2020-02-24", "2020-06-04"],
					'leg4': ["2020-06-04", "2020-08-12"],
					'leg5': ["2020-08-12", "2020-10-12"],
					'user': ["2020-01-01", "2020-07-31"]}
aux_info['date_start'] = daterange_options[aux_info['considered_period']][0]	# def: "2019-09-30"
aux_info['date_end'] = daterange_options[aux_info['considered_period']][1]		# def: "2020-10-02"



aux_info['seeed'] = 9			# seed for rng; default: 10; tests: 3,6,9,12,15,31,47,50,65,97,99
np.random.seed(seed=aux_info['seeed'])
tensorflow.random.set_seed(aux_info['seeed'])

# randomly select training and test years
yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
yrs_idx_training = sorted(yrs_idx_rng[:n_training])
yrs_idx_test = sorted(yrs_idx_rng[n_training:])

print("Years Training: %s"%(np.asarray(yrs)[yrs_idx_training]))
print("Years Testing: %s"%(np.asarray(yrs)[yrs_idx_test]))


# split training and test data:
data_files_training = list()
data_files_test = list()
for yyyy in yrs_idx_training:
	data_files_training.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])
for yyyy in yrs_idx_test:
	data_files_test.append(glob.glob(path_data + "rt_%s_*%s.nc"%(site, yrs[yyyy]))[0])


# Load radiometer TB data (independent predictor):
predictor_training = radiometers(data_files_training, instrument='synthetic', 
									include_pres_sfc=aux_info['include_pres_sfc'])
predictor_test = radiometers(data_files_test, instrument='synthetic', 
									include_pres_sfc=aux_info['include_pres_sfc'])

# Load IWV data (dependent predictand):
predictand_training = radiosondes(data_files_training, s_version=rs_version)
predictand_test = radiosondes(data_files_test, s_version=rs_version)

aux_info['n_training'] = len(predictand_training.iwv)
aux_info['n_test'] = len(predictand_test.iwv)

# Compute Day of Year in radians because the sin and cos of it shall also be used in input vector:
predictor_training.DOY = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in predictor_training.time])
predictor_training.DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days*2*np.pi/365 for ttt in predictor_training.DOY])
predictor_training.DOY_1 = np.reshape(np.cos(predictor_training.DOY), (aux_info['n_training'],1))
predictor_training.DOY_2 = np.reshape(np.sin(predictor_training.DOY), (aux_info['n_training'],1))

predictor_test.DOY = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in predictor_test.time])
predictor_test.DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days*2*np.pi/365 for ttt in predictor_test.DOY])
predictor_test.DOY_1 = np.reshape(np.cos(predictor_test.DOY), (aux_info['n_test'],1))
predictor_test.DOY_2 = np.reshape(np.sin(predictor_test.DOY), (aux_info['n_test'],1))


# Load observed MiRAC-P data:
mwr_dict = import_mirac_BRT_RPG_daterange(path_data_obs, aux_info['date_start'], aux_info['date_end'], verbose=1)
aux_info['n_obs'] = len(mwr_dict['time'])
if aux_info['include_pres_sfc']:
	mwr_dict_add = import_mirac_MET_RPG_daterange(path_data_obs, aux_info['date_start'], aux_info['date_end'], verbose=1)

	idx_time_overlap = np.full((aux_info['n_obs'],), -9999)
	mwr_dict['pres'] = np.full((aux_info['n_obs'],1), -9999.0)
	sii = 0
	for iii, mwrt in enumerate(mwr_dict['time']): 

		idi = np.where(mwr_dict_add['time'][sii:sii+1500] == mwrt)[0]
		if idi.size > 0:
			sii = idi[0] + sii
			idx_time_overlap[iii] = sii
			mwr_dict['pres'][iii,0] = mwr_dict_add['pres'][sii]

	# repair missing values:
	pres_fail_idx = np.where(mwr_dict['pres'] < 0)[0]
	assert np.all(np.diff(pres_fail_idx) > 1)		# then lin. interpolation with width=1 can be used
	for iii in pres_fail_idx:
		mwr_dict['pres'][iii,0] = 0.5*(mwr_dict['pres'][iii-1,0] + mwr_dict['pres'][iii+1,0])



# compute DOY_1 and DOY_2:
mwr_dict['time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(ttt) for ttt in mwr_dict['time']])
mwr_dict['DOY'] = np.asarray([(ttt - dt.datetime(ttt.year,1,1)).days*2*np.pi/365 for ttt in mwr_dict['time_dt']])
mwr_dict['DOY_1'] = np.reshape(np.cos(mwr_dict['DOY']), (aux_info['n_obs'],1))
mwr_dict['DOY_2'] = np.reshape(np.sin(mwr_dict['DOY']), (aux_info['n_obs'],1))

del mwr_dict['DOY']

# Load radiosonde data for comparison:
sonde_dict = import_radiosonde_daterange(path_radiosondes, aux_info['date_start'], aux_info['date_end'], s_version='level_2')
sonde_dict['launch_time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(lt) for lt in sonde_dict['launch_time']])

# Std RPG MiRAC for comparison:
mirac_dict = import_mirac_IWV_LWP_RPG_daterange(path_mirac_level1, aux_info['date_start'], 
					aux_info['date_end'], which_retrieval='iwv', minute_avg=False, verbose=1)
# mirac_dict['time_dt'] = np.asarray([dt.datetime.utcfromtimestamp(mt) for mt in mirac_dict['time']])


# Quality control of the data: See RPG Software Manual (RPG_MWR_STD_Software_Manual_G5_2021.pdf):
# # # height_dif_training = np.diff(predictand_training.height, axis=1)
# # # height_dif_test = np.diff(predictand_test.height, axis=1)
# # # pres_dif_training = np.diff(predictand_training.pres, axis=1)
# # # pres_dif_test = np.diff(predictand_test.pres, axis=1)

# check if height increases in subsequent levels, pressure decreases with height,
# temp in [190, 330], pres_sfc > 50000, pres in [1, 107000], height in [-200, 70000],
# temp and pres information at least up to 10 km; hum. information up to -30 deg C,
# n_levels >= 10
# (assert is split into many parts to more easily identify broken variables)
# Broken temp, pres, height, or humidity values might cause the computed IWV to be
# erroneour
# # # assert ((np.all(height_dif_training > 0)) and (np.all(height_dif_test > 0)) and 
		# # # (np.all(pres_dif_training < 0)) and (np.all(pres_dif_test < 0)))
# # # assert ((np.all(predictand_training.temp <= 330)) and (np.all(predictand_training.temp >= 190)) 
		# # # and (np.all(predictand_test.temp <= 330)) and (np.all(predictand_test.temp >= 190)))
# # # assert ((np.all(predictand_training.pres[:,0] > 50000)) and (np.all(predictand_test.pres[:,0] > 50000)) 
		# # # and (np.all(predictand_training.pres > 1)) and (np.all(predictand_training.pres < 107000)) 
		# # # and (np.all(predictand_test.pres > 1)) and (np.all(predictand_test.pres < 107000)))
# # # assert ((np.all(predictand_training.height[:,0] > -200)) and (np.all(predictand_training.height[:,-1] < 70000)) 
		# # # and (np.all(predictand_test.height[:,0] > -200)) and (np.all(predictand_test.height[:,-1] < 70000)))
# # # assert (predictand_training.height.shape[1] >= 10) and (predictand_test.height.shape[1] >= 10)

# on a regular grid, it's simple to check if temp and pres information exist up to 10 km height:
idx_10km_train = np.where(predictand_training.height[0,:] >= 10000)[0]
idx_10km_test = np.where(predictand_test.height[0,:] >= 10000)[0]

for k in range(aux_info['n_training']): 
	assert ((np.any(~np.isnan(predictand_training.temp[k,idx_10km_train]))) and 
			(np.any(~np.isnan(predictand_training.pres[k,idx_10km_train]))))

	# check if hum. information available up to -30 deg C:
	idx_243K = np.where(predictand_training.temp[k,:] <= 243.15)[0]
	assert np.any(~np.isnan(predictand_training.rh[k,idx_243K]))

for k in range(aux_info['n_test']): 
	assert ((np.any(~np.isnan(predictand_test.temp[k,idx_10km_test]))) and 
			(np.any(~np.isnan(predictand_test.pres[k,idx_10km_test]))))

	# check if hum. information available up to -30 deg C:
	idx_243K = np.where(predictand_test.temp[k,:] <= 243.15)[0]
	assert np.any(~np.isnan(predictand_test.rh[k,idx_243K]))

# further expand the quality control and check if IWV values are okay

# In the Ny Alesund radiosonde training data there are some stupid IWV values 
# (< -80000 kg m^-2), I replace those values:
# also need to repair training TBs at the same spot:
iwv_broken_training = np.argwhere(predictand_training.iwv < 0).flatten()
iwv_broken_test = np.argwhere(predictand_test.iwv < 0).flatten()
if iwv_broken_training.size > 0:
	predictand_training.iwv[iwv_broken_training] = np.asarray([(predictand_training.iwv[ib-1] + 
													predictand_training.iwv[ib+1]) / 2 for ib in iwv_broken_training])
	predictor_training.TB[iwv_broken_training,:] = np.asarray([(predictor_training.TB[ib-1,:] + 
													predictor_training.TB[ib+1,:]) / 2 for ib in iwv_broken_training])

if iwv_broken_test.size > 0:
	predictand_test.iwv[iwv_broken_test] = np.asarray([(predictand_test.iwv[ib-1] + predictand_test.iwv[ib+1]) / 2 for ib in iwv_broken_test])
	predictor_test.TB[iwv_broken_test,:] = np.asarray([(predictor_test.TB[ib-1,:] + predictor_test.TB[ib+1,:]) / 2 for ib in iwv_broken_test])


"""
	Define and build Neural Network model: Start with Multilayer Perceptron Model (MLP)
	which has got fully connected (Dense) layers only. Input_shape depends on whether or not
	DOY and surface pressure are included (and another bias term might also be included).
	One hidden layer with activation fct. tanh (try relu, sigmoid or others later), n_neurons =
	8 (or 9 depending on inclusion of yet another bias term). Last layer (output layer) has got
	1 node (IWV), linear (or tanh?) activation function. Eventually, the inclusion of the second
	bias term from hidden layer to output layer requires the use of the Functional API of Keras.

	Eventually more complex architectures might be tested (Convolutional Neural Networks (CNN))
	to better capture nonlinearities 
	Example: https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm

	Loss function: MSE, optimiser: adam (both options might also be changed during testing and
	build phase)

	Fit model: try different training data combinations (TB only, TB + bias, TB + DOY, 
	TB + DOY + pres_sfc, TB + bias + DOY + pres_sfc). try different batch sizes (1, 2**2 - 2**9,
	n_training) and learning rates (lr; lower values for small batch sizes)
	Eventually add validation_dataset=(predictor_test, predictand_test).
	Avoid overfitting by applying Early Stop: callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
"""

# choose activation function from input to hidden layer:
aux_info['activation'] = 'exponential'
predictor_training.input = np.concatenate((predictor_training.TB, 
											predictor_training.DOY_1,
											predictor_training.DOY_2), axis=1)
predictor_test.input = np.concatenate((predictor_test.TB, 
											predictor_test.DOY_1,
											predictor_test.DOY_2), axis=1)
mwr_dict['input'] = np.concatenate((mwr_dict['TBs'], mwr_dict['DOY_1'], mwr_dict['DOY_2']),
									axis=1)


# feature ranges can be defined:
# fr_ranges = [(-5.0,0.0), (-2.0,0.0)]
fr_ranges = [(-3.0,0.0), (-3.0,1.0)]

for fr in fr_ranges:

	fr_activation = {	'tanh': (fr[0],fr[1]),
						'exponential': (fr[0],fr[1])
						}

	aux_info['feature_range'] = fr_activation[aux_info['activation']]
	print(aux_info['feature_range'])

	# Rescale input: For the MinMaxScaler
	scaler = MinMaxScaler(feature_range=aux_info['feature_range']).fit(predictor_training.input)
	predictor_training.input_scaled = scaler.transform(predictor_training.input)
	predictor_test.input_scaled = scaler.transform(predictor_test.input)

	# Rescale obs:
	mwr_dict['input_scaled'] = scaler.transform(mwr_dict['input'])


	# # # Plot relations of input to IWV:
	# # plot_DOY_IWV(predictor_training, predictand_training, save_figures=True)
	# # plot_pres_sfc_IWV(predictor_training, predictand_training, save_figures=True)

	if fr == (-3.0,0.0):
		BS = [1,2]
	elif fr == (-3.0,1.0):
		BS = [1,4]

	# if fr == (-5.0,0.0):
		# BS = [4]
	# elif fr == (-2.0,0.0):
		# BS = [8]

	# for aux_info['batch_size'] in [8,32,64]:
	for aux_info['batch_size'] in BS:
		print("batch_size=", aux_info['batch_size'])

		if (aux_info['batch_size'] == 1) and (fr == (-3.0,0.0)):
			epochs = [16]
		elif (aux_info['batch_size'] == 2) and (fr == (-3.0,0.0)):
			epochs = [18]
		elif (aux_info['batch_size'] == 1) and (fr == (-3.0,1.0)):
			epochs = [16]
		elif (aux_info['batch_size'] == 4) and (fr == (-3.0,1.0)):
			epochs = [16]
		elif (aux_info['batch_size'] == 4) and (fr == (-5.0,0.0)):
			epochs = [16]
		elif (aux_info['batch_size'] == 8) and (fr == (-2.0,0.0)):
			epochs = [25]


		for aux_info['epochs'] in epochs:
			print("epochs=", aux_info['epochs'])

			NN_retrieval(predictor_training, predictand_training, predictor_test, 
							predictand_test, mwr_dict, mirac_dict, sonde_dict, aux_info,
							save_figures=True)

			plt.close()


print("Done....")
