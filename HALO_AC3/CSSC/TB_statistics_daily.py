import xarray
import matplotlib.pyplot as plt
import numpy as np


def run_TB_statistics_daily(
	input_filename,
	output_filename,
):
	"""
	Parameters
	----------
	input_filename : str
		File name of the clear sky sonde correction NETCDF file with individual sondes.
	output_filename : str
		Filename of daily averaged clear sky correction NETCDF file

	This functions calculates the mean bias and linear correction functions to correct offsets in
	TB of each flight.
	It is assumed that all sondes belonging to one flight were released on the same 'date'
	"""
	TB_stat_DS = xarray.open_dataset(input_filename)

	# get all dates to be used as coordinate
	dates = xarray.DataArray(np.unique(TB_stat_DS.date), dims=['date'], name='date', )

	# mean bias per frequency
	mean_bias = (TB_stat_DS.tb_mean - TB_stat_DS.tb_sonde).where(TB_stat_DS.tb_used).mean('time')


	# bias for each day and frequency
	daily_biases = xarray.DataArray(np.nan, coords=[TB_stat_DS.frequency, dates, ])

	# number of used sondes per day and frequency
	used_sondes_count = TB_stat_DS.tb_used.groupby(TB_stat_DS.date).sum()

	# offset and slope of least squares linear fit for each day and frequency
	offsets = xarray.DataArray(np.nan, coords=[TB_stat_DS.frequency, dates, ])
	slopes = xarray.DataArray(np.nan, coords=[TB_stat_DS.frequency, dates, ])

	# Pearson product-moment correlation coefficients per day and frequency
	Rs = xarray.DataArray(np.nan, coords=[TB_stat_DS.frequency, dates, ], attrs={
		'long_name': 'Pearson product-moment correlation coefficient between observed and synthetic BT',
	})

	for date, DS_date in TB_stat_DS.groupby('date'):
		daily_bias = (DS_date.tb_mean - TB_stat_DS.tb_sonde).where(DS_date.tb_used).mean('time')
		daily_biases.loc[dict(date=date)] = daily_bias

		for frequency, DS_frequency_date in DS_date.groupby('frequency'):
			DS_frequency_date = DS_frequency_date.load() # load data in to memory in necessary for some of the operations
			x_fit = DS_frequency_date.tb_mean.values
			y_fit = DS_frequency_date.tb_sonde.values

			# use only finite values at clear sky
			mask = np.isfinite(x_fit + y_fit) & DS_frequency_date.tb_used

			if mask.sum() < 2:
				continue # keep nan values in offsets, slopes and Rs

			x_fit = x_fit[mask.values]
			y_fit = y_fit[mask.values]

			slope, offset = np.polyfit(x_fit, y_fit, 1)

			offsets.loc[dict(frequency=frequency, date=date)] = offset
			slopes.loc[dict(frequency=frequency, date=date)] = slope
			Rs.loc[dict(frequency=frequency, date=date)] = np.corrcoef(x_fit, y_fit)[0, 1]


	bias = xarray.where(used_sondes_count > 3, daily_biases, mean_bias)
	bias_dataset = xarray.Dataset({
		'mean_bias': mean_bias,
		'used_sondes_count': used_sondes_count,
		'daily_bias': daily_biases,
		'daily_offset': offsets,
		'daily_slopes': slopes,
		'daily_R': Rs,
		'bias': bias,
		'slope': xarray.where(Rs > 0.9, slopes, 1),
		'offset': xarray.where(Rs > 0.9, offsets, -bias),
	})
	bias_dataset.bias.attrs['description'] = "Subtract this number from BT measurements to apply offset correction without linear correction.\n"
	bias_dataset.bias.attrs['description'] += "    corrected_tb = tb - bias \n"
	bias_dataset.bias.attrs['description'] += "'bias' is the same as 'daily_bias' if more than 3 clear-sky sondes of that flight are available. "
	bias_dataset.bias.attrs['description'] += "Otherwise the campaign mean `mean_bias` is used"
	bias_dataset.slope.attrs['description'] = "Use 'slope' and 'offset' to apply a correction that is either linear or offset only (i.e. slope==1.).\n"
	bias_dataset.slope.attrs['description'] += "    corrected_tb = slope * tb + offset \n"
	bias_dataset.slope.attrs['description'] += "True linear correction is provided if correlation between observed and synthetic BT is hight (R > 0.9)"
	bias_dataset.offset.attrs['description'] = bias_dataset.slope.attrs['description']

	bias_dataset.to_netcdf(output_filename)


	# make plot
	fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(12, 12))
	axes = axes.flatten()
	for ax, frequency in zip(axes, bias_dataset.frequency):
		ax.set_title(frequency.values)
		bias_ds = bias_dataset.sel(frequency=frequency)
		d = TB_stat_DS.sel(frequency=frequency)
		bias_ds_date = bias_ds.sel(date=d.date)

		limits = xarray.DataArray([d.tb_sonde.min()-5, d.tb_sonde.max() + 5], dims='time')

		tbs = bias_ds.offset + bias_ds.slope*limits
		ax.plot(tbs.transpose('time', 'date'), limits, linewidth=0.5)

		ax.plot(d.where(d.tb_used).tb_sonde, d.where(d.tb_used).tb_mean, 'x')

		# plot linear corrected with o
		ax.plot(
			d.where(d.tb_used).tb_sonde,
			bias_ds_date.offset + bias_ds_date.slope * d.where(d.tb_used).tb_mean,
			'o', markerfacecolor='none', linewidth=0.5)
		# plot offset corrected with *
		ax.plot(
			d.where(d.tb_used).tb_sonde,
			d.where(d.tb_used).tb_mean - bias_ds_date.bias,
			'*', linewidth=0.5)

		ax.set_xlim(limits)
		ax.set_ylim(limits)

	fig.tight_layout()
	fig.savefig(output_filename+'.pdf', dpi=200)
	plt.show()