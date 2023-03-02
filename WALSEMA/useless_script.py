import numpy as np
import xarray as xr
import matplotlib as mpl
import sys
import glob
import pdb
import datetime as dt

mpl.use("TkAgg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "monospace"})

sys.path.insert(0, "/mnt/d/Studium_NIM/work/Codes/MOSAiC/")
from import_data import import_radiosondes_PS131_txt
from matplotlib.ticker import AutoMinorLocator


"""
	Script visualizing the statistics whose sonde bursted at which altitude.
	- import radiosonde data
	- visualize

	Script can be called in the following ways:
	- "python3 useless_script.py"				# plots histogram
	- "python3 useless_script.py 'histogram'"	# plots histogram
	- "python3 useless_script.py 'boxplot'"		# plots boxplot
"""


# paths:
path_data = "/mnt/d/Studium_NIM/work/Data/WALSEMA/radiosondes/"
path_plots = "/mnt/d/Studium_NIM/work/Plots/WALSEMA/radiosondes/useless_stats/"


# deal with command line parameters:
if len(sys.argv) == 1 or sys.argv[1] == 'histogram':
	hihi = True
	boplo = False
elif sys.argv[1] == 'boxplot':
	hihi = False
	boplo = True
else:
	hihi, boplo = False, False


# settings:
set_dict = {'save_figures': True,
			'all_in_one': False,
			'histogram': hihi,
			'boxplot': boplo}


# import data:
files = sorted(glob.glob(path_data + "*.txt"))
sonde_dict = import_radiosondes_PS131_txt(files, add_info=True)


# sort data into lists or arrays:
ops = np.asarray([sonde_dict[s_idx]['op'] for s_idx in sonde_dict.keys()])
max_alts = np.array([sonde_dict[s_idx]['max_alt'] for s_idx in sonde_dict.keys()])

# correct typos of operators:
for k, op in enumerate(ops): 
	if op == "Christian Rohlede":
		ops[k] = "Christian Rohleder"
	elif op == "Christin Rohleder":
		ops[k] = "Christian Rohleder"

# count number of sonde launches for each operator:
ops_unique = np.unique(ops)		# every operator listed once
n_sondes_dict = dict()
for opu in ops_unique: n_sondes_dict[opu] = 0
for k, op in enumerate(ops):
	n_sondes_dict[op] += 1


# group data: dictionary with keys: unique operators, values: array of max alts
max_alts = max_alts[np.where(~np.isnan(max_alts))[0]]
n_data = float(len(max_alts))
weights_data = np.ones((int(n_data),)) / n_data

max_alts_grouped = dict()
for op in ops_unique:
	max_alts_grouped[op] = max_alts[np.where(ops == op)[0]]
weights_grouped = [np.ones_like(max_alts_grouped[key]) / float(n_sondes_dict[key]) for key in max_alts_grouped.keys()]



# visualize:
fs = 14
fs_small = fs - 2
fs_dwarf = fs_small -2

c_ops = {	'Andreas Walbröl': 		(0.0,0.0,0.8),
			'Christian Rohleder': 	(0.2,0.9,0.05),
			'Gunnar Spreen': 		(0.4,0.4,0.4),
			'Hannah Niehaus': 		(0.95,0.9,0.0),
			'Janna Rückert':		(0.97,0.6,0.0),
			'Linda Rehder': 		(0.4,0.95,0.65),
			'Klara Köhler & Annika Morische': (0.517,0.8,0),
			'Mara Neudert': 		(0.45,0.3,0.9),
			'Mario Hoppmann':		(0.6,0.33,0.745),
			'Patrick Suter': 		(0.8,0.8,0.8),
			'Philipp Oehlke': 		(0.9,0.3,0.3)}


if set_dict['histogram']:

	f1 = plt.figure(figsize=(16,7))
	a1 = plt.axes()

	# x_lim = [np.floor(np.min(max_alts)/500.0)*500.0, np.ceil(np.max(max_alts)/500.0)*500.0]
	x_lim = [25000, 36500]


	# plotting:
	if set_dict['all_in_one']:
		le_hist = a1.hist(max_alts, bins=np.arange(x_lim[0], x_lim[1]+0.01, 500.0),
							weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))		# all data in one colour


		# add auxiliary info: number of launches per person, mean, median
		a1.text(0.02, 0.98, f"Min = {int(np.min(max_alts))}\nMax = {int(np.max(max_alts))}\nMean = {max_alts.mean():.1f}\n" +
				f"Median = {np.median(max_alts):.1f}\nN = {len(max_alts)}", fontsize=fs_dwarf, ha='left', va='top',
				transform=a1.transAxes)


		# set labels:
		a1.set_ylabel("Freq. occurrence", fontsize=fs)
	else:
		# text printed on figure:
		hist_labels = list()
		longest_name = np.asarray([len(op) for op in ops_unique]).max()
		for op in ops_unique:
			n_spaces = longest_name - len(op)	# spaces to fill to align the text nicely with mono
			N = int(n_sondes_dict[op])

			if len(str(N)) == 1: n_spaces += 1	# to account for differences in nums of digits
			s_mean = max_alts_grouped[op].mean()
			s_median = int(np.median(max_alts_grouped[op]))
			s_max = int(max_alts_grouped[op].max())
			text_temp = (f"{op}: " + " "*n_spaces + f"{N: 2}, {s_median}, " +
							f"{s_max}")
			hist_labels.append(text_temp)


		max_alts_grouped = [max_alts_grouped[key] for key in max_alts_grouped.keys()]
		c_ops_grouped = [c_ops[key] for key in c_ops.keys()]
		le_hist = a1.hist(max_alts_grouped, bins=np.arange(x_lim[0], x_lim[1]+0.01, 500.0),
							density=False, color=c_ops_grouped, ec=(0,0,0),
							label=hist_labels)


		# legends and colorbars:
		lh, ll = a1.get_legend_handles_labels()
		le = a1.legend(handles=lh, labels=ll, fontsize=fs_dwarf, loc='upper left', markerscale=1.5)
		le.set_title("OP: N, median, max\n")
		le.get_title().set_fontsize(fs_dwarf)
		le.get_title().set_fontweight("bold")
		le.get_title().set_horizontalalignment("left")


		# set labels:
		a1.set_ylabel("Count", fontsize=fs)


	# set axis limits:
	a1.set_xlim(x_lim)


	# set ticks and tick labels and parameters:
	a1.tick_params(axis='both', labelsize=fs_small)


	# grid:
	a1.minorticks_on()
	a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)


	# set labels:
	a1.set_xlabel("Max. altitude (m)", fontsize=fs)
	a1.set_title(f"Last update: {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC", fontsize=fs)


	if set_dict['save_figures']:
		if set_dict['all_in_one']:
			plotname = "WALSEMA_radiosondes_max_alt_stats_all"
		else:
			plotname = "WALSEMA_radiosondes_max_alt_stats"
		f1.savefig(path_plots + plotname + ".pdf", bbox_inches='tight')
	else:
		plt.show()
		pdb.set_trace()


if set_dict['boxplot']:

	def make_boxplot_great_again(bp, col):	# change color and set linewidth to 1.5
		plt.setp(bp['boxes'], color=col, linewidth=1.1)
		plt.setp(bp['whiskers'], color=col, linewidth=1.1)
		plt.setp(bp['caps'], color=col, linewidth=1.1)
		plt.setp(bp['medians'], color=col, linewidth=1.1)

	f1 = plt.figure(figsize=(13,8))
	a1 = plt.axes()


	# y_lim = [np.floor(np.min(max_alts)/500.0)*500.0, np.ceil(np.max(max_alts)/500.0)*500.0]
	y_lim = [25000, 36500]

	# plot:
	max_alts_grouped = [max_alts_grouped[key] for key in max_alts_grouped.keys()]
	c_ops_grouped = [c_ops[key] for key in c_ops.keys()]

	bp_alt = a1.boxplot(max_alts_grouped, sym='ko', widths=0.5)
	make_boxplot_great_again(bp_alt, col=(0,0,0))


	# set axis limits:
	a1.set_ylim(y_lim)


	# set ticks and tick labels and parameters:
	a1.tick_params(axis='both', labelsize=fs_small)
	a1.xaxis.set_ticks(range(1, len(ops_unique)+1))
	a1.xaxis.set_ticklabels([])
	for k, op in enumerate(ops_unique):
		if op == "Klara Köhler & Annika Morische":
			a1.text(k+1, y_lim[0]-0.02*(y_lim[1]-y_lim[0]), op.replace(" & ", ",\n").replace(" ", "\n"), 
					color=c_ops[op], fontsize=fs_small, fontweight='bold', ha='center', va='top')
		else:
			a1.text(k+1, y_lim[0]-0.02*(y_lim[1]-y_lim[0]), op.replace(" ", "\n"), 
					color=c_ops[op], fontsize=fs_small, fontweight='bold', ha='center', va='top')


	# grid:
	a1.yaxis.set_minor_locator(AutoMinorLocator())
	a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)
	a1.set_axisbelow(True)


	# set labels:
	a1.set_ylabel("Max. altitude (m)", fontsize=fs)
	a1.set_title(f"Last update: {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC", fontsize=fs)


	if set_dict['save_figures']:
		plotname = "WALSEMA_radiosondes_max_alt_stats_boxplot"
		f1.savefig(path_plots + plotname + ".pdf", bbox_inches='tight')
	else:
		plt.show()
		pdb.set_trace()
