import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import pdb


"""
	Script for CFAD (contoured freq by altitude diagram).
"""


files_2019 = sorted(glob.glob("/net/aure/kebell/pangaea_nya/mrr/mrr_improtoo/2019/201903*_v02.nc"))
files_2020 = sorted(glob.glob("/net/aure/kebell/pangaea_nya/mrr/mrr_improtoo/2020/202003*_v02.nc"))

DS_2019 = xr.open_mfdataset(files_2019, concat_dim='time', combine='nested')
DS_2020 = xr.open_mfdataset(files_2020, concat_dim='time', combine='nested')
DS_all = xr.concat([DS_2019, DS_2020], dim='time')

radar_reflectivity = DS_all['Ze']

del DS_2019, DS_2020, DS_all

"""
print(radar_reflectivity.values[500000,:])
print(radar_reflectivity.values[500001,:])
"""

n_time = len(radar_reflectivity.time)
n_range = len(radar_reflectivity.range)

rr_reshaped = np.reshape(radar_reflectivity.values, (n_time*n_range))

"""
# then, the following yields the same as the print commands above:
# Therefore, the reshaped array is arranged like this:
# rr_reshaped[0] is time_idx 0, height_idx 0,
# rr_reshaped[1] is time_idx 0, height_idx 1,
# rr_reshaped[30] is time_idx 0, height_idx 30,
# rr_reshaped[31] is time_idx 1, height_idx 0,
# ...
print(rr_reshaped[500000*31+0:500000*31+31])
print(rr_reshaped[500001*31+0:500001*31+31]
"""

# 
# Dann fehlt noch das range_array (hieß x_axis bei dir) und das Filtern nach refl >= -25:
range_array = np.tile(np.arange(31), n_time)    # or also called x_axis

# filter for refl >= -25:
rr_reshaped[np.isnan(rr_reshaped)] = -99999.0
idx_okay = np.where(rr_reshaped >= -25)[0]


# otherwise, just overwrite:
rr_reshaped = rr_reshaped[idx_okay]
range_array = range_array[idx_okay]


# numpy histogram: count how often each height bin appears:
# and convert n_data_bins array so that it corresponds to the 1D height bin array:
len_height = 31
n_data_bins = np.zeros((len_height,))
bin_occurrence = np.zeros((len(range_array),))
for k in range(len_height):
	n_data_bins[k] = np.count_nonzero(range_array == k)

	# convert
	bin_occurrence[range_array == k] = n_data_bins[k]

# now compute weights to correctly norm the histogram:
weights_data = np.ones((len(range_array),)) / bin_occurrence


# relative occurrence (freq. occurrence):
f1, a1 = plt.subplots(1, figsize=[12,10])
mrr_ze = a1.hist2d(rr_reshaped, range_array, bins=(75,28), cmap=plt.cm.jet, weights=weights_data)
a1.set_xlabel("RR")
a1.set_ylabel("MRR bins")
cbar = f1.colorbar(mrr_ze[3], ax=a1)
cbar.set_label("Frequency occurrence (%)", size='large')

# change color bar tick labels:
cbar_ticks = cbar.get_ticks()
cbar_ticklabels = [f"{cbar_tick*100:.1f}" for cbar_tick in cbar_ticks]
cbar.set_ticks(cbar_ticks)	# must be set again because otherwise a warning will be printed
cbar.set_ticklabels = cbar.set_ticklabels(cbar_ticklabels)
plt.show()

pdb.set_trace()

# abs. occurrence:
# f1, a1 = plt.subplots(1, figsize=[12,10])
# mrr_ze = a1.hist2d(y_axis, x_axis, bins=(75,28), cmap=plt.cm.jet)
# a1.set_xlabel("RR")
# a1.set_ylabel("MRR bins")
# cbar = f1.colorbar(mrr_ze[3], ax=a1)
# cbar.set_label("obs", size='large')
# plt.show()