import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb

file = "download.nc"
DS = xr.open_dataset(file)


# select time and region:
DS = DS.sel(time=slice("2022-03-10", "2022-04-15"), latitude=slice(82.5,74.5), longitude=slice(-10.0, 15.0))
merid_wv_flux_plot = DS['p72.162'].mean(dim='latitude')[:,1,:]
merid_heat_flux_plot = DS['p70.162'].mean(dim='latitude')[:,1,:]

f1, a1 = plt.subplots(1,1)
f1.set_size_inches((8,12))

bounds = np.arange(-120, 421, 60)
bounds_pos = np.arange(0, 421, 60)
bounds_neg = np.arange(-120, 0, 60)
norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=bounds[0], vmax=bounds[-1])
wv_flux_plot = a1.contourf(merid_wv_flux_plot.longitude.values, merid_wv_flux_plot.time.values, merid_wv_flux_plot.values, 
							cmap='seismic', norm=norm, levels=bounds)
contours_pos = a1.contour(merid_wv_flux_plot.longitude.values, merid_wv_flux_plot.time.values, merid_wv_flux_plot.values,
							levels=bounds_pos, colors=np.full(bounds_pos.shape, "#000000"))
contours_neg = a1.contour(merid_wv_flux_plot.longitude.values, merid_wv_flux_plot.time.values, merid_wv_flux_plot.values,
							levels=bounds_neg, colors=np.full(bounds_neg.shape, "#000000"), linestyles='dashed')
cb1 = f1.colorbar(wv_flux_plot, ax=a1, orientation='vertical')
cb1.set_label(f"{DS['p72.162'].units}")

x_ticks = np.arange(-10.0, 15.1, 5.0)
x_tick_labels = list()
for x_tick in x_ticks:
	if x_tick < 0:
		x_tick_labels.append(f"{int(-1*x_tick)}" + "$^{\circ}$W")
	else:
		x_tick_labels.append(f"{int(x_tick)}" + "$^{\circ}$E")
a1.set_xticks(x_ticks)
a1.set_xticklabels(x_tick_labels)

plt.show()
pdb.set_trace()