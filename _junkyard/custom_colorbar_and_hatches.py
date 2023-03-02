    # plot anomalies:
    levels_1 = np.concatenate((np.arange(-8.0, -0.9, 0.5), np.arange(1.0, 8.1, 0.5)))
    levels_2 = np.arange(-20.0, 30.1, 2.0)
    levels_hat = np.arange(8.0, 9999.1, 100.0)      ### for hatches
    n_levels = len(levels_1)

    # colorbar (which will be modified so that -1.0 - +1.0 show the same colour (white):
    cmap = mpl.cm.get_cmap('RdBu_r', n_levels)
    cmap = cmap(range(n_levels))            # must be done to access the colormap values
    cmap[np.where(levels_1==-1.0)[0][0],:] = np.array([1.0,1.0,1.0,1.0])            # adapt colormap
    cmap = mpl.colors.ListedColormap(cmap)

    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=levels_1[0], vmax=levels_1[-1])
    var_plot = ERA5_DS.t2m - ERA5_clim_DS_grouped.t2m
    contourf_0 = a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, 
                            cmap=cmap, norm=norm, levels=levels_1, extend='both', 
                            transform=ccrs.PlateCarree())
    contour_05 = a1[0].contour(var_plot.longitude.values, var_plot.latitude.values, var_plot.values,
                            levels=levels_2, colors='black', linewidths=0.75, linestyles='dashed',
                            transform=ccrs.PlateCarree())
    a1[0].clabel(contour_05, levels=levels_2, inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)

    # hatched area for deviation > 10.0:
    a1[0].contourf(var_plot.longitude.values, var_plot.latitude.values, var_plot, colors='none', 
                    levels=levels_hat, hatches=['///'], transform=ccrs.PlateCarree())