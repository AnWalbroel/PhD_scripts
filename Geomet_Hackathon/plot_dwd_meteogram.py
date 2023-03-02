import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

import datetime as dt
import glob
import pdb


def numpydatetime64_to_epochtime(npdt_array):

	"""
	Converts numpy datetime64 array to array in seconds since 1970-01-01 00:00:00 UTC (type:
	float).

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	return sec_epochtime


def numpydatetime64_to_datetime(npdt_array):

	"""
	Converts numpy datetime64 array to a datetime object array.

	Parameters:
	-----------
	npdt_array : numpy array of type np.datetime64 or np.datetime64 type
		Array (1D) or directly a np.datetime64 type variable.
	"""

	sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

	# sec_epochtime can be an array or just a float
	if sec_epochtime.ndim > 0:
		time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in sec_epochtime])

	else:
		time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)

	return time_dt


def create_artificial_cloudnet(DATA_DS):

	"""
	Create artificial cloudnet from specific cloud water content, specific cloud ice content,
	mixing ratios of rain, snow, graupel, and hail.

	Parameters:
	-----------
	DATA_DS : xarray dataset
		Input data array containing information about cloud properties. 
	"""

	# Classification
	# class_clear = Clear sky
	# class_cldrop = Cloud droplets only
	# class_dr = Drizzle or rain
	# class_dr_cldrop = Drizzle/rain & cloud droplets
	# class_isg = Ice & Snow & Graupel
	# class_ildrop = Ice & liquid droplets

	thold = 1.e-8
	classes = DATA_DS.QC.to_dataset(name="clear")
	classes["clear"] = (DATA_DS.QC < thold) & (DATA_DS.QR < thold) & (DATA_DS.QI < thold)
	classes["cldrop"] = DATA_DS.QC > thold
	classes["dr"] = (DATA_DS.QC < thold) & (DATA_DS.QR > thold)
	classes["dr_cldrop"] = (DATA_DS.QC > thold) & (DATA_DS.QR > thold)
	classes["isg"] = (((DATA_DS.QC < thold) | (DATA_DS.QR < thold))
						& ((DATA_DS.QI > thold) | (DATA_DS.QH > thold)
						| (DATA_DS.QG > thold) | (DATA_DS.QS > thold)))
	classes["ildrop"] = (((DATA_DS.QC > thold) | (DATA_DS.QR > thold))
						& ((DATA_DS.QI > thold) | (DATA_DS.QH > thold)
						| (DATA_DS.QG > thold) | (DATA_DS.QS > thold)))

	# Each class will get a number ranging from 2-7 (avoid 0 and 1 as they can be interpreted as boolean).
	classes["nclass"] = classes.clear.where(classes.clear == False, 2)
	classes["nclass"] = classes.nclass.where(classes.cldrop == False, 3)
	classes["nclass"] = classes.nclass.where(classes.dr == False, 4)
	classes["nclass"] = classes.nclass.where(classes.dr_cldrop == False, 5)
	classes["nclass"] = classes.nclass.where(classes.isg == False, 6)
	classes["nclass"] = classes.nclass.where(classes.ildrop == False, 7)

	return classes


###################################################################################################
###################################################################################################


"""
	This script aims to imitate the typical DWD meteograms found on flugwetter.de.
"""


# find files:
path_data = "/net/blanc/awalbroe/Data/Geomet_Hackathon/"
plot_path = "/net/blanc/awalbroe/Plots/Geomet_Hackathon/"
file = sorted(glob.glob(path_data + "*.nc"))[0]

wind_10m_labels = True				# if True: surface wind (10 m) will be labeled
precip_bar_labels = True			# if True: precipitation bar plot will have labels on top
manual_position_T_contour = False	# if True: attempted to manually position the temperature contour plot labels
save_figures = True					# if True: created figure(s) will be saved to plot_path


# Import data and remove useless variables (and dimensions):
DATA_DS = xr.open_dataset(file)

# select only useful variables:
useful_vars = ['time_step', 'time', 'date', 'height_2', 'P', 'T', 'U', 'V', 'QV', 'QC', 'QI', 'QR',
					'REL_HUM', 'CLC', 'QV_S', 'T_S', 'H_SNOW', 'P_SFC', 'T2M', 'QS', 'QG',
					'TD2M', 'U10M', 'V10M', 'VBMAX10M', 'QH',
					'RAIN_GSP', 'SNOW_GSP', 'RAIN_CON', 'SNOW_CON', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 
					'TQV', 'TQC', 'TQI', 'TQR', 'TQS']
DATA_DS = DATA_DS[useful_vars]


# create an array of datetime time stamps:
time_dt = numpydatetime64_to_datetime(DATA_DS.time.values)
n_time = len(time_dt)
n_height = len(DATA_DS.height_2)


# Convert units:
DATA_DS['U10M'] = DATA_DS.U10M*1.94384				# from m s^-1 to knots
DATA_DS['V10M'] = DATA_DS.V10M*1.94384				# from m s^-1 to knots
DATA_DS['VBMAX10M'] = DATA_DS.VBMAX10M*1.94384		# from m s^-1 to knots
DATA_DS['T2M'] = DATA_DS.T2M - 273.15				# from K to deg C
DATA_DS['TD2M'] = DATA_DS.TD2M - 273.15				# from K to deg C
DATA_DS['P_SFC'] = DATA_DS.P_SFC*0.01				# from Pa to hPa
DATA_DS['H_SNOW'] = DATA_DS.H_SNOW*100				# from m to cm
DATA_DS['P'] = DATA_DS.P*0.01						# from Pa to hPa
DATA_DS['T'] = DATA_DS.T - 273.15					# from K to deg C


# some auxiliary quantities:
DATA_DS['RAIN_GSP'] = DATA_DS.RAIN_GSP + DATA_DS.RAIN_CON			# combine rain with convective rain (parametrized)
T_850 = np.asarray([np.interp(850, DATA_DS.P.values[k,:], DATA_DS.T.values[k,:]) 
					for k in range(n_time)])

# min-max ranges of some quantities:
p_sfc_range = DATA_DS.P_SFC.values.max() - DATA_DS.P_SFC.values.min()
iwv_range = DATA_DS.TQV.values.max() - DATA_DS.TQV.values.min()
snow_range = DATA_DS.H_SNOW.values.max() - DATA_DS.H_SNOW.values.min()
T_curtain_range = np.arange(-80,61,5)				# range for the temperature (time x height) plot


# compute hourly rain:
HOURLY_RAIN = DATA_DS.RAIN_GSP.resample(time="1H").nearest().diff("time")


# 3-hourly wind: U_ip and V_ip are u and v wind components on a regular pressure grid, frequency
# of 3 hours. Those arrays must be flattened so that pyplot's barb function can be used. Time
# and pressure grid must be repeated accordingly.
WIND_3H = DATA_DS[['U10M', 'V10M', 'U', 'V', 'P', 'VBMAX10M']].resample(time="3H").nearest()
P_ip = np.arange(350,1051,50)
U_ip = np.asarray([np.interp(P_ip, WIND_3H.P.values[k,:], WIND_3H.U.values[k,:],
					left=np.nan, right=np.nan) for k in range(len(WIND_3H.time))]).flatten()
V_ip = np.asarray([np.interp(P_ip, WIND_3H.P.values[k,:], WIND_3H.V.values[k,:],
					left=np.nan, right=np.nan) for k in range(len(WIND_3H.time))]).flatten()
time_ip = np.repeat(numpydatetime64_to_datetime(WIND_3H.time.values), len(P_ip))
P_ip_repeat = np.tile(P_ip, 9)


# Temperature contour plot colormap and eventual manual positioning of the contour plot 
# labels for the temperature (time x height) plot:
T_curtain_c = np.full(T_curtain_range.shape, '#ff0000')
T_curtain_c[T_curtain_range == 0.0] = '#00ff00'
T_curtain_c[T_curtain_range < 0.0] = '#0000ff'
if manual_position_T_contour:
	x_posi_idx = np.argmin(abs(time_dt - (time_dt[0] + dt.timedelta(hours=1))))
	label_position = np.full((T_curtain_range.shape[0],2), (0.0,0.0))
	reftime = dt.datetime(1970,1,1)
	for k, Tc in enumerate(T_curtain_range):
		# x location will be 1 hours after the first time step; y location depends on the
		# pressure for that given time:
		y_posi_idx = np.argmin(np.abs(Tc - DATA_DS.T.values[x_posi_idx,:]))

		if np.abs(DATA_DS.T.values[x_posi_idx, y_posi_idx] - Tc) < 1:
			label_position[k] = ((time_dt[x_posi_idx] - reftime).total_seconds()/86400, 
									DATA_DS.P.values[x_posi_idx, y_posi_idx])


# cloud classification for the artificial cloudnet; then define the colormap for the
# cloudnet plot:
classes = create_artificial_cloudnet(DATA_DS)
cloud_range = np.arange(2,9)
# cloud_c = np.array(['#99cdff', '#e8e8e8', '#bbbbbb', '#a6a6a6', '#919191', '#7a7a7a', '#646464'])	# blue-> gray-scale
cloud_c = ['#ffffff', '#16ddff', '#990000', '#005fa5', '#e9e900', '#00bf00']
cloud_cmap = matplotlib.colors.ListedColormap(cloud_c)


# Define the x ticks:
# for major ticks, find first 12 UTC; for thicker grid lines, find first 00 UTC:
f_12 = np.asarray([t_12 for t_12 in time_dt if t_12.time() == dt.time(12,0)])[0]
f_00 = np.asarray([t_00 for t_00 in time_dt if t_00.time() == dt.time(0,0)])[0]

x_grid_bold_dt = mdates.drange(f_00, time_dt[-1]+dt.timedelta(hours=1), dt.timedelta(hours=24)).flatten()

x_ticks_minor_dt = mdates.drange(time_dt[0], time_dt[-1]+dt.timedelta(hours=1), dt.timedelta(hours=6))
x_ticks_major_dt = mdates.drange(f_12, time_dt[-1]+dt.timedelta(hours=1), dt.timedelta(hours=24)).flatten()			# must be changed to first 12 UTC
dt_fmt_minor = mdates.DateFormatter("%H UTC")
dt_fmt_major = mdates.DateFormatter("%H UTC\n%a %d %b")


# some axis properties are defined here (and set later): y limits, basic font size, grid color
fs = 12							# fontsize
fs_small = fs - 3				# smaller font size
c_grid = (0.63,0.63,0.63)		# color of grid lines
y_lim_time_height = [1050, 140]
y_lim_wind = [-9.99, 85]
y_lim_T = [np.hstack((DATA_DS.T2M.values, DATA_DS.TD2M.values, T_850)).min()-2, 
			np.hstack((DATA_DS.T2M.values, DATA_DS.TD2M.values, T_850)).max()+2]
y_lim_P = [DATA_DS.P_SFC.values.min()-0.05*p_sfc_range, 
			DATA_DS.P_SFC.values.max()+0.05*p_sfc_range]
y_lim_IWV = [DATA_DS.TQV.values.min()-0.05*iwv_range,
			DATA_DS.TQV.values.max()+0.05*iwv_range]
y_lim_precip = [0, np.ceil(1.2*DATA_DS.RAIN_GSP.values.max())]
if (DATA_DS.H_SNOW.values.min() == 0) and (snow_range > 0):
	y_lim_snow = [0, np.ceil(1.2*DATA_DS.H_SNOW.values.max())]
elif snow_range == 0:
	y_lim_snow = [0,5]
else:
	y_lim_snow = [DATA_DS.H_SNOW.values.min()-0.05*snow_range,
					DATA_DS.H_SNOW.values.max()+0.05*snow_range]


# Create figure and axes; update global style parameters:
matplotlib.rcParams.update({'xtick.labelsize': fs_small, 'ytick.labelsize': fs-4, 'axes.grid.which': 'both', 
							'axes.grid': True, 'axes.grid.axis': 'x', 'grid.color': c_grid,
							'axes.labelsize': fs_small})

fig1 = plt.figure(figsize=(16,12), linewidth=1.3, edgecolor=(0,0,0))			# aspect ratio: 4:3
ax_time_height = plt.subplot2grid((15,1), (0,0), rowspan=4)			# time height variables
ax_wind = plt.subplot2grid((15,1), (4,0), rowspan=2)				# surface winds
ax_T = plt.subplot2grid((15,1), (6,0), rowspan=2)					# surface T, T_dp and 850 hPa T
ax_P = plt.subplot2grid((15,1), (8,0), rowspan=2)					# surface P, IWV
ax_IWV = ax_P.twinx()												# IWV
ax_precip = plt.subplot2grid((15,1), (10,0), rowspan=2)				# surface precip (snow, rain)
ax_snow = ax_precip.twinx()											# snow
ax_time = plt.subplot2grid((15,1), (12,0), rowspan=1)				# time axis
ax_descr = plt.subplot2grid((15,1), (14,0), rowspan=1)				# meta data


# x axis grid: additional bold lines
for x_dt_00 in x_grid_bold_dt:
	ax_time_height.plot([x_dt_00, x_dt_00], y_lim_time_height, color=c_grid, linewidth=1.6)
	ax_wind.plot([x_dt_00, x_dt_00], y_lim_wind, color=c_grid, linewidth=1.6)
	ax_T.plot([x_dt_00, x_dt_00], y_lim_T, color=c_grid, linewidth=1.6)
	ax_P.plot([x_dt_00, x_dt_00], y_lim_P, color=c_grid, linewidth=1.6)
	ax_precip.plot([x_dt_00, x_dt_00], y_lim_precip, color=c_grid, linewidth=1.6)
	ax_time.plot([x_dt_00, x_dt_00], [0,1], color=c_grid, linewidth=1.6)


# Data plots: Time height plot
T_curtain = ax_time_height.contour(np.repeat(np.reshape(time_dt, (n_time,1)), n_height, axis=1),
						DATA_DS.P.values, DATA_DS.T.values, levels=T_curtain_range, colors=T_curtain_c,
						linewidths=1.1)
if manual_position_T_contour:
	ax_time_height.clabel(T_curtain, inline=True, fmt="%i", inline_spacing=18, fontsize=fs-2, manual=label_position)
else:
	ax_time_height.clabel(T_curtain, inline=True, fmt="%i", inline_spacing=18, fontsize=fs-2)
cloudnet_plot = ax_time_height.pcolormesh(np.repeat(np.reshape(time_dt, (n_time,1)), n_height, axis=1),
						DATA_DS.P.values, classes.nclass.values, vmin=cloud_range[0], vmax=cloud_range[-1],
						cmap=cloud_cmap)
cb1 = fig1.colorbar(mappable=cloudnet_plot, ax=ax_time_height, use_gridspec=False, 
						orientation='vertical', shrink=0.9, anchor=(0.60,0.5), format="%i",
						ticks=cloud_range[:-1]+0.5)
cb1.ax.set_yticklabels(["Clear", "Cloud droplet\nonly", "Drizzle or\nrain", "Drizzle/rain &\ncloud droplets",
						"Ice & Snow &\nGraupel", "Ice & liquid\ndroplets"])
cb1.ax.tick_params(labelsize=fs-4)		# change size of colorbar ticks

ax_time_height.barbs(time_ip, P_ip_repeat, U_ip, V_ip,
						length=4.5, pivot='middle', barbcolor=(0,0,0), rounding=True,
						zorder=100) # zorder=100 ensures that the barbs are on top of everything else

# flight level labels:
FL_labels = ["FL 450", "FL 390", "FL 340", "FL 300", "FL 240", "FL 180", "FL 140", "FL 100", "FL 080", "FL 050"]
P_labels = np.array([150, 200, 250, 300, 400, 500, 600, 700, 750, 850])
t_range_whole = (time_dt[-1] + dt.timedelta(hours=2)) - (time_dt[0] - dt.timedelta(hours=2))
FL_x = time_dt[0] - dt.timedelta(hours=2) - (1/20)*t_range_whole
for (lab, p_lab) in zip(FL_labels, P_labels):
	ax_time_height.text(FL_x, p_lab, lab, ha='right', va='center', fontsize=fs-4, 
						transform=ax_time_height.transData)


# Data plots: Surface wind plot
ax_wind.plot(time_dt, np.sqrt((DATA_DS.U10M.values)**2 + (DATA_DS.V10M.values)**2), 
						linewidth=1.3, color=(0,0,0), label="Wind 10$\,$m (kt)")
ax_wind.plot(time_dt, DATA_DS.VBMAX10M.values, linewidth=1.3, color=(1,0,0), 
						label="Gust 10$\,$m (kt)")
ax_wind.barbs(numpydatetime64_to_datetime(WIND_3H.time.values), 
						0.775*(y_lim_wind[1] - y_lim_wind[0]), WIND_3H.U10M.values, WIND_3H.V10M.values,
						length=5, pivot='middle', barbcolor=(0,0,0), rounding=True, 
						zorder=100) # zorder=100 ensures that the barbs are on top of everything else
ax_wind.text(-0.29, 0.75, "Wind barbs: Wind 10$\,$m", ha='left', transform=ax_wind.transAxes, fontsize=fs_small)
# add labels / annotations:
n_wind3h = len(WIND_3H.time.values)
for kkk in range(n_wind3h):
	wind10m = np.sqrt((WIND_3H.U10M.values[kkk])**2 + (WIND_3H.V10M.values[kkk])**2)
	ax_wind.annotate(f"{int(round(wind10m))}",
						xy=(numpydatetime64_to_datetime(WIND_3H.time.values[kkk]), wind10m - 0.025*(y_lim_wind[1] - y_lim_wind[0])), fontsize=fs-6,
						horizontalalignment='center', verticalalignment='top', color=(0,0,0))
	ax_wind.annotate(f"{int(round(WIND_3H.VBMAX10M.values[kkk]))}",
						xy=(numpydatetime64_to_datetime(WIND_3H.time.values[kkk]), WIND_3H.VBMAX10M.values[kkk] + 0.025*(y_lim_wind[1] - y_lim_wind[0])), 
						fontsize=fs-6, horizontalalignment='center', verticalalignment='bottom', color=(1,0,0))


# Data plots: Temperature plot (at 2 m and 850 hPa)
ax_T.fill_between(time_dt, DATA_DS.T2M.values, DATA_DS.TD2M.values, facecolor=(1,1,166/255),
						label="T - T$_{\mathrm{d}}$ ($^{\circ}$C)")
ax_T.plot(time_dt, DATA_DS.T2M.values, linewidth=1.3, color=(0,0,0), 
						label="Temperature")
ax_T.plot(time_dt, DATA_DS.TD2M.values, linewidth=1.3, color=(20/255,208/255,16/255),
						label="Dewpoint")
ax_T.plot(time_dt, T_850, linewidth=1.3, color=(1,0,0), label="T 850$\,$hPa ($^{\circ}$C)")


# Data plots: Surface pressure and IWV plot
ax_P.plot(time_dt, DATA_DS.P_SFC.values, linewidth=1.3, color=(0,0,0), label="Pressure MSL (hPa)")
ax_IWV.plot(time_dt, DATA_DS.TQV.values, linewidth=1.3, color=(100/255,134/255,227/255), 
						label="Integrated Water Vapor\n(mm)")


# Precipitation plot: Snow and rain
ax_precip.plot(time_dt, DATA_DS.RAIN_GSP.values, linewidth=1.3, color=(0,1,102/255), label="Accum. rain (mm)")

# bar plot shifted by one hour to display the bar like covering the hour when it rains: 
# e.g. 13-14 UTC might be covered by bar, displaying that it rained 6 mm in from 13 to 14 UTC
precip_bar = ax_precip.bar(numpydatetime64_to_datetime(HOURLY_RAIN.time.values)-dt.timedelta(hours=1), 
						HOURLY_RAIN.values, width=1/24,		# width=1 hour
						align='edge', edgecolor=(0,0,0), linewidth=0.8, color=(0,1,102/255,0.30))
if precip_bar_labels:
	# ax_precip.bar_label(precip_bar, fontsize=fs-6, fmt="%.1f", label_type='edge')		# requires MATPLOTLIB V3.4 or later
	n_precip = len(HOURLY_RAIN.time.values)
	for kkk in range(n_precip):
		ax_precip.annotate(f"{HOURLY_RAIN.values[kkk]:.1f}", 
							xy=(numpydatetime64_to_datetime(HOURLY_RAIN.time.values[kkk])-dt.timedelta(minutes=30), 
								HOURLY_RAIN.values[kkk]), fontsize=fs-6,
							horizontalalignment='center', verticalalignment='bottom') 
ax_precip.text(time_dt[-1], DATA_DS.RAIN_GSP.values[-1], f"{DATA_DS.RAIN_GSP.values[-1]:.2f}", fontsize=fs_small,
						fontweight='bold', ha='left', va='bottom', color=(0,1,102/255))
ax_snow.plot(time_dt, DATA_DS.H_SNOW.values, linewidth=1.3, color=(11/255,11/255,1), label="Snow depth (cm)")
ax_snow.text(time_dt[-1], DATA_DS.H_SNOW.values[-1], f"{DATA_DS.H_SNOW.values[-1]:.1f}", fontsize=fs_small,
						fontweight='bold', ha='left', va='bottom', color=(11/255,11/255,1))


# Legends:
lh, ll = ax_wind.get_legend_handles_labels()
ax_wind.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,0.7), fontsize=fs_small,
				frameon=False)
lh, ll = ax_T.get_legend_handles_labels()
ax_T.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,1.0), fontsize=fs_small,
				frameon=False)
lh, ll = ax_P.get_legend_handles_labels()
ax_P.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,1.0), fontsize=fs_small,
				frameon=False)
lh, ll = ax_IWV.get_legend_handles_labels()
ax_IWV.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,0.8), fontsize=fs_small,
				frameon=False)
lh, ll = ax_precip.get_legend_handles_labels()
ax_precip.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,1.0), fontsize=fs_small,
				frameon=False)
lh, ll = ax_snow.get_legend_handles_labels()
ax_snow.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(-0.30,0.8), fontsize=fs_small,
				frameon=False)


# Simulation meta data in the description axis: Station data to dictionary:
station = DATA_DS.station.split("\n")
station_dict = dict()
for st in station:
	st_split = st.split("=")
	station_dict[st_split[0]] = st_split[1]
ax_descr.text(0.01, 0.9, f"{DATA_DS.title}, {station_dict['station_name'].capitalize()}, LAT/LON: " +
				f"N {station_dict['station_lat']} E {station_dict['station_lon']}\n" +						
				f"{dt.datetime.strftime(time_dt[0], '%Y-%m-%d %H UTC')}, station height (asl): " +
				f"{float(station_dict['station_hsurf']):.0f} m", ha='left', va='top', transform=ax_descr.transAxes,
				fontsize=fs-1)
ax_descr.text(0.99, 0.1, f"{DATA_DS.institution}", ha='right', va='bottom', transform=ax_descr.transAxes,
				fontsize=fs_small)


# Set up axes:
# ax_descr.axis('off')			# uncomment to remove axis for meta data (which would other wise serve as text box)

# Labels:
ax_time_height.set_ylabel(("Temperature ($^{\circ}$C)\nStratiform cloud cover (octa)\n" +
							"Upper wind\n \n Flightlevel FL / Pressure (hPa)"), 
							fontsize=fs-4, labelpad=80)
ax_wind.set_ylabel("Wind speed\n(kt)")
ax_wind.yaxis.set_label_position('right')
ax_T.set_ylabel("Temperature\n($^{\circ}$C)")
ax_T.yaxis.set_label_position('right')
axP_ylabel = ax_P.set_ylabel("Pressure (hPa)", fontsize=fs-4)
ax_IWV.set_ylabel("IWV (mm)")
ax_precip_ylabel= ax_precip.set_ylabel("Precipitation\n(mm)")
ax_snow.set_ylabel("Snow (cm)")

# set x axis limits:
time_dt_minus_2h = time_dt[0] - dt.timedelta(hours=2)
time_dt_plus_2h = time_dt[-1] + dt.timedelta(hours=2)
ax_time_height.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)
ax_wind.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)
ax_T.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)
ax_P.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)
ax_precip.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)
ax_time.set_xlim(left=time_dt_minus_2h, right=time_dt_plus_2h)

# set y axis limits:
ax_time_height.set_ylim(bottom=y_lim_time_height[0], top=y_lim_time_height[1])
ax_wind.set_ylim(bottom=y_lim_wind[0], top=y_lim_wind[1])
ax_T.set_ylim(bottom=y_lim_T[0], top=y_lim_T[1])
ax_P.set_ylim(bottom=y_lim_P[0], top=y_lim_P[1])
ax_IWV.set_ylim(bottom=y_lim_IWV[0], top=y_lim_IWV[1])
ax_precip.set_ylim(bottom=y_lim_precip[0], top=y_lim_precip[1])
ax_snow.set_ylim(bottom=y_lim_snow[0], top=y_lim_snow[1])
ax_time.set_ylim(0,1)

# set y ticks and tick labels:
ax_time_height.yaxis.set_ticks(np.arange(150, 1001, 50))
ax_wind.yaxis.tick_right()
ax_wind.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax_T.yaxis.tick_right()
ax_T.yaxis.set_ticks(np.arange((y_lim_T[0]//5 + 1)*5, (y_lim_T[1]//5)*5+1, 5))

# remove some eventually overlapping ticks:
if ax_T.get_yticks()[-1] == y_lim_T[1]:
	ax_T.yaxis.set_ticks(ax_T.get_yticks()[:-1])			# remove top tick
if ax_IWV.get_yticks()[-1] == y_lim_IWV[1]:
	ax_IWV.yaxis.set_ticks(ax_IWV.get_yticks()[:-1])		# remove top tick
if ax_precip.get_yticks()[-1] == y_lim_precip[1]:
	ax_precip.yaxis.set_ticks(ax_precip.get_yticks()[:-1])	# remove top tick
if ax_snow.get_yticks()[-1] == y_lim_snow[1]:
	ax_snow.yaxis.set_ticks(ax_snow.get_yticks()[:-1])		# remove top tick
ax_time.yaxis.set_ticks(np.array([]))
ax_time.yaxis.set_ticklabels([])
ax_descr.yaxis.set_ticks([])

# set face color of time height plot:
# ax_time_height.set_facecolor((153/255,205/255,255/255))	# uncomment to get that blue face color back

# Remove x tick labels for nearly all axes
ax_time_height.xaxis.set_ticks(x_ticks_major_dt)
ax_time_height.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_time_height.xaxis.set_ticklabels([])
ax_wind.xaxis.set_ticks(x_ticks_major_dt)
ax_wind.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_wind.xaxis.set_ticklabels([])
ax_T.xaxis.set_ticks(x_ticks_major_dt)
ax_T.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_T.xaxis.set_ticklabels([])
ax_P.xaxis.set_ticks(x_ticks_major_dt)
ax_P.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_P.xaxis.set_ticklabels([])
ax_precip.xaxis.set_ticks(x_ticks_major_dt)
ax_precip.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_precip.xaxis.set_ticklabels([])
ax_descr.xaxis.set_ticks([])

# Set time labels in ax_time:
ax_time.xaxis.set_ticks(x_ticks_major_dt)
ax_time.xaxis.set_ticks(x_ticks_minor_dt, minor=True)
ax_time.xaxis.set_major_formatter(dt_fmt_major)
ax_time.xaxis.set_minor_formatter(dt_fmt_minor)

# y axis grid:
ax_time_height.yaxis.grid(which='major', linewidth=1.0, linestyle='dotted', color=c_grid)
ax_wind.yaxis.grid(which='major', linewidth=1.0, linestyle='dotted', color=c_grid)
ax_wind.yaxis.grid(which='minor', linestyle='none')
ax_T.yaxis.grid(which='major', linewidth=1.0, linestyle='dotted', color=c_grid)
ax_P.yaxis.grid(which='major', linewidth=1.0, linestyle='dotted', color=c_grid)
ax_precip.yaxis.grid(which='major', linewidth=1.0, linestyle='dotted', color=c_grid)

# Limit axis width:
plt.subplots_adjust(hspace=0.0)			# removes space between subplots

# Adjust axis width and position:
ax_time_height_pos = ax_time_height.get_position().bounds
ax_time_height.set_position([ax_time_height_pos[0] + 0.18*ax_time_height_pos[2], ax_time_height_pos[1],
							ax_time_height_pos[2]*0.75, ax_time_height_pos[3]])
ax_wind_pos = ax_wind.get_position().bounds
ax_wind.set_position([ax_wind_pos[0] + 0.18*ax_wind_pos[2], ax_wind_pos[1],
							ax_wind_pos[2]*0.75, ax_wind_pos[3]])
ax_T_pos = ax_T.get_position().bounds
ax_T.set_position([ax_T_pos[0] + 0.18*ax_T_pos[2], ax_T_pos[1],
							ax_T_pos[2]*0.75, ax_T_pos[3]])
ax_P_pos = ax_P.get_position().bounds
ax_P.set_position([ax_P_pos[0] + 0.18*ax_P_pos[2], ax_P_pos[1],
							ax_P_pos[2]*0.75, ax_P_pos[3]])
ax_precip_pos = ax_precip.get_position().bounds
ax_precip.set_position([ax_precip_pos[0] + 0.18*ax_precip_pos[2], ax_precip_pos[1],
							ax_precip_pos[2]*0.75, ax_precip_pos[3]])
ax_time_pos = ax_time.get_position().bounds
ax_time.set_position([ax_time_pos[0] + 0.18*ax_time_pos[2], ax_time_pos[1],
							ax_time_pos[2]*0.75, ax_time_pos[3]])
ax_descr_pos = ax_descr.get_position().bounds
ax_descr.set_position([ax_descr_pos[0] + 0.18*ax_descr_pos[2], ax_descr_pos[1],
							ax_descr_pos[2]*0.75, ax_descr_pos[3]])

# Save figure .. or maybe not
if save_figures:
	fig1.savefig(plot_path + file.replace(path_data, "")[:-3] + ".png", dpi=300)
else:
	plt.show()