import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pdb

"""
Script to compute the bending moment curve of a girder that's equally and
fully loaded (line load covering the entire girder).

Example computation shown below. The generalized form is used in the actual code:

for i, x_i in enumerate(x):
	if (x_i >= 0) and (x_i < pos_supports[0]):
		M[i] = 0.5*q*x_i**2
	elif (x_i >= pos_supports[0]) and (x_i < pos_supports[1]):
		M[i] = 0.5*q*x_i**2 + support_forces[0]*(x_i - pos_supports[0])
	elif (x_i >= pos_supports[1]) and (x_i < pos_supports[2]):
		M[i] = 0.5*q*x_i**2 + support_forces[0]*(x_i - pos_supports[0]) + support_forces[1]*(x_i - pos_supports[1])
	elif (x_i >= pos_supports[2]) and (x_i < pos_supports[3]):
		M[i] = (0.5*q*x_i**2 + support_forces[0]*(x_i - pos_supports[0]) + support_forces[1]*(x_i - pos_supports[1]) + 
				support_forces[2]*(x_i - pos_supports[2]))
	elif (x_i >= pos_supports[3]) and (x_i < pos_supports[4]):
		M[i] = (0.5*q*x_i**2 + support_forces[0]*(x_i - pos_supports[0]) + support_forces[1]*(x_i - pos_supports[1]) + 
				support_forces[2]*(x_i - pos_supports[2]) + support_forces[3]*(x_i - pos_supports[3]))
	elif (x_i >= pos_supports[4]) and (x_i < l_tot):
		M[i] = (0.5*q*x_i**2 + support_forces[0]*(x_i - pos_supports[0]) + support_forces[1]*(x_i - pos_supports[1]) + 
				support_forces[2]*(x_i - pos_supports[2]) + support_forces[3]*(x_i - pos_supports[3]) +
				support_forces[4]*(x_i - pos_supports[4]))
"""

save_figures = True			# specify whether or not to save the figure


# Define system: structure your arrays as follows: 
# start on the left (x=0); end on the right (x=sum(legs))
legs = np.array([4.0,4.0,4.0,4.0])		# length of each field (in m)
pos_supports = np.array([0.0, 4.0, 8.0, 12.0, 16.0])		# position on x-axis of supports (in m)
q = -15.0				# total (non- and permanent) line load: negative if facing downwards (in kN/m)
support_forces = np.array([24.0, 66.0, 60.0, 66.0, 24.0])		# support reactions (in kN): positive if upwards


# Compute bending moment:
l_tot = np.sum(legs)	# tot. length of the system
dx = 0.001				# resolution of x axis
x = np.arange(0, l_tot + dx, dx)		# x axis
M = np.full_like(x, 0.0)				# momentum curve initialized

for i, x_i in enumerate(x):

	# find out how many supports have been passed already:
	# count supports where x_i >= support positions
	M_addition = 0
	for j in range(np.count_nonzero(pos_supports <= x_i)):
		M_addition += support_forces[j]*(x_i - pos_supports[j])

	M[i] = 0.5*q*x_i**2 + M_addition


# Visualize M:
min_M = np.nanmin(M)
max_M = np.nanmax(M)
M_range = max_M - min_M


fs = 18
fig1 = plt.figure(figsize=(14,8))
ax1 = plt.axes()

ax1.plot(x, M, color=(0,0,0), linewidth=1.3, label="Biegemoment")
ax1.plot(x, np.full_like(x, 0.0), color=(0,0,0), linewidth=1.0)
ax1.plot(pos_supports, np.full_like(pos_supports, 0.0), linestyle='none', marker=6, 
			markersize=12.0, color=(0,0,0), label="Auflager")

ax1.set_xlim(x[0]-0.05*l_tot, x[-1] + 0.05*l_tot)
ax1.set_ylim(max_M + 0.05*M_range, min_M - 0.05*M_range)

ax1.minorticks_on()
ax1.grid(which='both', axis='both', color=(0.5,0.5,0.5))

lh, ll = ax1.get_legend_handles_labels()
ax1.legend(handles=lh, labels=ll, loc='upper right', fontsize=fs-2)

ax1.set_xlabel("Länge (m)", fontsize=fs)
ax1.set_ylabel("Drehmoment (kNm)", fontsize=fs)

if save_figures:
	date_now = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	fig1.savefig("/net/blanc/awalbroe/Codes/tragwerke/" + f"Biegemomentkurve_{date_now}" + ".png", dpi=300)
else:
	plt.show()