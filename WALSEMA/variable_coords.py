import numpy as np
import pdb
import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "monospace"})


# O and P represent two GPS stations with a fixed distance to each other.
# O and P move to O_D and P_D, respectively with a certain translation and rotation.
# Define coordinate system that moves with O and P. Find the rotation and translation.
# points in CAPITAL, vectors in small letters.


# stations before moving; at time 0:
O = np.array([3.0, 0.0])		# point location (i.e., in [deg E, deg N])
P = np.array([5.0, 1.0])

# and after moving; at time 1:
O_D = np.array([6.2, -2.6])
P_D = np.array([8.4, -3.3])


# vector p: p: O -> P; later: p_d: O_D -> P_D
p = P - O
p_d = P_D - O_D
e1 = np.array([1.0, 0.0])		# base vector of loc coord sys
e2 = np.array([0.0, 1.0])		# base vector of loc coord sys


# define vector p in local coordinate system before translation and rotation:
x, y = np.dot(p, e1), np.dot(p, e2)
p_loc = np.array([x, y])


# find translation:
tr = P_D - P


# find rotation: we can subtract it from 360.0 deg because we know the rotation sort of
p_norm = np.sqrt(np.dot(p,p))
alpha = 360.0 - np.arccos(np.dot(p, p_d) / (p_norm**2))*360.0 / (2.0*np.pi)

# find a, b, c, d to describe new coordinate system base vectors e1_d, e2_d as linear combi
# of e1 and e2: e1_d = a*e1 + b*e2; e2_d = c*e1 + d*e2
# use scalar product (dot product): < e1_d, e1 > = ||e1_d|| ||e1|| cos alpha = cos alpha = < (a,b), (1,0) >
# <=> cos alpha = a; similar for <e2_d, e2> => d = cos alpha;
# and because ||e1_d|| and ||e2_d|| == 1; b = +/- sqrt(1 - (cos alpha)**2); c = +/- sqrt(1 - (cos alpha)**2)
# then, testing < e1_d, e2_d > = 0 (orthonormal base) yields that either b xor c needs negative sign:
# I choose b to have negative sign.
a = np.cos(np.radians(alpha))
d = a
b = - np.sqrt(1.0 - a**2)
c = -b
e1_d = np.array([a, b])
e2_d = np.array([c, d])
p_d_control = x*e1_d + y*e2_d		# is the true p_d vector. it shows where P_D should lie. the value we
									# had for P_D was just 'measured'. p_d_control is the true vector if
									# the rotation was correct and the length of O->P remained the same in
									# O'->P'. 


# application: define a random track and move it along with the coordinate system:
a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y = 1.0, 1.0, 3.0, 1.3, 2.7, 2.5, 1.8, 2.9
A0 = O + a0x*e1 + a0y*e2
A1 = O + a1x*e1 + a1y*e2
A2 = O + a2x*e1 + a2y*e2
A3 = O + a3x*e1 + a3y*e2
o_a0 = A0 - O
o_a1 = A1 - O
o_a2 = A2 - O
o_a3 = A3 - O
# a0 = A1 - A0
# a1 = A2 - A1
# a2 = A3 - A2
# a3 = A0 - A3


# in new system:
A0_D = O_D + a0x*e1_d + a0y*e2_d
A1_D = O_D + a1x*e1_d + a1y*e2_d
A2_D = O_D + a2x*e1_d + a2y*e2_d
A3_D = O_D + a3x*e1_d + a3y*e2_d


# visualize stuff:
fs = 14
fs_small = fs- 2
fs_dwarf = fs_small - 2

c_ev = (0.4,0.9,0.4)
c_tr = (0.75,0,0)


f1 = plt.figure(figsize=(9,9))
a1 = plt.axes()

xlims = [-1.0, 11.0]
ylims = [-7.0, 5.0]

# visualize points:
a1.scatter(O[0], O[1], s=50, marker='.', color=(0,0,0))
a1.scatter(P[0], P[1], s=50, marker='.', color=(0,0,0))
a1.scatter(O_D[0], O_D[1], s=50, marker='.', color=(0,0,0))
a1.scatter(P_D[0], P_D[1], s=50, marker='.', color=(0,0,0))

# # P_D control:
# a1.scatter(O_D[0] + p_d_control[0], O_D[1] + p_d_control[1], s=50, marker='.', color=(0,0,0))

# points of random track in old system:
a1.scatter(A0[0], A0[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A1[0], A1[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A2[0], A2[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A3[0], A3[1], s=81, marker='.', color=(0,0,0))

# in new system:
a1.scatter(A0_D[0], A0_D[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A1_D[0], A1_D[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A2_D[0], A2_D[1], s=81, marker='.', color=(0,0,0))
a1.scatter(A3_D[0], A3_D[1], s=81, marker='.', color=(0,0,0))


# visualize vectors:
a1.plot(np.array([O[0], P[0]]), np.array([O[1], P[1]]), linewidth=1.2, color=(0,0,0))
a1.plot(np.array([O_D[0], P_D[0]]), np.array([O_D[1], P_D[1]]), linewidth=1.2, color=(0,0,0))
# a1.plot(np.array([O_D[0], O_D[0] + p_d_control[0]]), np.array([O_D[1], O_D[1] + p_d_control[1]]),
		# linestyle='dotted', color=(0,0,0), linewidth=1.2)

# base vectors:
a1.plot(np.array([O[0], O[0] + e1[0]]), np.array([O[1], O[1] + e1[1]]), linewidth=2, color=c_ev)
a1.plot(np.array([O[0], O[0] + e2[0]]), np.array([O[1], O[1] + e2[1]]), linewidth=2, color=c_ev)
a1.plot(np.array([O_D[0], O_D[0] + e1_d[0]]), np.array([O_D[1], O_D[1] + e1_d[1]]), linewidth=2, color=c_ev)
a1.plot(np.array([O_D[0], O_D[0] + e2_d[0]]), np.array([O_D[1], O_D[1] + e2_d[1]]), linewidth=2, color=c_ev)

# random track in old system:
a1.plot(np.array([A0[0], A1[0], A2[0], A3[0], A0[0]]), 
		np.array([A0[1], A1[1], A2[1], A3[1], A0[1]]), color=c_tr, linewidth=2.0)

# and in new system:
a1.plot(np.array([A0_D[0], A1_D[0], A2_D[0], A3_D[0], A0_D[0]]),
		np.array([A0_D[1], A1_D[1], A2_D[1], A3_D[1], A0_D[1]]), color=c_tr, linewidth=2.0)


# auxilliary lines to emphasize coordinate system origin:
a1.plot([0,0], ylims, color=(0,0,0), linewidth=0.75)
a1.plot(xlims, [0,0], color=(0,0,0), linewidth=0.75)


# put labels next to points:
a1.text(O[0], O[1], "O", color=(0,0,0), ha='right', va='bottom', fontsize=fs)
a1.text(O_D[0], O_D[1], "O'", color=(0,0,0), ha='right', va='top', fontsize=fs)
a1.text(P[0], P[1], "P", color=(0,0,0), ha='left', va='bottom', fontsize=fs)
a1.text(P_D[0], P_D[1], "P'", color=(0,0,0), ha='left', va='top', fontsize=fs)

# at base vectors:
a1.text(O[0] + 0.5*e1[0], O[1] + 0.5*e1[1], "e1", color=c_ev, ha='center', va='top', fontsize=fs_dwarf)
a1.text(O[0] + 0.5*e2[0], O[1] + 0.5*e2[1], "e2", color=c_ev, ha='right', va='center', fontsize=fs_dwarf)

# to put labels of new base vectors 'nicely' next to the vectors, use a vector perpendicular to base
# vectors:
e1_d_ot = np.array([-e1_d[1]/e1_d[0], 1.0])*(-0.15)
e2_d_ot = np.array([-e2_d[1]/e2_d[0], 1.0])*(0.15)

""" can be plotted to see how to align them (to get the factor +/- 0.05)
a1.plot(np.array([O_D[0] + 0.5*e1_d[0], O_D[0] + 0.5*e1_d[0] + e1_d_ot[0]]),
		np.array([O_D[1] + 0.5*e1_d[1], O_D[1] + 0.5*e1_d[1] + e1_d_ot[1]]),
		color=(0,0,1))
a1.plot(np.array([O_D[0] + 0.5*e2_d[0], O_D[0] + 0.5*e2_d[0] + e2_d_ot[0]]),
		np.array([O_D[1] + 0.5*e2_d[1], O_D[1] + 0.5*e2_d[1] + e2_d_ot[1]]),
		color=(0,0,1))
"""

a1.text(O_D[0] + 0.5*e1_d[0] + e1_d_ot[0], O_D[1] + 0.5*e1_d[1] + e1_d_ot[1], "e1'", color=c_ev, ha='center', va='center', 
			rotation=alpha, fontsize=fs_dwarf)
a1.text(O_D[0] + 0.5*e2_d[0] + e2_d_ot[0], O_D[1] + 0.5*e2_d[1] + e2_d_ot[1], "e2'", color=c_ev, ha='center', va='center', 
			rotation=alpha, fontsize=fs_dwarf)


# labels at points of random track:
a1.text(A0[0], A0[1], "A0", color=c_tr, ha='right', va='top', fontsize=fs)
a1.text(A1[0], A1[1], "A1", color=c_tr, ha='left', va='top', fontsize=fs)
a1.text(A2[0], A2[1], "A2", color=c_tr, ha='left', va='center', fontsize=fs)
a1.text(A3[0], A3[1], "A3", color=c_tr, ha='right', va='bottom', fontsize=fs)

a1.text(A0_D[0], A0_D[1], "A0'", color=c_tr, ha='right', va='top', fontsize=fs)
a1.text(A1_D[0], A1_D[1], "A1'", color=c_tr, ha='left', va='top', fontsize=fs)
a1.text(A2_D[0], A2_D[1], "A2'", color=c_tr, ha='left', va='center', fontsize=fs)
a1.text(A3_D[0], A3_D[1], "A3'", color=c_tr, ha='right', va='bottom', fontsize=fs)


a1.set_xlim(xlims)
a1.set_ylim(ylims)
a1.set_aspect("equal")
a1.tick_params(axis='both', labelsize=fs_small)

a1.minorticks_on()
a1.grid(which='both', axis='both', color=(0.75,0.75,0.75), alpha=0.5)

a1.set_xlabel("x", fontsize=fs)
a1.set_ylabel("y", fontsize=fs)

plt.show()
# f1.savefig("/mnt/d/Studium_NIM/work/Plots/WALSEMA/useless_stuff/rotating_coords.png", dpi=400, bbox_inches='tight')