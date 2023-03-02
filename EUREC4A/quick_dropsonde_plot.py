import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
unified_version = 'v0.9'
path_unified = f"/data/obs/campaigns/eurec4a/HALO/unified/{unified_version}/"
from my_classes_eurec4a import *
from met_tools import compute_IWV, e_sat
which_date = "20200202"
HALO_dropsondes = dropsondes(path_unified, version=unified_version, which_date=which_date, save_time_height_matrix=False)
HALO_dropsondes.ntime = len(HALO_dropsondes.launch_time)
HALO_dropsondes.nheight = len(HALO_dropsondes.height)
launch_time_dt = np.array([dt.datetime.utcfromtimestamp(ttt) for ttt in HALO_dropsondes.launch_time])
y_lim_height = [0, 10500]

fs = 15
fig1 = plt.figure(figsize=(7,9))
ax_q = plt.subplot2grid((1,1), (0,0))

# ax_q.plot(HALO_dropsondes.q[17,:], HALO_dropsondes.height, color=(0.2,0.3,0.95), label='17')
ax_q.plot(HALO_dropsondes.q[17,:], HALO_dropsondes.height, color=(0.2,0.3,0.95), label='17')

ax_q.plot(HALO_dropsondes.q[-5,:], HALO_dropsondes.height, color=(0.77,0,0), label='-5')

# ax_q.plot(HALO_dropsondes.q[-5,:], HALO_dropsondes.height, color=(0.77,0,0,0.65), linestyle='dotted', label='-5')

ax_q.legend()
ax_q.grid()
ax_q.set_ylim(y_lim_height[0], y_lim_height[1])
ax_q.set_xlim(0, 0.017)
ax_q.set_ylabel("Height (m)")
ax_q.set_xlabel("Spec hum (kg kg^-1)")

plt.show()
plt.close()


fig1 = plt.figure(figsize=(7,9))
ax_q = plt.subplot2grid((1,1), (0,0))

# ax_q.plot(HALO_dropsondes.q[17,:], HALO_dropsondes.height, color=(0.2,0.3,0.95), label='17')
ax_q.plot(HALO_dropsondes.q[17,:], HALO_dropsondes.height, color=(0.2,0.3,0.95), label='17')

ax_q.plot(HALO_dropsondes.q[-5,:], HALO_dropsondes.height, color=(0.77,0,0), label='-5')

# ax_q.plot(HALO_dropsondes.q[-5,:], HALO_dropsondes.height, color=(0.77,0,0,0.65), linestyle='dotted', label='-5')

ax_q.legend()
ax_q.grid()
ax_q.set_ylim(y_lim_height[0], y_lim_height[1])
ax_q.set_xlim(0, 0.004)
ax_q.set_ylabel("Height (m)")
ax_q.set_xlabel("Spec hum (kg kg^-1)")

plt.show()