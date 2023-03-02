import numpy as np
import datetime as dt
import glob
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
from my_mrr_classes import *
import sys

sys.path.insert(0, '/net/blanc/awalbroe/Codes/MOSAiC/')    # so that functions and modules from MOSAiC can be used


"""
	Testing the FESSTVaL MRR data if they produce any visible and reasonable signals.
"""

path_data = "/net/blanc/awalbroe/Data/MRR/"
path_plot = "/net/blanc/awalbroe/Plots/MRR/"


file_filter = "_raw"
files = sorted(glob.glob(path_data + "*%s.txt"%(file_filter)))
file = files[0]

mrr_data = MRR(file, with_datetime=True)



# some quick quicklooks:
fig1, ax0 = plt.subplots(nrows=4, ncols=1, sharex=True, squeeze=True,
							gridspec_kw={'hspace': 0.2}, figsize=(10,20))

mrr_data.z[mrr_data.z == -9999.0] = np.nan
mrr_data.Z[mrr_data.Z == -9999.0] = np.nan
mrr_data.RR[mrr_data.RR == -9999.0] = np.nan
mrr_data.LWC[mrr_data.LWC == -9999.0] = np.nan

cf_z = ax0[0].contourf(mrr_data.time_dt, mrr_data.height, np.transpose(mrr_data.z))
ax0[0].set_ylabel("Height (m)")
cb0 = fig1.colorbar(cf_z, ax=ax0[0])
cb0.set_label(label="z (dBZ?)")

cf_Z = ax0[1].contourf(mrr_data.time_dt, mrr_data.height, np.transpose(mrr_data.Z))
ax0[1].set_ylabel("Height (m)")
cb1 = fig1.colorbar(cf_Z, ax=ax0[1])
cb1.set_label(label="Reflectivity factor Z (dBZ)")

cf_RR = ax0[2].contourf(mrr_data.time_dt, mrr_data.height, np.transpose(mrr_data.RR))
ax0[2].set_ylabel("Height (m)")
cb2 = fig1.colorbar(cf_RR, ax=ax0[2])
cb2.set_label(label="Rain rate (mm/h)")

cf_LWC = ax0[3].contourf(mrr_data.time_dt, mrr_data.height, np.transpose(mrr_data.LWC))
ax0[3].set_ylabel("Height (m)")
cb3 = fig1.colorbar(cf_LWC, ax=ax0[3])
cb3.set_label(label="Liquid water content (g/m^3)")

plt.show()
fig1.savefig(path_plot + "MRR_test_%s_%s.png"%(file_filter, 
							dt.datetime.strftime(mrr_data.time_dt[0], "%Y%m%d")), dpi=300)


print("Done....")