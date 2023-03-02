from shutil import copyfile
import os
import glob
import datetime as dt
import pdb


"""
	Copy files from /data/obs/campaigns/mosaic/hatpro/... to another destination
	so that I have all HATPRO files in one place.
"""

path_hatpro_base = "/data/obs/campaigns/WALSEMA/atm/hatpro/l2/"
final_destination = "/net/blanc/awalbroe/Data/WALSEMA/hatpro/"


considered_period = 'walsema'		# specify which period shall be plotted or computed:
									# 'mosaic': entire mosaic period (2019-09-20 - 2020-10-12)
									# 'walsema': entire mosaic period (2022-06-28 - 2022-08-12)
# Date range of (mwr and radiosonde) data: Please specify a start and end date 
# in yyyy-mm-dd!
daterange_options = {'mosaic': ["2019-09-20", "2020-10-12"],
					'walsema': ["2022-06-28", "2022-08-12"]}
date_start = dt.datetime.strptime(daterange_options[considered_period][0], "%Y-%m-%d")	# def: "2019-09-30"
date_end = dt.datetime.strptime(daterange_options[considered_period][1], "%Y-%m-%d")	# def: "2020-10-02"
n_days = (date_end - date_start).days + 1

# cycle through days:
for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):
	
	yyyy = now_date.year
	mm = now_date.month
	dd = now_date.day

	day_path = path_hatpro_base + "%04i/%02i/%02i/"%(yyyy,mm,dd)

	if not os.path.exists(os.path.dirname(day_path)):
		continue

	# list of files:
	file_list = sorted(glob.glob(day_path + "*_clwvi_v00_*.nc"))

	if len(file_list) == 0:
		continue

	# copy file(s):
	file = file_list[0]
	copyfile(file, final_destination + os.path.basename(file))
