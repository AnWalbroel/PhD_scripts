import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import glob
import gc
import pdb

date0 = dt.datetime.strptime("2022-06-28", "%Y-%m-%d")
date1 = dt.datetime.strptime("2022-08-12", "%Y-%m-%d")

path_data = "/data/obs/campaigns/WALSEMA/atm/hatpro/l1/"

nowdate = date0
while nowdate <= date1:
	print(nowdate.strftime("%Y-%m-%d"))
	yyyy = f"{nowdate.year:04}"
	mm = f"{nowdate.month:02}"
	dd = f"{nowdate.day:02}"

	files = sorted(glob.glob(path_data + f"{yyyy}/{mm}/{dd}/" + f"ioppol_tro_mwr00_l1_tb_v00_{yyyy}{mm}{dd}*.nc"))
	# pdb.set_trace()
	if len(files) == 0:
		nowdate += dt.timedelta(days=1)
		continue
	elif len(files) > 1:
		pdb.set_trace()
	else:
		DS = xr.open_dataset(files[0])

		f1, a1 = plt.subplots(2,1)
		for k in range(7): a1[0].plot(DS.time, DS.tb[:,k])

		for k in range(7): a1[1].plot(DS.time, DS.tb[:,k+7])

		plt.show()

		DS.close()
		del DS
		gc.collect()
	nowdate += dt.timedelta(days=1)