import os
import sys
import shutil
import datetime as dt
import glob
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pdb


def get_exif_data(image):

	"""
	Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags
	"""

	exif_data = {}
	info = image._getexif()
	if info:
		for tag, value in info.items():
			decoded = TAGS.get(tag, tag)
			if decoded == "GPSInfo":
				gps_data = {}
				for gps_tag in value:
					sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
					gps_data[sub_decoded] = value[gps_tag]
				exif_data[decoded] = gps_data
			else:
				exif_data[decoded] = value

	return exif_data


"""
	Copy (or move) GoPro files, renaming them according to their creation
	date. Please specify the date of which the files should be copied by
	calling this script as 'python3 skycam_gopro_rename.py "2022-07-20"' or
	any other date in "yyyy-mm-dd" format.
"""


# path:
path_data = {	'in': "/mnt/e/SkyCam_GOPRO/",
				'out': "/mnt/d/heavy_data/WALSEMA/SkyCam_GOPRO/"}


# specify date of files to be renamed / copied:
if len(sys.argv) == 1:
	workdate = dt.datetime.utcnow()
else:
	workdate = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d")

# update paths and create output path if not existing:
for key in path_data.keys(): path_data[key] += workdate.strftime("%Y%m%d/")

path_data_out_dir = os.path.dirname(path_data['out'])
if not os.path.exists(path_data_out_dir):
	os.makedirs(path_data_out_dir)


# find files:
files = sorted(glob.glob(path_data['in'] + "*.JPG"))
if len(files) == 0: raise RuntimeError("No files found for " + workdate.strftime("%Y-%m-%d") + ".")


# iterate through files, find their creation / modification date and rename / copy them:
# dictionary respecting the time stamp offsets: start and end time where offsets apply, and offset itself
# saved to it.
offset_dict = {	
				'60min': [dt.datetime(2022,7,8,6,0,0), dt.datetime(2022,7,20,9,50,0), dt.timedelta(minutes=60)],
				'400min': [dt.datetime(2022,6,28,0,0), dt.datetime(2022,7,8,14,0,0), dt.timedelta(minutes=-400)]}
for file in files:
	meta = get_exif_data(Image.open(file))
	date = dt.datetime.strptime(meta["DateTime"], "%Y:%m:%d %H:%M:%S")

	# # check if offsets must be applied:
	# for key in offset_dict.keys():
		# if (date <= offset_dict[key][1]) & (date >= offset_dict[key][0]):
			# date += offset_dict[key][2]

	print("Copying " + file)
	shutil.copyfile(file, f"{path_data['out']}GOPRO{date:%Y%m%d%H%M%S}.JPG") # copy file