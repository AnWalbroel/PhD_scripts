import pdb
import sys
import os
import glob
from PIL import Image
import datetime as dt


def crop_img(file, path_output):

	"""
	Cropping manually saved MSS images.

	Parameters:
	-----------
	file : str
		Path + filename of image to be cropped.
	path_output : str
		Path where the cropped image will be saved to.
	which_map : str
		Indicates if this is the large scale or local Longyearbyen map.
		Cropping excerpt changes accordingly.
	"""

	with Image.open(file) as uncropped_img:
		uncropped_img.crop((2137,2379,2137+2022,2379+2373)).save(path_output + os.path.basename(file))

"""
	Script to crop images saved manually from internal browser on Polarstern.
	Old files will be overwritten when the output path is identical to the path 
	where the original files lie. 
"""



# Locate images
path_img = f"/mnt/c/Users/tenweg/Downloads/"
path_output = path_img + "cropped/"

# Create output path if not existent:
if not os.path.exists(path_output):
	os.makedirs(path_output)


# find files and loop through them:
files = list()
files = files + sorted(glob.glob(path_img + "*.tif"))
files = files + sorted(glob.glob(path_img + "*.tiff"))
files = files + sorted(glob.glob(path_img + "*.TIF"))
files = files + sorted(glob.glob(path_img + "*.TIFF"))
for file in files:
	print(f"Cropping {os.path.basename(file)}....")
	crop_img(file, path_output)