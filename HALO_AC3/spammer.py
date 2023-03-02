import sys
import datetime as dt
import os
import pdb

sys.path.insert(0, "/net/blanc/awalbroe/Codes/MOSAiC/")


"""
	Test script that first creates a file and then writes some stuff in it.
	The script will be called every 2 minutes via a crontab command.
	At each full hour a new file will be created.
"""

# paths:
path_output = "/net/blanc/awalbroe/Plots/spammer/"

# check if path exists:
path_dir = os.path.dirname(path_output)
if not os.path.exists(path_dir): os.makedirs(path_dir)


# Inquire current time:
nowtime = dt.datetime.utcnow()

# Set filename and check existence:
filename_time = nowtime.strftime("spam%y%m%d%H")
filename = path_output + filename_time + ".txt"

try:	# read out last number and append increment to existing file
	with open(filename, 'r') as f:	# also used to check existence
		lines = f.readlines()
		last_num = int(lines[-1].split('\n')[0])

	with open(filename, 'a') as f:
		f.write(str(last_num + 1))
		f.write('\n')

except FileNotFoundError:	# create new file
	with open(filename, 'w') as f:
		f.write(str(0))
		f.write('\n')