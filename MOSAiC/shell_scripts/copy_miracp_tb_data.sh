#!/bin/bash

# Copy MiRAC-P data from MOSAiC campaign to /net/blanc/awalbroe/Data/MOSAiC_radiometers/MiRAC-P_l1_v01/.


for yy in $(seq -w 2019 2020);
do
	for mm in $(seq -w 01 12); # -w detects max input width and adapts output
	do
		for dd in $(seq -w 01 31);
		do
			# quotation marks must be closed when finishing addressing variables and the string
			# shall continue without variables. If in a later part of the string another variable
			# needs to be addressed you will have to use quotation marks again.
			if [ -e "/data/obs/campaigns/mosaic/mirac-p/l1/$yy/$mm/$dd/MOSAiC_uoc_lhumpro-243-340_l1_tb_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/mirac-p/l1/$yy/$mm/$dd/MOSAiC_uoc_lhumpro-243-340_l1_tb_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/MiRAC-P_l1_v01/"
			fi
		done
	done
done

exit
