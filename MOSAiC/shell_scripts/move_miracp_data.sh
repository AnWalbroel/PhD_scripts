#!/bin/bash

# Move MiRAC-P data from MOSAiC campaign to the temporary directory 
# /data/obs/campaigns/mosaic/mirac-p/l2/del_files/.


for yy in $(seq -w 2019 2020);
do
	for mm in $(seq -w 01 12); # -w detects max input width and adapts output (e.g. padding zeros)
	do
		for dd in $(seq -w 01 31);
		do
			# quotation marks must be closed when finishing addressing variables and the string
			# shall continue without variables. If in a later part of the string another variable
			# needs to be addressed you will have to use quotation marks again.
			if [ -e "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i01_$yy$mm$dd"000000.nc ]
			then
				/usr/bin/mv "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i01_$yy$mm$dd"000000.nc "/data/obs/campaigns/mosaic/mirac-p/l2/del_files/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i02_$yy$mm$dd"000000.nc ]
			then
				/usr/bin/mv "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i02_$yy$mm$dd"000000.nc "/data/obs/campaigns/mosaic/mirac-p/l2/del_files/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i03_$yy$mm$dd"000000.nc ]
			then
				/usr/bin/mv "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i03_$yy$mm$dd"000000.nc "/data/obs/campaigns/mosaic/mirac-p/l2/del_files/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i04_$yy$mm$dd"000000.nc ]
			then
				/usr/bin/mv "/data/obs/campaigns/mosaic/mirac-p/l2/$yy/$mm/$dd/ioppol_uoc_mwr01_l2_prw_i04_$yy$mm$dd"000000.nc "/data/obs/campaigns/mosaic/mirac-p/l2/del_files/"
			fi
		done
	done
done

exit