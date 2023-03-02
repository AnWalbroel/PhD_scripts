#!/bin/bash

# Copy HATPRO data from MOSAiC campaign to /net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/.


for yy in $(seq -w 2019 2020);
do
	for mm in $(seq -w 01 12); # -w detects max input width and adapts output
	do
		for dd in $(seq -w 01 31);
		do
			# quotation marks must be closed when finishing addressing variables and the string
			# shall continue without variables. If in a later part of the string another variable
			# needs to be addressed you will have to use quotation marks again.
			if [ -e "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_clwvi_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_clwvi_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_prw_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_prw_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_hua_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_hua_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_ta_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwr00_l2_ta_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"
			fi

			if [ -e "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwrBL00_l2_ta_v01_$yy$mm$dd"*.nc ]
			then
				/usr/bin/cp "/data/obs/campaigns/mosaic/hatpro/l2/$yy/$mm/$dd/ioppol_tro_mwrBL00_l2_ta_v01_$yy$mm$dd"*.nc "/net/blanc/awalbroe/Data/MOSAiC_radiometers/HATPRO_l2_v01/"
			fi
		done
	done
done

exit
