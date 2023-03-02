#!/bin/bash

# Copy MRR data plots from FESSTVaL campaign to /net/blanc/awalbroe/Plots/MRR/.

yyyy='2021'

for mm in $(seq -w 01 12); # -w detects max input width and adapts output
do
	for dd in $(seq -w 01 31);
	do
		# quotation marks must be closed when finishing addressing variables and the string
		# shall continue without variables. If in a later part of the string another variable
		# needs to be addressed you will have to use quotation marks again.
		if [ -e "/data/obs/campaigns/FESSTVaL/mrr/l1/$yyyy/$mm/$dd/$yyyy$mm$dd"_fes_mrr_ave.png ]
		then
			/usr/bin/cp "/data/obs/campaigns/FESSTVaL/mrr/l1/$yyyy/$mm/$dd/$yyyy$mm$dd"_fes_mrr_ave.png "/net/blanc/awalbroe/Plots/MRR/"
		fi

		if [ -e "/data/obs/campaigns/FESSTVaL/mrr/l1/$yyyy/$mm/$dd/$yyyy$mm$dd"_fes_mrr_improtoo.png ]
		then
			/usr/bin/cp "/data/obs/campaigns/FESSTVaL/mrr/l1/$yyyy/$mm/$dd/$yyyy$mm$dd"_fes_mrr_improtoo.png "/net/blanc/awalbroe/Plots/MRR/"
		fi
	done
done

exit
