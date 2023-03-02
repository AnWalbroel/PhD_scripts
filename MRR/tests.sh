#!/bin/bash

noDaysAgo='10'


print_date()
{
noDayAgo=$1

echo $(/usr/bin/date -d "$noDayAgo day ago" +%Y%m%d)



}

for noDayAgo in $(seq 1 $noDaysAgo);
do
	print_date $noDayAgo
	# echo $noDayAg
        process_date=$(/usr/bin/date -d "$noDayAgo day ago" +%Y-%m-%d)
        echo "Fetching MRR data for " $process_date....
done

str1="Gekki "
str2="wirkd"
str3="$str1$str2"

echo "$str3"
