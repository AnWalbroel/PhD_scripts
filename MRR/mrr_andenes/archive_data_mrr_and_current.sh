#!/bin/bash

# File esxample: /cygdrive/c/archive_mrr/mrr_ave-6-0-0-6/2017/04/27/20170427_nya_mrr_ave-6-0-0-6.txt.gz

site=and
mrrPath='/data/obs/campaigns/comble/mrr_andenes/l1/'
archivePath='/data/obs/campaigns/comble/and/mrr/l1/'

ourProduct1='mrr_ave-6-0-0-7'
mrrProductSuffix1='ave'

ourProduct2='mrr_raw'
mrrProductSuffix2='raw'

start_date='20200101'
end_date='20201231'

start_date=$(/usr/bin/date -d "$start_date" +%Y%m%d)
end_date=$(/usr/bin/date -d "$end_date" +%Y%m%d)

archive_data ()
{
site=$1
mrrPath=$2
archivePath=$3
ourProduct=$4
mrrProductSuffix=$5
start_date=$6

longYearMonth=$(/usr/bin/date -d "$start_date" +%Y%m)
monthDay=$(/usr/bin/date -d "$start_date" +%m%d)

yearMonth=$(/usr/bin/date -d "$start_date" +%y%m)
yearMonthDay=$(/usr/bin/date -d "$start_date" +%y%m%d)

year=$(/usr/bin/date -d "$start_date" +%Y)
month=$(/usr/bin/date -d "$start_date" +%m)
day=$(/usr/bin/date -d "$start_date" +%d)

#make dirs
/usr/bin/mkdir -p $archivePath/$year/$month/$day

if [ -e "$mrrPath$year/$month/$day/$year$month$day"_"$site"_"$ourProduct".txt ]
then
  #gzip, move file
  echo $(date +"%Y-%m-%d %H:%M:%S") archiving "$mrrPath$year/$month/$day/$year$month$day"_"$site"_"$ourProduct".txt
  /usr/bin/cp -a "$mrrPath$year/$month/$day/$year$month$day"_"$site"_"$ourProduct".txt $archivePath$year/$month/$day/"$year$month$day"_"$site"_"$ourProduct".txt
  /usr/bin/gzip -f $archivePath$year/$month/$day/"$year$month$day"_"$site"_"$ourProduct".txt
fi
}

while [[ $start_date -le $end_date ]]
do
  archive_data $site $mrrPath $archivePath $ourProduct1 $mrrProductSuffix1 $start_date
  archive_data $site $mrrPath $archivePath $ourProduct2 $mrrProductSuffix2 $start_date

  start_date=$(/usr/bin/date -d "$start_date + 1 day" +"%Y%m%d")
done

exit