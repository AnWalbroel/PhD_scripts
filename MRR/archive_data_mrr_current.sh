#!/bin/bash

# File esxample: /cygdrive/c/archive_mrr/mrr_ave-6-0-0-6/2017/04/27/20170427_nya_mrr_ave-6-0-0-6.txt.gz

site=fes
mrrPath='/work/tbkoelnbonn/Daten_Birkholz/MRR/'
archivePath='/work/tbkoelnbonn/Daten_Birkholz/MRR/archive_mrr/'

ourProduct1='mrr_ave-6-0-0-6'
mrrProduct1='AveData'
mrrProductSuffix1='ave'

ourProduct2='mrr_raw'
mrrProduct2='RawSpectra'
mrrProductSuffix2='raw'

noDaysAgo='0'

archive_data ()
{
site=$1
mrrPath=$2
archivePath=$3
ourProduct=$4
mrrProduct=$5
mrrProductSuffix=$6
noDayAgo=$7

longYearMonth=$(/usr/bin/date -d "$noDayAgo day ago" +%Y%m)
monthDay=$(/usr/bin/date -d "$noDayAgo day ago" +%m%d)

yearMonth=$(/usr/bin/date -d "$noDayAgo day ago" +%y%m)
yearMonthDay=$(/usr/bin/date -d "$noDayAgo day ago" +%y%m%d)

year=$(/usr/bin/date -d "$noDayAgo day ago" +%Y)
month=$(/usr/bin/date -d "$noDayAgo day ago" +%m)
day=$(/usr/bin/date -d "$noDayAgo day ago" +%d)

#make dirs
/usr/bin/mkdir -p $archivePath/$ourProduct/$year/$month/$day

#if [ -e '$archivePath/$ourProduct/$year/$month/$day/$year$month$day"_"$site"_"$ourProduct".txt.gz' ]
if [ -e "$mrrPath/$mrrProduct/$longYearMonth/$monthDay.$mrrProductSuffix" ]
then
  #gzip, move file
  echo $(date +"%Y-%m-%d %H:%M:%S") archiving "$mrrPath/$mrrProduct/$longYearMonth/$monthDay.$mrrProductSuffix"
  /usr/bin/cp -a "$mrrPath/$mrrProduct/$longYearMonth/$monthDay.$mrrProductSuffix" $archivePath/$ourProduct/$year/$month/$day/"$year$month$day"_"$site"_"$ourProduct".txt
  /usr/bin/gzip -f $archivePath/$ourProduct/$year/$month/$day/"$year$month$day"_"$site"_"$ourProduct".txt
fi
}

for noDayAgo in $(seq 0 $noDaysAgo);
do
  archive_data $site $mrrPath $archivePath $ourProduct1 $mrrProduct1 $mrrProductSuffix1 $noDayAgo
  archive_data $site $mrrPath $archivePath $ourProduct2 $mrrProduct2 $mrrProductSuffix2 $noDayAgo
done

exit