#!/bin/bash

# Fetch MRR data from FESSTVaL campaign. Save to a temporary directory, convert to .txt, gzip them
# and move the zipped files to their destination folder.

USER=fesstval
PASS='7Tq9c4Yg'

mrrPath="/work/tbkoelnbonn/Daten_Birkholz/MRR/MRR Data/"
archivePath="/net/blanc/awalbroe/Data/MRR/archive_mrr/"

ourProduct1='mrr_ave-6-0-0-6'
mrrProduct1='AveData'
mrrProductSuffix1='ave'

ourProduct2='mrr_raw'
mrrProduct2='RawSpectra'
mrrProductSuffix2='raw'

n_days_ago='2'
site='fes'


fetch_data()
{
USER=$1
PASS=$2
mrrPath=$3
ourProduct=$4
mrrProduct=$5
mrrProductSuffix=$6
i_day_ago=$7
archivePath=$8
site=$9


# use date to access correct paths and files:
longYearMonth=$(/usr/bin/date -d "$i_day_ago day ago" +%Y%m)
monthDay=$(/usr/bin/date -d "$i_day_ago day ago" +%m%d)

longYear=$(/usr/bin/date -d "$i_day_ago day ago" +%Y)
month=$(/usr/bin/date -d "$i_day_ago day ago" +%m)
day=$(/usr/bin/date -d "$i_day_ago day ago" +%d)

# make directory and fetch data:
# /usr/bin/mkdir -p "$savePath$mrrProduct/$longYearMonth/"   # not needed, directly move it to final destination
/usr/bin/mkdir -p $archivePath$ourProduct/$longYear/$month/$day/

fetch_path=$mrrPath$mrrProduct/$longYearMonth/$monthDay.$mrrProductSuffix
echo $fetch_path

echo "Fetching MRR data: ftp://ftp-projects.cen.uni-hamburg.de$fetch_path"
wget --user=$USER --password=$PASS -P $archivePath$ourProduct/$longYear/$month/$day/ ftp://ftp-projects.cen.uni-hamburg.de"$fetch_path"


# check existence of file:
if [ -e "$archivePath$ourProduct/$longYear/$month/$day/$monthDay.$mrrProductSuffix" ]
then
	/usr/bin/mv "$archivePath$ourProduct/$longYear/$month/$day/$monthDay.$mrrProductSuffix" "$archivePath$ourProduct/$longYear/$month/$day/$longYear$month$day"_"$site"_"$ourProduct.txt"
	/usr/bin/gzip -f "$archivePath$ourProduct/$longYear/$month/$day/$longYear$month$day"_"$site"_"$ourProduct.txt"
fi


}

for i_day_ago in $(seq 1 $n_days_ago);
do
	fetch_data $USER $PASS "$mrrPath" $ourProduct1 $mrrProduct1 $mrrProductSuffix1 $i_day_ago $archivePath $site
	fetch_data $USER $PASS "$mrrPath" $ourProduct2 $mrrProduct2 $mrrProductSuffix2 $i_day_ago $archivePath $site
done



exit
