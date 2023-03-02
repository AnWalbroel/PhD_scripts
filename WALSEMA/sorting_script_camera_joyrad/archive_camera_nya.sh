#!/bin/bash

# Camera

year=$(/usr/bin/date '+%Y')
month=$(/usr/bin/date '+%m')
day=$(/usr/bin/date '+%d')
hourmin=$(/usr/bin/date '+%H%M')

timestamp=$(date +'%F %T')
echo --- $timestamp ---

url="http://unikoeln:MOBOTIX_10@172.20.4.202/record/current.jpg"

archivePath="/cygdrive/c/archive_camera_joyrad/"
programDir="/home/parsivel/sorting_script_camera_joyrad/"
tmpDir="/home/parsivel/sorting_script_camera_joyrad/tmp/"

# localFullPath="$archivePath/$year$month"
localFullPath="$archivePath/$year/$month/$day"

prefix="nya_joyrad_cam_"
fileName=$prefix$year$month$day$hourmin.jpg

/bin/mkdir -p $localFullPath
#/usr/bin/curl -s $url --connect-timeout 5 > $tmpDir/current.jpg
/cygdrive/c/Windows/system32/curl -s $url --connect-timeout 5 > $tmpDir/current.jpg
if [ $? -eq 0 ]
then
    if [[ $(file -b $tmpDir/current.jpg | awk '{print $1}') == "JPEG" ]]
    then
        echo "picture downloaded"
        /bin/mv $tmpDir/current.jpg $localFullPath/$fileName
    else
        echo file downloaded but it is not an image. 
        /bin/cp $programDir/no-data.jpg $localFullPath/$fileName
   fi
else
    echo "cannot connect to camera"
    /bin/cp $programDir/no-data.jpg $localFullPath/$fileName
fi

echo done.