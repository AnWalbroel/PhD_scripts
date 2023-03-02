# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import sys
#import Image
from PIL import Image
import warnings
import traceback
import calendar
import datetime
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'

#add containing folder to path to load IMProToo
sys.path.append("/".join(sys.argv[0].split("/")[0:-1]))
import IMProToo

###settings ###
noDaysAgo = 31
skipExisting = False



#sites = ["jue","ufs","nya"]
#sites = ["jue","ufs","uf2","ant"]

date_start = datetime.datetime.strptime(sys.argv[1], "%Y%m%d")
date_end = datetime.datetime.strptime(sys.argv[2], "%Y%m%d")
sites = sys.argv[3:]
#sites=["jue"]

siteLongs = {
  "jue":"FZ Juelich",
  "ufs":"UFS Schneefernerhaus MRR DLR",
  "uf2":"UFS Schneefernerhaus MRR2 UK",
  "nya":"Ny-Alesund",
  "comble":"Bjornoya",
  "fes":"FESSTVaL",
  "and":"Andenes",
}
siteUnicodeLongs = {
  "jue":u"FZ Jülich",
  "ufs":"UFS Schneefernerhaus MRR DLR",
  "uf2":"UFS Schneefernerhaus MRR2 UK",
  "nya":u"AWIPEV (Ny-Ålesund)",
  "comble":u"Bjørnøya",
  "fes":u"FESSTVaL Campaign",
  "and":u"Andenes",
}

# added Longitude, Latitude and Altitude information to stations: 2022-06-08 by A. Walbroel
siteLongitudes = {
  "jue":6.414,
  "ufs":10.980, # source: Wikipedia
  "uf2":10.980, # source: Wikipedia
  "ant":np.nan,
  "nya":11.921,
  "comble":np.nan,
  "fes":np.nan,
  "and":np.nan,
}
siteLatitudes = {
  "jue":50.909,
  "ufs":47.417, # source: Wikipedia
  "uf2":47.417, # source: Wikipedia
  "ant":np.nan,
  "nya":78.923,
  "comble":np.nan,
  "fes":np.nan,
  "and":np.nan,
}
siteAltitudes = {
  "jue":111.0,
  "ufs":2656.0, # source: Wikipedia
  "uf2":2656.0, # source: Wikipedia
  "ant":np.nan,
  "nya":0.0,
  "comble":np.nan,
  "fes":np.nan,
  "and":np.nan,
}

products= {
  "jue":"mrr_ave-6-0-0-6",
  "ufs":"mrr_ave-5-20",
  "uf2":"mrr_ave-6-0-0-4",
  "nya":"mrr_ave-6-0-0-6",
  "comble":"mrr_ave-6-0-0-9",
  "fes":"mrr_ave-6-0-0-6",
  "and":"mrr_ave-6-0-0-7",
}

contactPersons= {
  "jue":"Bernhard Pospichal (bernhard.pospichal@uni-koeln.de)",
  "ufs":"Martin Hagen (martin.hagen@dlr.de)",
  "uf2":"NN",
  "nya":"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
  "comble":"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
  "fes":"Andreas Walbroel (a.walbroel@uni-koeln.de)",
  "and":"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
}

#institution here as the owner of the instrument/data
institutions= {
  "jue":"University of Cologne, Institute for Geophysics and Meteorology",
  "ufs":"German Aerospace Center, Institute of Atmospheric Physics",
  "uf2":"NN",
  "nya":"University of Cologne, Institute for Geophysics and Meteorology",
  "comble":"University of Cologne, Institute for Geophysics and Meteorology",
  "fes":"University of Cologne, Institute for Geophysics and Meteorology",
  "and":"University of Cologne, Institute for Geophysics and Meteorology",
}

productPng = "mrr_ave"
productLong ="MRR AverageData"




pathIns = {
  #"jue":"/data/hatpro/jue/data/mrr/"+products["jue"]+"/*/*txt.gz",
  "jue":"/data/obs/site/jue/mrr/l1/*/*/*/*_jue_"+products["jue"]+".txt.gz",
  "ufs":"/data/tosca/mrr/data/*/*gz",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/data/"+products["uf2"]+"/*/*txt.gz",
  "nya":"/data/obs/site/nya/mrr/l1/*/*/*/*_nya_"+products["nya"]+".txt.gz",
  "comble":"/data/obs/campaigns/comble/mrr/l1/*/*/*/*_comble_"+products["comble"]+".txt.gz",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/*/*/*/*_fes_"+products["fes"]+".txt.gz",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/*/*/*/*_and_"+products["and"]+".txt.gz",
}
pathNcs = {
  #"jue":"/data/hatpro/jue/data/mrr/"+products["jue"]+"_nc/%y%m/%y%m%d_jue_"+products["jue"]+".nc",
  "jue":"/data/obs/site/jue/mrr/l1/%Y/%m/%d/%Y%m%d_jue_"+products["jue"]+".nc",
  "ufs":"/data/hatpro/ufs_other/ufs_mrr/data/"+products["ufs"]+"_nc/%y%m/%y%m%d_ufs_"+products["ufs"]+".nc",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/data/"+products["uf2"]+"_nc/%y%m/%y%m%d_uf2_"+products["uf2"]+".nc",
  "nya":"/data/obs/site/nya/mrr/l1/%Y/%m/%d/%Y%m%d_nya_"+products["nya"]+".nc",
  "comble":"/data/obs/campaigns/comble/mrr/l1/%Y/%m/%d/%Y%m%d_comble_"+products["comble"]+".nc",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/%Y/%m/%d/%Y%m%d_fes_"+products["fes"]+".nc",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/%Y/%m/%d/%Y%m%d_and_"+products["and"]+".nc",
}
pathPlots = {
  #"jue":"/data/hatpro/jue/plots/mrr/"+productPng+"/%y%m/%y%m%d_jue_"+productPng+".png",
  "jue":"/data/obs/site/jue/mrr/l1/%Y/%m/%d/%Y%m%d_jue_"+productPng+".png",
  "ufs":"/data/hatpro/ufs_other/ufs_mrr/plots/"+productPng+"/%y%m/%y%m%d_ufs_"+productPng+".png",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/plots/"+productPng+"/%y%m/%y%m%d_uf2_"+productPng+".png",
  "nya":"/data/obs/site/nya/mrr/l1/%Y/%m/%d/%Y%m%d_nya_"+productPng+".png",
  "comble":"/data/obs/campaigns/comble/mrr/l1/%Y/%m/%d/%Y%m%d_comble_"+productPng+".png",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/%Y/%m/%d/%Y%m%d_fes_"+productPng+".png",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/%Y/%m/%d/%Y%m%d_and_"+productPng+".png",
}

logos = {
  "jue":"/home/hatpro/mrr_scripts/uk_joyce_mini.png",
  "ufs":None,
  "uf2":"/home/hatpro/mrr_scripts/uk_mini.png",
  "nya":None,#"/home/hatpro/mrr_scripts/uk_arctic-amplification-logo-ac3.png",
  "comble":None,
  "fes":None,
  "and":None,
}

symboliclLink = {
  #"jue":None,
  "jue":"/home/hatpro/public_html/jue/mrr/level1/%Y/%m/%Y%m%d_jue_"+productPng+".png",
  "ufs":None,
  "uf2":None,
  "nya":"/home/hatpro/public_html/nya/mrr/level1/%Y/%m/%Y%m%d_nya_"+productPng+".png",
  "comble":"/home/hatpro/public_html/comble/mrr/level1/%Y/%m/%Y%m%d_comble_"+productPng+".png",
  "fes":None,
  "and":None,
}

### end settings ###

def mkdir_p(path, parents=-1):
  """make directories with parents
  Make all directories that are given after the `parents`th '/' counted from the end.
  e.g. mkdir_p('/tmp/1/2/3/4/',5) makes everything below `tmp`. It is the same as mkdir_p('/tmp/1/2/3/4',4)
  """
  if parents != int(parents) or parents < -1:
    raise ValueError('`parents` must be larger or equal to -1. given %s' % parents)
  d = '/'.join(path.split('/')[:-parents]) + '/'
  for directory in path.split('/')[-parents:]:
      d = d + directory + '/'
      if not os.path.isdir(d):
          os.mkdir(d)

for site in sites:

  # finding files according to given date range (date_start - date_end):
  files_all = sorted(glob.glob(pathIns[site]))
  files_dates = np.asarray([datetime.datetime.strptime(fn.replace(os.path.dirname(fn) + "/", "")[:8], "%Y%m%d") for fn in files_all])
  files = np.asarray(files_all)[(files_dates >= date_start) & (files_dates <= date_end)].tolist()
#  files = sorted(glob.glob(pathIns[site]))[-1:-noDaysAgo-1:-1]   ### default

  for ff, fIn in enumerate(files):
    try:
      date = re.findall('\d+', fIn.split("/")[-1])[0]
      if len(date) == 6:
        date = '20' + date
      date_time = datetime.datetime.strptime(date, '%Y%m%d')
      ncOut = date_time.strftime(pathNcs[site])
      pngOut = date_time.strftime(pathPlots[site])
      if symboliclLink[site]:
        link_name = date_time.strftime(symboliclLink[site])
      else:
        link_name = ''

      print("output files", date, ncOut, pngOut)

      if skipExisting:
        modificationTime_fIn = os.path.getmtime(fIn)
        ncOut_exists_and_is_older = (os.path.isfile(ncOut) and modificationTime_fIn < os.path.getmtime(ncOut))
        ncOutgz_exists_and_is_older = (os.path.isfile(ncOut+".gz") and modificationTime_fIn < os.path.getmtime(ncOut+".gz"))
        pngOut_exists_and_is_older = (os.path.isfile(pngOut) and modificationTime_fIn < os.path.getmtime(pngOut))
        if (
          (ncOut_exists_and_is_older or ncOutgz_exists_and_is_older) and
          pngOut_exists_and_is_older and
          (link_name == '' or os.path.islink(link_name))
        ):
          print("skipped", date, site)
          continue

      mkdir_p(os.path.dirname(ncOut), 3)
      mkdir_p(os.path.dirname(pngOut), 3)

      #read Data
      mrrAveData = IMProToo.mrrProcessedData(fIn)
      #make netcdf
      mrrAveData.writeNetCDF(ncOut,
        author=contactPersons[site],
        location=siteUnicodeLongs[site],
        institution=institutions[site],
        latitude=siteLatitudes[site],    # information added on 2022-06-08 by A. Walbroel
        longitude=siteLongitudes[site],  # information added on 2022-06-08 by A. Walbroel
        altitude=siteAltitudes[site],    # information added on 2022-06-08 by A. Walbroel
        ncForm = 'NETCDF3_CLASSIC'
      )
      os.system("gzip -f "+ncOut)

      #start  plot
      plotDict = dict()
      plotDict["MRR Z"] = mrrAveData.mrrCapitalZ
      plotDict["MRR RR"] = mrrAveData.mrrRR
      plotDict["MRR W"] = mrrAveData.mrrW

      pLevels = dict()
      pLevels["MRR Z"] = np.arange(-15,40,1)
      pLevels["MRR RR"] = np.arange(0,40,1)
      pLevels["MRR W"] = np.arange(0,12,0.25)

      pLabels = dict()
      pLabels["MRR Z"] = "Z (att. corr.) [dBz]"
      pLabels["MRR RR"] = "RR [mm/h]"
      pLabels["MRR W"] = "W [m/s]"

      #make 2D array with nhours of day, first get seconds of 00:00
      startUnix = calendar.timegm(date_time.timetuple())
      times2D = np.zeros(mrrAveData.shape2D)
      # remove seconds at 00:00 and convert to hours
      times2D.T[:] = (mrrAveData.mrrTimestamps-startUnix)/(60.*60.)
      heights2D = mrrAveData.mrrH.data

      #work around for older data
      if np.all(np.isnan(heights2D[:,-1])):
        heights2D[:,-1] = heights2D[:,-2]+(heights2D[:,-2]-heights2D[:,-3])

      heights2D[np.isnan(heights2D)] = 0

      pTitle =  productLong + date_time.strftime(" %Y-%m-%d ") + siteUnicodeLongs[site]

      fig=plt.figure(figsize=(10, 6))
      noPlots = len(plotDict.keys())
      for pp, pKey in enumerate(['MRR RR', 'MRR Z', 'MRR W']):
        sp = fig.add_subplot(noPlots, 1, pp+1)
        if pp ==0: sp.set_title(pTitle)
        #make plot
        #cf = sp.contourf(times2D, heights2D, plotDict[pKey], levels = pLevels[pKey], extend='both', cmap='jet')
        cf = sp.pcolormesh(times2D,heights2D,plotDict[pKey],vmin = pLevels[pKey][0],vmax = pLevels[pKey][-1], cmap='jet')
        cb = plt.colorbar(cf)
        cb.set_label(pLabels[pKey], fontsize=8)

        sp.set_xlim(0,24)
        sp.set_ylabel("Height [m]")
        sp.set_xticks(range(0,25,3))
        if pp!=noPlots-1:
          sp.set_xticklabels("")
        else:
          sp.set_xlabel("Time in hours UTC")
        sp.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
      fig.subplots_adjust(hspace=0.03,right=0.77)

      #logo
      if logos[site]:
        im = Image.open(logos[site], mode='r')
        im = np.array(im).astype(np.float) / 255
        fig.figimage(im, fig.bbox.xmax-430,0)

      fig.savefig(pngOut,dpi=100)
      print("written: "+pngOut)
      plt.close(fig)

      if link_name != '' and not os.path.islink(link_name):
        mkdir_p(os.path.dirname(link_name), 2)
        print('linking: ln -s "%s" "%s"' % (pngOut, link_name))
        os.symlink(pngOut, link_name)

    except KeyboardInterrupt:
      raise
    except:
      warnings.warn("converting failed: "+fIn)
      traceback.print_exc(file=sys.stdout)

