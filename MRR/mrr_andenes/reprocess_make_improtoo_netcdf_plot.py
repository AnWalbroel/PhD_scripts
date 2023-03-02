# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import sys
from PIL import Image
import warnings
import traceback
import calendar
import datetime
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'
import matplotlib.pyplot as plt
#add containing folder to path to load IMProToo
sys.path.append("/".join(sys.argv[0].split("/")[0:-1]))
import IMProToo

###settings ###
noDaysAgo = 90
skipExisting = False


#sites = ["ufs","uf2","jue"]

date_start = datetime.datetime.strptime(sys.argv[1], "%Y%m%d")
date_end = datetime.datetime.strptime(sys.argv[2], "%Y%m%d")
sites = sys.argv[3:]
#sites=["ant"]
siteLongs = {
  "jue":"FZ Juelich",
  "ufs":"UFS Schneefernerhaus MRR DLR",
  "uf2":"UFS Schneefernerhaus MRR2 UK",
  "ant":"Antarctica Princess Elisabeth",
  "nya":"Ny-Alesund",
  "comble":"Bjornoya",
  "fes":"FESSTVaL",
  "and":"Andenes",
}
siteUnicodeLongs = {
  "jue":u"FZ Jülich",
  "ufs":"UFS Schneefernerhaus MRR DLR",
  "uf2":"UFS Schneefernerhaus MRR2 UK",
  "ant":"Antarctica Princess Elisabeth",
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

contactPersons= {
  "jue":u"Bernhard Pospichal (bernhard.pospichal@uni-koeln.de)",
  "ufs":"Martin Hagen (martin.hagen@dlr.de)",
  "uf2":"NN",
  "nya":u"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
  "comble":u"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
  "fes":"Andreas Walbroel (a.walbroel@uni-koeln.de)",
  "and":u"Kerstin Ebell (kebell@meteo.uni-koeln.de)",
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

product="mrr_improtoo_"+IMProToo.__version__.replace(".","-")
productPng = "mrr_improtoo"
productLong ="MRR IMProToo "+IMProToo.__version__+" (60s)"


pathIns = {
  #"jue":"/data/hatpro/jue/data/mrr/mrr_raw/*/*txt.gz",
  "jue":"/data/obs/site/jue/mrr/l1/*/*/*/*_jue_mrr_raw.txt.gz",
  "ufs":"/data/tosca/mrr/data_raw/*/*gz",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/data/mrr_raw/*/*gz",
  "ant":"/data/hatpro/ant/mrr/mrr_raw/*",
  "nya":"/data/obs/site/nya/mrr/l1/*/*/*/*_nya_mrr_raw.txt.gz",
  "comble":"/data/obs/campaigns/comble/mrr/l1/*/*/*/*_comble_mrr_raw.txt.gz",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/*/*/*/*_fes_mrr_raw.txt.gz",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/*/*/*/*_and_mrr_raw.txt.gz",
}
pathNcs = {
  #"jue":"/data/hatpro/jue/data/mrr/"+product+"/%y%m/%y%m%d_jue_"+product+".nc",
  "jue":"/data/obs/site/jue/mrr/l1/%Y/%m/%d/%Y%m%d_jue_"+product+".nc",
  "ufs":"/data/hatpro/ufs_other/ufs_mrr/data/"+product+"/%y%m/%y%m%d_ufs_"+product+".nc",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/data/"+product+"/%y%m/%y%m%d_uf2_"+product+".nc",
  "ant":"/data/hatpro/ant/mrr/mrr_improtoo_0.99/%y%m/%y%m%d_uf2_"+product+".nc",
  "nya":"/data/obs/site/nya/mrr/l1/%Y/%m/%d/%Y%m%d_nya_"+product+".nc",
  "comble":"/data/obs/campaigns/comble/mrr/l1/%Y/%m/%d/%Y%m%d_comble_"+product+".nc",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/%Y/%m/%d/%Y%m%d_fes_"+product+".nc",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/%Y/%m/%d/%Y%m%d_and_"+product+".nc",
}
pathPlots = {
  #"jue":"/data/hatpro/jue/plots/mrr/"+productPng+"/%y%m/%y%m%d_jue_"+productPng+".png",
  "jue":"/data/obs/site/jue/mrr/l1/%Y/%m/%d/%Y%m%d_jue_"+productPng+".png",
  "ufs":"/data/hatpro/ufs_other/ufs_mrr/plots/"+productPng+"/%y%m/%y%m%d_ufs_"+productPng+".png",
  "uf2":"/data/hatpro/ufs_other/ufs_mrr2/plots/"+productPng+"/%y%m/%y%m%d_uf2_"+productPng+".png",
  "ant":"/data/hatpro/ant/mrr/plots/"+productPng+"/%y%m/%y%m%d_uf2_"+productPng+".png",
  "nya":"/data/obs/site/nya/mrr/l1/%Y/%m/%d/%Y%m%d_nya_"+productPng+".png",
  "comble":"/data/obs/campaigns/comble/mrr/l1/%Y/%m/%d/%Y%m%d_comble_"+productPng+".png",
  "fes":"/data/obs/campaigns/FESSTVaL/mrr/l1/%Y/%m/%d/%Y%m%d_fes_"+productPng+".png",
  "and":"/data/obs/campaigns/comble/and/mrr/l1/%Y/%m/%d/%Y%m%d_and_"+productPng+".png",
}

logos = {
  "jue":"/home/hatpro/mrr_scripts/uk_joyce_mini.png",
  "ufs":None,
  "uf2":"/home/hatpro/mrr_scripts/uk_mini.png",
  "ant":None,
  "nya":None, #"/home/hatpro/mrr_scripts/uk_arctic-amplification-logo-ac3.png",
  "comble":None,
  "fes":None,
  "and":None,
}

pngLatest = {
  "jue":"/home/hatpro/public_html/jue/mrr/"+productPng+"/latest_"+"jue"+"_"+productPng+".png",
  "ufs":"/data/hatpro/ufs_other/ufs_mrr/plots/"+productPng+"/latest_"+"ufs"+"_"+productPng+".png",
  "uf2":False,
  "ant":"/data/hatpro/ant/mrr/plots/"+productPng+"/latest_"+"ant"+"_"+productPng+".png",
  "nya":False,
  "comble":False,
  "fes":False,
  "and":False,
}

symboliclLink = {
  #"jue":None,
  "jue":"/home/hatpro/public_html/jue/mrr/level1/%Y/%m/%Y%m%d_jue_"+productPng+".png",
  "ufs":None,
  "uf2":None,
  "ant":None,
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
#  files = sorted(glob.glob(pathIns[site]))[-1:-noDaysAgo-1:-1]      ### default

  for ff,fIn in enumerate(files):
    try:
      date = re.findall('\d+', fIn.split("/")[-1])[0]
      if len(date) == 6:
        date = '20' + date
      date_time = datetime.datetime.strptime(date, '%Y%m%d')
      # Use IMProToo 0.101.1 starting in 2021
#      if date_time < datetime.datetime(2021, 1, 1) and IMProToo.__version__ >= '0.101.1':
#          print('Skip files for', date_time, ". (Don't use MProToo.__version__ %s before 2021-01-01)" % IMProToo.__version__)
#          continue

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
      mrrRawData = IMProToo.mrrRawData(fIn)

      #calculate improtoo data
      mrrSnow = IMProToo.MrrZe(mrrRawData)
      mrrSnow.averageSpectra(60)
      mrrSnow.co["ncCreator"] = contactPersons[site]
      mrrSnow.co["ncLocation"] = siteUnicodeLongs[site]
      mrrSnow.co["ncInstitution"] = institutions[site]
      mrrSnow.co["ncDescription"] = "MRR data from "+siteLongs[site]+" ("+site+"), "+date[2:]
      mrrSnow.co["Latitude"] = siteLatitudes[site]
      mrrSnow.co["Longitude"] = siteLongitudes[site]
      mrrSnow.co["Altitude"] = siteAltitudes[site]
      mrrSnow.rawToSnow()

      #make netcdf
      mrrSnow.writeNetCDF(ncOut, ncForm = 'NETCDF3_CLASSIC')
      os.system("gzip -f %s"%ncOut)
      print("netcdf written:%s.gz"%ncOut)

      #start  plot
      plotDict = dict()
      plotDict["MRR Ze"] = np.ma.masked_array(mrrSnow.Ze,mrrSnow.Ze==-9999)
      plotDict["MRR W"] = np.ma.masked_array(mrrSnow.W,mrrSnow.W==-9999)
      plotDict["MRR Spectral Width"] = np.ma.masked_array(mrrSnow.specWidth,mrrSnow.specWidth==-9999)
      plotDict["MRR Noise"] = 10*np.log10(np.ma.masked_array(mrrSnow.etaNoiseAve,mrrSnow.etaNoiseAve==-9999)) # Noise expressed in dBz

      pLevels = dict()
      pLevels["MRR Ze"] = np.arange(-15,40,1)
      pLevels["MRR Noise"] = np.arange(-20, -5, 0.25) # Noise expressed in dBz
      pLevels["MRR W"] = np.arange(-4,10,0.25)
      pLevels["MRR Spectral Width"] = np.arange(0,1.5,0.03)

      pLabels = dict()
      pLabels["MRR Ze"] = "Ze [dBz]"
      pLabels["MRR Noise"] = 'Mean Spec. Noise [dBz]' # Noise expressed in dBz
      pLabels["MRR W"] = "W [m/s]"
      pLabels["MRR Spectral Width"] = "Spectral Width [m/s]"

      #make 2D array with nhours of day, first get seconds of 00:00
      startUnix = calendar.timegm(date_time.timetuple())
      times2D = np.zeros(mrrSnow._shape2D)
      # remove seconds at 00:00 and convert to hours
      times2D.T[:] = (mrrSnow.time-startUnix)/(60.*60.)
      heights2D = mrrSnow.H.data
      heights2D[np.isnan(heights2D)] = -9999.

      pTitle =  productLong + date_time.strftime(" %Y-%m-%d ") + siteUnicodeLongs[site]

      fig=plt.figure(figsize=(10, 6))
      noPlots = len(plotDict.keys())
      for pp, pKey in enumerate(['MRR Ze', 'MRR Spectral Width', 'MRR W',"MRR Noise"]):
        #print pp, pKey
        sp = fig.add_subplot(noPlots,1,pp+1)
        if pp ==0: sp.set_title(pTitle)
        #make plot
        #cf = sp.contourf(times2D,heights2D,plotDict[pKey],levels = pLevels[pKey],extend='both')
        cf = sp.pcolormesh(times2D,heights2D,plotDict[pKey],vmin = pLevels[pKey][0],vmax = pLevels[pKey][-1], cmap='jet')
        cb = plt.colorbar(cf,ticks=pLevels[pKey][::8])
        cb.set_label(pLabels[pKey], fontsize=8)

        sp.set_xlim(0,24)
        sp.set_ylim((np.min(heights2D[heights2D!=-9999]),np.max(heights2D)))
        sp.set_ylabel("Height [m]")
        sp.set_xticks(range(0,25,3))
        sp.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        if pp!=noPlots-1:
          sp.set_xticklabels("")
        else:
          sp.set_xlabel("Time in hours UTC")
      fig.subplots_adjust(hspace=0.03,right=0.77)

      #logo
      if logos[site]:
        im = Image.open(logos[site], mode='r')
        width = im.size[0]
        im = np.array(im).astype(np.float) / 255
        fig.figimage(im, fig.bbox.xmax-430,0)

      fig.savefig(pngOut,dpi=100)
      if ff == 0 and pngLatest[site]:
        fig.savefig(pngLatest[site],dpi=100)
      print("written: "+pngOut)
      plt.close(fig)

      if link_name != '' and not os.path.islink(link_name):
        mkdir_p(os.path.dirname(link_name), 2)
        print('linking: ln -s "%s" "%s"' % (pngOut, link_name))
        os.symlink(pngOut, link_name)

    except KeyboardInterrupt:
      raise
    except:
      raise
      warnings.warn("converting failed: "+fIn)
      traceback.print_exc(file=sys.stdout)
