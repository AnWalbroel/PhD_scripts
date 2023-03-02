# CSSC: Clear Sky Sonde Comparison.
This is a code package to find offsets of microwave radiometer channels onboard the HALO research 
aircraft (conventions are set for HALO-(AC)3) by comparing measured with simulated 
brightness temperatures (TBs) in clear sky conditions. Information about executing the code 
package can be found below.

The code package generates clear sky sonde comparison netCDF4 files for each research flight that
shows the bias, RMSE and correlation (\*TB_comparison\*.nc). Afterwards, netCDF4 files containing the
correction terms that must be applied to the observed microwave radiometer measurements to obtain 
corrected TBs (\*TB_offset_correction\*.nc).


*Python requirements*: tested with Python version **3.10.6**
- numpy: **1.21.5**
- xarray: **0.16.1**
- pandas: **1.3.5**
- matplotlib: **3.5.1**


Data prerequisites:
- One file for each research flight containing all microwave radiometer measurements 
	(all channels: K+V, 119+90, 183)
- BAHAMAS data must be availalbe as one file for each research flight containing at least aircraft
	time, altitude, latitude and longitude, as well as temperature, relative humidity and static 
	air pressure measurements


Assign paths in CSSC_main_unified.py (with a slash "/" at the end of the path):
path_data dictionary must be set for 
- "BAH": BAHAMAS data
- "mwr_concat": HAMP microwave radiometer data
- "radar": HAMP radar data (can also be blank: set the path to "")
- "dropsonde": Dropsonde data
- "dropsonde_rep": Path where output of dropsonde_gap_filler is to be saved to
- "sst": Path where data downloaded with download_sst_data is to be saved to
- "dropsonde_sim": Path where simulated TBs from dropsonde data (fwd_sim_dropsondes_to_TB) is to be saved to
- "cssc_output": Path where output of TB_comparison and get_TB_offsets is to be saved to

path_plot: string indicating the path where plots are saved to


Set settings in the dictionary set_dict:
- Assign dropsonde type in set_dict['sonde_dataset_type'] as string, valid options: "raw"
- Assign a height grid to which dropsonde data is interpolated in set_dict['sonde_height_grid'] as 
	numpy array of floats
- Assign latitude boundaries in set_dict['sst_lat'] as a list of integers (or float) with the first
		entry being the southern and the second entry the northern boundary (southern hemisphere 
		with a negative sign). Valid range: [-90, 90]
		E.g. for latitude boundaries being 20°N and 50.5°N: set_dict['sst_lat'] = [20, 50.5]
- Assign longitude boundaries in set_dict['sst_lon'] as a list of integers (or float) with the 
		first entry being the western and the second entry the eastern boundary (west of prime 
		meridian (0°E) with a negative sign). Valid range: [-180, 179.9]
		E.g. for latitude boundaries being 40°W and 10.5°E: set_dict['sst_lon'] = [-40, 10.5]
- Assign a start date in set_dict['start_date'] and an end date in set_dict['end_date'] to create
		a daterange via pandas daterange method. The daterange will cover all days from start 
		to end date. Both start_date and end_date must be given as a string, formatted as 
		"YYYY-MM-DD".
		E.g. if we need SST data from 25th January 2020 until 2nd March 2020:
		set_dict['start_date'] = "2020-01-25"
		set_dict['end_date'] = "2020-03-02"


1.	Prepare the dropsonde files via "dropsonde_gap_filler":
	Small gaps in the profiles will be filled via linear interpolation, and, if necessary,
	extrapolation at the top and bottom will be performed if data is missing and the data void is 
	not too large. 


2.	Two options: Manually select and download SST data or assign location and time boundaries for an
	automated download.

	Manual selection and download (last update 2021-03-01):
	Before forward simulating the dropsondes we may want to include sea surface temperature data 
	from "https://podaac.jpl.nasa.gov/dataset/CMC0.1deg-CMC-L4-GLOB-v3.0" using the OPENDAP tool. 
	When having navigated to a desired date, click on the ".nc" file (not ".nc.md5") where you will
	be given the opportunity to choose variables. If hard drive space is not your concern you may 
	want to check all variables and download the global data. If disk space is sparse you may
	want to check 'time', 'lat', 'lon', 'analyzed_sst' and 'mask' only and limit the variables to 
	the location of the dropsondes (make sure that the whole region is covered). Once you have 
	selected the variables (and region) you click on "Get as NetCDF 4" in the row "Actions". As 
	soon as the file is downloaded move it to your desired path.

	Automated version: "download_sst_data"
	The CMC0.1deg-CMC-L4-GLOB-v3.0 data will be automatically downloaded and saved to a specified 
	path. Latitude and longitude boundaries must be set and the dates of each required day must 
	be given.


3.	Forward simulate dropsonde data to HAMP frequency TB using "fwd_sim_dropsondes_to_TB". Those 
	profiles that still contain missing data after "dropsonde_gap_filler" will be skipped during 
	execution because PAMTRA cannot handle NaNs. Only the nadir looking and polarization-averaged
	TBs will be saved.


4.	Create a comparison of the measured brightness temperatures by HAMP with the forward simulated
	brightness temperatures from dropsondes using "TB_comparison". Some basic plots showing
	e.g. a scatterplot of measured and simulated brightness temperatures will be created by 
	default. 

	It produces netCDF4 files containing simulated TBs from dropsonde data, as well as
	mean and standard deviation of measured TBs around dropsonde launches. Also clear sky filtering
	masks and metrics to compare measured and simulated TBs are indcluded.


5.	Compute the offset (or linear correction) that needs to be applied on measured TB data to 
	obtain offset corrected TBs in "get_TB_offsets". The correction terms will be computed based on
	the output of "TB_comparison".

	It produces netCDF4 files containing offsets and slopes of a correction based on a linear fit,
	but also directly a bias which can be applied if there were an insufficient amount of clear sky
	dropsondes to have a reliable linear fit.


6.	(optional): "visualise_TB_offsets" visualises the output of "get_TB_offsets" and 
	"TB_comparison" by plotting corrected TB measurements in relation to simulated TBs in a 
	scatter plot.
