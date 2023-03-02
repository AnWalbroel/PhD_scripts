# CSSC: Clear Sky Sonde Comparison.
This is a code package to find offsets of microwave radiometer
channels onboard the HALO research aircraft by comparing measured with simulated brightness
temperatures in clear sky conditions. Information about executing the code package can be found below.

Guide to generate clear sky sonde comparison netCDF4 file that shows the bias, RMSE and correlation.
Lines that are most likely to be edited by the user in the code are marked with "##################".

1. 	(This step can be skipped when using MWR data in uniform file format (Konow et al. 2019))
	Make sure that your MWR data is concatenated, which is usually not the case. The MWR data for each
	module (KV, 11990, 183) is saved separately as <current_date>.BRT.NC (current_date in YYMMDD,
	e.g. "200119" being the 19th January 2020) file. Other file extensions like .ATN.NC (attenuation)
	are also stored in the module folder but irrelevant for the clear sky sonde comparison.

	To concatenate the MWR data (glueing all modules together and getting them on a mutual time axis)
	use the code "concat_mwr.py":
	- Assign the base path of the mwr in "mwr_base_path = ". The 'base path' is the directory in which
		folders for each operating day of the MWR are located
		(e.g.
				abc@xyz:base_path$ ls
				20200118  20200122  20200126  20200129  20200131  20200205
		).
		Inside these day folders you either directly find the module folders "KV", "11990" and "183" or
		another subfolder may appear like "radiometer" so that you have:
		abc@xyz:base_path/20200118/radiometer/KV$ or another module instead of "KV". Inside that folder
		the .BRT.NC files should be located. Example: mwr_base_path = "/data/hamp/flights/EUREC4A/"
	- Make sure that "day_folders" is a list that contains folder paths of dates (e.g.
		day_folders = sorted(glob.glob(mwr_base_path + "*")) is not possible if
		mwr_base_path = "/data/hamp/flights/EUREC4A/" because there are other folders inside mwr_base_path).
		Example: day_folders = sorted(glob.glob(mwr_base_path + "*"))[1:17]
	- Assign an output path where the concatenated (v01) mwr files shall be saved to: "mwr_concat_path = "
	- WITHIN concat_mwr.py, make sure that "folder_modules = ", when the loop over the "day_folders" has
		started, contains the path of all modules as a dictionary so that inside "folder_modules" the .BRT.NC
		files can be read in later, e.g.
		folder_modules = {
			'KV': folder + "/radiometer/KV/",
			'11990': folder + "/radiometer/11990/",
			'183': folder + "/radiometer/183/"
			}
		with folder being one of the day folders ("base_path/20200118/").
	- Make sure that "folder_date = " is the date of the current folder (in YYMMDD). For the example
		above the folder_date should be "200118" by slicing the string of the current day folder
		(e.g. "folder_date = folder[-6:]"). Usually this doesn't have to be changed but make sure it is
		assigned correctly. Otherwise the execution is stopped due to errors.
	- Wait for the execution to finish.

2.	Prepare the dropsonde files via "dropsonde_raw_gap_filler.py":
	Small gaps in the profiles will be filled via linear interpolation and, if necessary,
	extrapolation at the top and bottom will be performed if data is missing and the data void is not
	too large. Here it is possible to choose different datasets of dropsondes (e.g. raw dropsondes,
	JOANNE dataset, ...); all options are listed below:

	- Assign "path_raw_sondes = " to the path where the D<date>_PQC.nc (date in YYYYMMDD_hhmmss (e.g.
		D20200207_175237_PQC.nc)) are located.
	- Assign "path_halo_dropsonde = " where the gap-filled version (v01) shall be saved.
	- Assign the dataset description in "dropsonde_dataset" and string. Valid options are "raw" and
		"joanne_level_3". Default is "raw".
	- Assign "path_BAH_data = " as the path where BAHAMAS measurements are stored because BAHAMAS
		measurements can partly be used as extrapolation target. If BAHAMAS data is not to be used
		or not available, set "path_BAH_data = None" , comment it out or remove it from the
		call of the function "run_dropsonde_raw_gap_filler".
	- Wait for the execution to finish.

3.	Two options: Manually select and download SST data or assign location and time boundaries for an
	automated download (be aware that roughly 7 MB diskspace per global SST (=per day) is needed).

	Manual selection and download:
	Before forward simulating the dropsondes we may want to include sea surface temperature data from
	"https://podaac.jpl.nasa.gov/dataset/CMC0.1deg-CMC-L4-GLOB-v3.0" using the OPENDAP tool. When having
	navigated to a desired date, click on the ".nc" file (not ".nc.md5") where you will be given the
	opportunity to choose variables. If hard drive space is not your concern you may want to check all
	variables and download the global data. If hard drive space is sparse you may want to check 'time',
	'lat', 'lon', 'analyzed_sst' and 'mask' only and limit the variables to the location of the
	dropsondes (make sure that the whole region is covered). Once you have selected the variables (and
	region) you click on "Get as NetCDF 4" in the row "Actions". As soon as the file is downloaded move
	it to your desired path.

	Automated version: "get_sst_data.py"
	The CMC0.1deg-CMC-L4-GLOB-v3.0 data will be automatically downloaded and saved to a specified path.
	Latitude and longitude boundaries must be set and the dates of each required day must be given.

	- Assign "path_sst = " to the path where the SST data shall be saved to.
	- Assign latitude boundaries in "lat_bound = " as a list of integers (or float) with the first
		entry being the southern and the second entry the northern boundary (southern hemisphere with
		a negative sign). Valid range: [-90, 90]
		E.g. for latitude boundaries being 20°N and 50.5°N: lat_bound = [20, 50.5]
	- Assign longitude boundaries in "lon_bound = " as a list of integers (or float) with the first
		entry being the western and the second entry the eastern boundary (locations to the west
		of the prime meridian (0°E) with a negative sign). Valid range: [-180, 179.9]
		E.g. for longitude boundaries being 40°W and 10°E: lon_bound = [-40, 10]
	- Assign a start date in "start_date = " and an end date in "end_date = " to create a daterange via
		pandas daterange method. The daterange will cover all days from start to end date. Both
		start_date and end_date must be given as a string, formatted as "YYYY-MM-DD".
		E.g. if we need SST data from 25th January 2020 until 2nd March 2020:
		start_date = "2020-01-25"
		end_date = "2020-03-02"


4.	Forward simulate dropsonde data to HAMP frequency brightness temperatures using
	"HALO_raw_dropsonde_to_TB.py". Those profiles that still contain missing data after
	"dropsonde_raw_gap_filler.py" will be skipped during execution because PAMTRA cannot handle NaNs.
	Only the nadir looking and polarization-averaged brightness temperatures will be saved.

	- Assign a path where the PAMTRA output shall be saved to in "path_pam_ds = ".
	- Wait for the execution to finish (depending on the amount of dropsondes, this may take a while...
		I suggest a coffee break).

5.	Create a comparison of the measured brightness temperatures by HAMP with the forward simulated
	brightness temperatures from dropsondes using "TB_statistics_raw.py". Some basic plots showing
	e.g. a scatterplot of measured and simulated brightness temperatures can be included.

	- If uniform MWR data is used (and step 1 was skipped) set "mwr_concat_path =" pointing to the
		merged uniform data. Uniform MWR data file is detected by the importer if the path includes
		"uniform" or "unified". If MWR data in "v0.8" is used, a correction of timings is applied.
	- Set a path where the comparison netCDF4 file shall be saved to in "out_path = ".
	- Set a path where the plots shall be saved to in "plot_path = ".
	- Set a name for the scatterplot WITHOUT file extension (will be saved as .png and .pdf) in
		"scatterplot_name = ".
	- Set a name for the bias evolution plot WITHOUT file extension (will be saved as .png and .pdf)
		in "bias_ev_plotname = ".
	- Set a name for the brightness temperature comparison netCDF4 output file WITHOUT file
		extension in "output_filename = ".
	- Wait for the execution to finish.
