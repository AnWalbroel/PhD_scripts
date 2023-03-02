;
;******************
PRO PREPARE_RPG_TRAINING_DATA
;******************

path_data = "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/"

filename = path_data + 'L4_NP_arctic_v2_all.ball'

read_binary_data_RPG, filename, data


; truncate dataset: erase unwanted frequencies and angles:
; frq_idx = [201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 261, 277]    ; for monochr.
frq_idx = [200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,$
           218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,$
           261,276,277,278]
angle_idx = 0
freq_used = data.FREQUENCIES(frq_idx)
angle_used = data.ANGLES(angle_idx)

Z_used = data.DATA_GRIDDED(*,0,*)
T_used = data.DATA_GRIDDED(*,1,*)
RH_used = data.DATA_GRIDDED(*,2,*)
ABSHUM_used = data.DATA_GRIDDED(*,3,*)

yy = data.DATA_ADDITIONAL(0,*)
mm = data.DATA_ADDITIONAL(1,*)
dd = data.DATA_ADDITIONAL(2,*)
hh = data.data_ADDITIONAL(3,*)
IWV_used = data.DATA_ADDITIONAL(7,*)
LWP_used = data.DATA_ADDITIONAL(6,*)
T_s_used = data.DATA_ADDITIONAL(8,*)
RHs_used = data.DATA_ADDITIONAL(9,*)
P_s_used = data.DATA_ADDITIONAL(10,*)


TBs_used = data.DATA_RTM(2,frq_idx,angle_idx,*)


; manual bandpass filter:
print, "Applying a manual bandpass filter...."

TBs_bp = TBs_used
TBs_bp(0,0,0,*) = (TBs_used(0,0,0,*) + TBs_used(0,1,0,*) + TBs_used(0,2,0,*))/3
TBs_bp(0,1,0,*) = (TBs_used(0,3,0,*) + TBs_used(0,4,0,*) + TBs_used(0,5,0,*))/3
TBs_bp(0,2,0,*) = (TBs_used(0,6,0,*) + TBs_used(0,7,0,*) + TBs_used(0,8,0,*))/3
TBs_bp(0,3,0,*) = (TBs_used(0,9,0,*) + TBs_used(0,10,0,*) + TBs_used(0,11,0,*))/3
TBs_bp(0,4,0,*) = (TBs_used(0,12,0,*) + TBs_used(0,13,0,*) + TBs_used(0,14,0,*))/3
TBs_bp(0,5,0,*) = (TBs_used(0,15,0,*) + TBs_used(0,16,0,*) + TBs_used(0,17,0,*))/3
TBs_bp(0,6,0,*) = (TBs_used(0,18,0,*) + TBs_used(0,19,0,*) + TBs_used(0,20,0,*))/3
TBs_bp(0,7,0,*) = (TBs_used(0,21,0,*) + TBs_used(0,22,0,*) + TBs_used(0,23,0,*))/3
TBs_bp(0,8,0,*) = (TBs_used(0,24,0,*) + TBs_used(0,25,0,*) + TBs_used(0,26,0,*))/3
TBs_bp(0,9,0,*) = (TBs_used(0,27,0,*) + TBs_used(0,28,0,*) + TBs_used(0,29,0,*))/3
TBs_bp(0,10,0,*) = (TBs_used(0,30,0,*) + TBs_used(0,31,0,*) + TBs_used(0,32,0,*))/3
TBs_bp(0,11,0,*) = (TBs_used(0,33,0,*) + TBs_used(0,34,0,*) + TBs_used(0,35,0,*))/3
TBs_bp(0,12,0,*) = TBs_used(0,36,0,*)
TBs_bp(0,13,0,*) = (TBs_used(0,37,0,*) + TBs_used(0,38,0,*) + TBs_used(0,39,0,*))/3
TBs_bp = TBs_bp(0,0:13,0,*)
frq_idx = [1,4,7,10,13,16,19,22,25,28,31,34,36,38]
freq_used = freq_used(frq_idx)


; double side band average:
print, "Applying double side band average to G band channels...."

TBs_dsba = TBs_bp
TBs_dsba(0,0,0,*) = (TBs_bp(0,5,0,*) + TBs_bp(0,6,0,*))/2
TBs_dsba(0,1,0,*) = (TBs_bp(0,4,0,*) + TBs_bp(0,7,0,*))/2
TBs_dsba(0,2,0,*) = (TBs_bp(0,3,0,*) + TBs_bp(0,8,0,*))/2
TBs_dsba(0,3,0,*) = (TBs_bp(0,2,0,*) + TBs_bp(0,9,0,*))/2
TBs_dsba(0,4,0,*) = (TBs_bp(0,1,0,*) + TBs_bp(0,10,0,*))/2
TBs_dsba(0,5,0,*) = (TBs_bp(0,0,0,*) + TBs_bp(0,11,0,*))/2
TBs_dsba(0,6,0,*) = TBs_bp(0,12,0,*)
TBs_dsba(0,7,0,*) = TBs_bp(0,13,0,*)
TBs_dsba = TBs_dsba(0,0:7,0,*)
freq_dsba = freq_used(6:13)



; interpolate gridded data to new grid for mvr_mwr.pro:
; to compute pressure, I use the barometric height formula (Holton, p.21)
print, "Interpolating meteorological variables to new grid...."

Z_new = [0, 50, 100, 150, 200, 250, 325, 400, 475, 550, 625, 700, 800,$
         900, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2250, 2500, 2750, 3000,$
         3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5500, 6000, 6500, 7000,$
         7500, 8000, 8500, 9000, 9500, 10000, 15000, 20000, 25000, 30000]
; need to use a loop because my limited IDL knowledge keeps me from using more advanced methods:
n_hgt_new = N_ELEMENTS(Z_new)
T_ip = FLTARR(n_hgt_new, 1, data.NOF_ENTRIES)
RH_ip = FLTARR(n_hgt_new, 1, data.NOF_ENTRIES)
ABSHUM_ip = FLTARR(n_hgt_new, 1, data.NOF_ENTRIES)
P_ip = FLTARR(n_hgt_new, 1, data.NOF_ENTRIES)

R_d = 287.04
g0 = 9.80665
R_d_g0 = R_d/g0
FOR I = 0, data.NOF_ENTRIES-1 DO BEGIN
	T_ip(*,0,I) = INTERPOL(T_used(*,0,I), Z_used(*,0,I), Z_new)
	RH_ip(*,0,I) = INTERPOL(RH_used(*,0,I), Z_used(*,0,I), Z_new)
	ABSHUM_ip(*,0,I) = INTERPOL(ABSHUM_used(*,0,I), Z_used(*,0,I), Z_new)

	; correct interpolation errors: set humidity to 0 after 10 km altitude:
	; and set T to the value at 10 km
	RH_ip(43:46,0,I) = 0
	ABSHUM_ip(43:46,0,I) = 0
	T_ip(43:46,0,I) = T_ip(42,0,I)

	H = R_d_g0 * MEAN(T_ip(0:42,0,I))
	P_ip(*,0,I) = P_s_used(I)*EXP(-Z_new / H)
ENDFOR


; create date vector: temporarily convert to string, then back to integer
yy = ROUND(yy)
mm = ROUND(mm)
dd = ROUND(dd)
hh = ROUND(hh)

;yy_string = STRING(yy)
;mm_string = STRING(mm)
;dd_string = STRING(dd)
;hh_string = STRING(hh)
date_string = STRARR(data.NOF_ENTRIES)
FOR I = 0, data.NOF_ENTRIES-1 DO BEGIN
	yy_i_string = STRING(yy(I), FORMAT='(I04)')
	mm_i_string = STRING(mm(I), FORMAT='(I02)')
	dd_i_string = STRING(dd(I), FORMAT='(I02)')
	hh_i_string = STRING(hh(I), FORMAT='(I02)')
	date_string(I) = yy_i_string + mm_i_string + dd_i_string + hh_i_string
ENDFOR
date_int = LONG(date_string)


; reshape arrays:
; T_ip, ABSHUM_ip, P_ip, T_s, ABSHUM_s, P_s, IWV, LWP?, TBs
print, "Reshaping arrays...."

T_done = REFORM(T_ip)							; now in K
ABSHUM_done = 0.001*REFORM(ABSHUM_ip)					; now in kg m^-3
P_done = REFORM(P_ip)					; now in Pa
T_s_done = REFORM(T_s_used)						; now in K
RH_s_done = 0.01*REFORM(RHs_used)					; now in (1)
P_s_done = REFORM(P_s_used)						; now in Pa
IWV_done = REFORM(IWV_used)						; now in kg m^-2
LWP_done = REFORM(LWP_used, [1,data.NOF_ENTRIES])		; now in kg m^-2
TBs_done = REFORM(TBs_dsba, [N_ELEMENTS(freq_dsba),1,1,data.NOF_ENTRIES]) ; now in K

; convert RH_s to ABSHUM_s:
R_v = 461.5
e_sat_water = 100*1013.246 * 10^(-7.90298*(373.16/T_s_done-1) + 5.02808*ALOG10(373.16/T_s_done) - 1.3816e-7*(10^(11.344*(1-T_s_done/373.16))-1) + 8.1328e-3 * (10^(-3.49149*(373.16/T_s_done-1))-1))
ABSHUM_s_done = RH_s_done*e_sat_water / (R_v * T_s_done)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; write loaded data to netcdf:
; many unused dimensions and variables will be created for the purpose
; to run mvr_mwr.pro without problems
print, "Start writing into netCDF...."

; split into yearly data:
print, "Splitting into yearly files...."

years = LONG(STRMID(date_string, 0, 4))
min_year = MIN(years)
max_year = MAX(years)

FOR I = min_year, max_year DO BEGIN
	ix_year = WHERE(years EQ I)		; indices where the date equals the currently selected year
	n_ix_year = N_ELEMENTS(ix_year)		; number of dates for the current year

	; truncate data to current year:
	date_save = date_int(ix_year)
	T_save = T_done(*, ix_year)
	ABSHUM_save = ABSHUM_done(*, ix_year)
	P_save = P_done(*, ix_year)
	T_s_save = T_s_done(ix_year)
	ABSHUM_s_save = ABSHUM_s_done(ix_year)
	P_s_save = P_s_done(ix_year)
	IWV_save = IWV_done(ix_year)
	LWP_save = LWP_done(*, ix_year)
	TBs_save = TBs_done(*, *, *, ix_year)
	

	; save the file:
	station_id = 'pol'
	outfile = path_data + 'combined/' + 'rt_' + station_id + '_v01_' + STRING(I, FORMAT='(I04)') + '.nc'	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	outfile_id = NCDF_CREATE(outfile, /clobber) ; clobber acts to erase existing files

	; define dimensions:
	date_dim_id = NCDF_DIMDEF(outfile_id, 'n_date', /UNLIMITED)
	str3_dim_id = NCDF_DIMDEF(outfile_id, 'string3', 3)  ; for station id: e.g. 'pol', gas abs model, ...
	str42_dim_id = NCDF_DIMDEF(outfile_id, 'string42', 42) ; for 'path to radiosonde'
	freq_dim_id = NCDF_DIMDEF(outfile_id, 'n_frequency', 8)
	angl_dim_id = NCDF_DIMDEF(outfile_id, 'n_angle', 1)
	hght_dim_id = NCDF_DIMDEF(outfile_id, 'n_height', 47)
	str4_dim_id = NCDF_DIMDEF(outfile_id, 'string4', 4)  ; for clear sky or cloudy description
	str5_dim_id = NCDF_DIMDEF(outfile_id, 'string5', 5)  ; for line width specification, ...
	str15_dim_id = NCDF_DIMDEF(outfile_id, 'string15', 15) ; for air mass corr. specification (here: unknown or not applied)
	nwdm_dim_id = NCDF_DIMDEF(outfile_id, 'n_wet_delay_models', 12) ; number of wet delay models
	cloud_model_dim_id = NCDF_DIMDEF(outfile_id, 'n_cloud_model', 1)
	cloud_max_dim_id = NCDF_DIMDEF(outfile_id, 'n_cloud_max', 15) ; max number of clouds?


	; define variables:
	station_id_id = NCDF_VARDEF(outfile_id, 'station_id', [str3_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, station_id_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, station_id_id, 'long_name', 'radiosonde station identifier'

	data_path_to_rs_id = NCDF_VARDEF(outfile_id, 'data_path_to_rs', [str42_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, data_path_to_rs_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, data_path_to_rs_id, 'long_name', 'data path to radiosonde data used'

	rh_thres_id = NCDF_VARDEF(outfile_id, 'rh_thres', /FLOAT)
	NCDF_ATTPUT, outfile_id, rh_thres_id, 'units', '1'
	NCDF_ATTPUT, outfile_id, rh_thres_id, 'long_name', 'rel humidity threshold for cloud detection (mod. adiabatic & Decker models)'

	height_above_sea_level_id = NCDF_VARDEF(outfile_id, 'height_above_sea_level', /FLOAT)

	cap_height_above_sea_level_id = NCDF_VARDEF(outfile_id, 'cap_height_above_sea_level', /FLOAT)
	NCDF_ATTPUT, outfile_id, cap_height_above_sea_level_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, cap_height_above_sea_level_id, 'long_name', 'height of radiosonde capping above MSL'

	frequency_id = NCDF_VARDEF(outfile_id, 'frequency', [freq_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, frequency_id, 'units', 'GHz'
	NCDF_ATTPUT, outfile_id, frequency_id, 'long_name', 'microwave radiometer frequency'

	elevation_angle_id = NCDF_VARDEF(outfile_id, 'elevation_angle', [angl_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, elevation_angle_id, 'units', 'degree'
	NCDF_ATTPUT, outfile_id, elevation_angle_id, 'long_name', 'microwave radiometer elevations angle'

	height_grid_id = NCDF_VARDEF(outfile_id, 'height_grid', [hght_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, height_grid_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, height_grid_id, 'long_name', 'retrieval height grid'

	cloud_model_id = NCDF_VARDEF(outfile_id, 'cloud_model', /BYTE)
	NCDF_ATTPUT, outfile_id, cloud_model_id, 'units', '-'
	cloud_model_id_long_name_1 = 'cloud models used, bit 1: modified adiabatic cloud model Carstens et al. 1994, '
	cloud_model_id_long_name_2 = 'bit 2: Decker et al. (1978) cloud model, bit 3: Salonen (2006) cloud model'
	NCDF_ATTPUT, outfile_id, cloud_model_id, 'long_name', cloud_model_id_long_name_1 + cloud_model_id_long_name_2

	cscl_id = NCDF_VARDEF(outfile_id, 'cscl', [str4_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, cscl_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, cscl_id, 'long_name', 'clear, cloudy or both: cscl - both; cs - clear sky only; cl - cloud sky only'

	gas_absorption_model_id = NCDF_VARDEF(outfile_id, 'gas_absorption_model', [str3_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, gas_absorption_model_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, gas_absorption_model_id, 'long_name', 'gas absorption model: l87 - Liebe 1987; r98 - Rosenkranz 1998; l93 - Liebe 1993; mrtm - MonoRTM'

	cloud_absorption_model_id = NCDF_VARDEF(outfile_id, 'cloud_absorption_model', [str3_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, cloud_absorption_model_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, cloud_absorption_model_id, 'long_name', 'liquid absorption model: ula - Ullaby I, 1983; lie - Liebe et al., 1991; ell - Ellisson, 2006'

	linewidth_id = NCDF_VARDEF(outfile_id, 'linewidth', [str5_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, linewidth_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, linewidth_id, 'long_name', 'line width 22GHz for R98: org - original as in Rosenkranz 1998 model; lil05 - modified according to Liljgren 2005'

	cont_corr_id = NCDF_VARDEF(outfile_id, 'cont_corr', [str5_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, cont_corr_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, cont_corr_id, 'long_name', 'WV cont. correction: org - original model; tur09 - modified according to Turner et al 2009'

	air_mass_correction_id = NCDF_VARDEF(outfile_id, 'air_mass_correction', [str15_dim_id], /CHAR)
	NCDF_ATTPUT, outfile_id, air_mass_correction_id, 'units', '-'
	air_mass_correction_id_long_name_1 = 'air mass correction: rueeger_avai_02 - Rueeger 2002 (best avai.); sphere - spherical atmosphere, '
	air_mass_correction_id_long_name_2 = 'no refraction; no - no correction: plane parall atmosphere, no refraction'
	NCDF_ATTPUT, outfile_id, air_mass_correction_id, 'long_name', air_mass_correction_id_long_name_1 + air_mass_correction_id_long_name_2

	date_id = NCDF_VARDEF(outfile_id, 'date', [date_dim_id], /LONG)
	NCDF_ATTPUT, outfile_id, date_id, 'units', 'yyyymmddhh'
	NCDF_ATTPUT, outfile_id, date_id, 'long_name', 'year, month, day, hour'

	atmosphere_temperature_id = NCDF_VARDEF(outfile_id, 'atmosphere_temperature', [hght_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_temperature_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, atmosphere_temperature_id, 'long_name', 'temperature profile on height grid'

	atmosphere_humidity_id = NCDF_VARDEF(outfile_id, 'atmosphere_humidity', [hght_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_humidity_id, 'units', 'kg/m^3'
	NCDF_ATTPUT, outfile_id, atmosphere_humidity_id, 'long_name', 'humidity profile on height grid'

	atmosphere_pressure_id = NCDF_VARDEF(outfile_id, 'atmosphere_pressure', [hght_dim_id,  date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_pressure_id, 'units', 'Pa'
	NCDF_ATTPUT, outfile_id, atmosphere_pressure_id, 'long_name', 'pressure profile on height grid'

	number_of_levels_in_rs_ascent_id = NCDF_VARDEF(outfile_id, 'number_of_levels_in_rs_ascent', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, number_of_levels_in_rs_ascent_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, number_of_levels_in_rs_ascent_id, 'long_name', 'number of levels in original radiosonde profile'

	highest_level_in_rs_ascent_id = NCDF_VARDEF(outfile_id, 'highest_level_in_rs_ascent', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, highest_level_in_rs_ascent_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, highest_level_in_rs_ascent_id, 'long_name', 'highest level of original radiosonde profile'

	atmosphere_temperature_sfc_id = NCDF_VARDEF(outfile_id, 'atmosphere_temperature_sfc', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_temperature_sfc_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, atmosphere_temperature_sfc_id, 'long_name', 'temperature surface value'

	atmosphere_humidity_sfc_id = NCDF_VARDEF(outfile_id, 'atmosphere_humidity_sfc', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_humidity_sfc_id, 'units', 'kg/m^3'
	NCDF_ATTPUT, outfile_id, atmosphere_humidity_sfc_id, 'long_name', 'humidity surface value'

	atmosphere_pressure_sfc_id = NCDF_VARDEF(outfile_id, 'atmosphere_pressure_sfc', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, atmosphere_pressure_sfc_id, 'units', 'Pa'
	NCDF_ATTPUT, outfile_id, atmosphere_pressure_sfc_id, 'long_name', 'pressure surface value'

	k_index_id = NCDF_VARDEF(outfile_id, 'k_index', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, k_index_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, k_index_id, 'long_name', 'K Index'

	ko_index_id = NCDF_VARDEF(outfile_id, 'ko_index', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, ko_index_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, ko_index_id, 'long_name', 'KO Index'

	tt_index_id = NCDF_VARDEF(outfile_id, 'tt_index', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, tt_index_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, tt_index_id, 'long_name', 'Totals Totals Index'

	li_index_id = NCDF_VARDEF(outfile_id, 'li_index', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, li_index_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, li_index_id, 'long_name', 'Lifted Index'

	si_index_id = NCDF_VARDEF(outfile_id, 'si_index', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, si_index_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, si_index_id, 'long_name', 'Showalter Index'

	cape_mu_id = NCDF_VARDEF(outfile_id, 'cape_mu', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, cape_mu_id, 'units', 'J/m^2'
	NCDF_ATTPUT, outfile_id, cape_mu_id, 'long_name', 'Cape - Convective available potential energy'

	integrated_water_vapor_id = NCDF_VARDEF(outfile_id, 'integrated_water_vapor', [date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, integrated_water_vapor_id, 'units', 'kg/m-2'
	NCDF_ATTPUT, outfile_id, integrated_water_vapor_id, 'long_name', 'integrated water vapour'

	wet_delay_id = NCDF_VARDEF(outfile_id, 'wet_delay', [nwdm_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, wet_delay_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, wet_delay_id, 'long_name', 'wet path delay using 12 different refractivity models'

	liquid_water_path_id = NCDF_VARDEF(outfile_id, 'liquid_water_path', [cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, liquid_water_path_id, 'units', 'kg/m-2'
	NCDF_ATTPUT, outfile_id, liquid_water_path_id, 'long_name', 'liquid water path'

	cloud_base_id = NCDF_VARDEF(outfile_id, 'cloud_base', [cloud_max_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, cloud_base_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, cloud_base_id, 'long_name', 'cloud base array'

	cloud_top_id = NCDF_VARDEF(outfile_id,'cloud_top', [cloud_max_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, cloud_top_id, 'units', 'm'
	NCDF_ATTPUT, outfile_id, cloud_top_id, 'long_name', 'cloud top array'

	liquid_water_path_single_cloud_id = NCDF_VARDEF(outfile_id, 'liquid_water_path_single_cloud', [cloud_max_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, liquid_water_path_single_cloud_id, 'units', 'kg/m-2'
	NCDF_ATTPUT, outfile_id, liquid_water_path_single_cloud_id, 'long_name', 'liquid water path for each single (separated in space) cloud'

	optical_depth_id = NCDF_VARDEF(outfile_id, 'optical_depth', [freq_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, optical_depth_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, optical_depth_id, 'long_name', 'zenith optical depth'

	optical_depth_wv_id = NCDF_VARDEF(outfile_id, 'optical_depth_wv', [freq_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, optical_depth_wv_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, optical_depth_wv_id, 'long_name', 'zenith optical depth water vapor'

	optical_depth_o2_id = NCDF_VARDEF(outfile_id, 'optical_depth_o2', [freq_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, optical_depth_o2_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, optical_depth_o2_id, 'long_name', 'zenith optical depth O2'

	optical_depth_liq_l91_id = NCDF_VARDEF(outfile_id, 'optical_depth_liq_l91', [freq_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, optical_depth_liq_l91_id, 'units', '-'
	NCDF_ATTPUT, outfile_id, optical_depth_liq_l91_id, 'long_name', 'zenith optical depth liquid water'

	aperture_correction_id = NCDF_VARDEF(outfile_id, 'aperture_correction', /FLOAT)

	bandwidth_correction_id = NCDF_VARDEF(outfile_id, 'bandwidth_correction', /FLOAT)

	brightness_temperatures_id  = NCDF_VARDEF(outfile_id, 'brightness_temperatures', [freq_dim_id, angl_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, brightness_temperatures_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, brightness_temperatures_id, 'long_name', 'brightness temperature at frequency and angle'

	brightness_temperatures_instrument_id = NCDF_VARDEF(outfile_id, 'brightness_temperatures_instrument', [freq_dim_id, angl_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, brightness_temperatures_instrument_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, brightness_temperatures_instrument_id, 'long_name', 'brightness temperature instrument specific (including bandwidth and antenna aperture characteristics)'

	mean_radiating_temperature_id = NCDF_VARDEF(outfile_id, 'mean_radiating_temperature', [freq_dim_id, angl_dim_id, cloud_model_dim_id, date_dim_id], /FLOAT)
	NCDF_ATTPUT, outfile_id, mean_radiating_temperature_id, 'units', 'K'
	NCDF_ATTPUT, outfile_id, mean_radiating_temperature_id, 'long_name', 'mean radiating temperature temperature at frequency and angle'


	; Global attributes:
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'location', 'Central-Arctic'
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'longitude', 5.25
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'latitude', 87.00
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'altitude MSL', 0
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'system', 'radiative transfer calculations performed by Emiliano Orlandi (RPG)'
	NCDF_ATTPUT, outfile_id, /GLOBAL, 'institution', 'data processed by IGMK, contact: a.walbroel@uni-koeln.de'

	; fill all variables with dummy values
	NCDF_CONTROL, outfile_id, /fill

	; close define mode
	NCDF_CONTROL, outfile_id, /verbose
	NCDF_CONTROL, outfile_id, /endef


	; assign data to variables:
	NCDF_VARPUT, outfile_id, station_id_id, station_id
	NCDF_VARPUT, outfile_id, data_path_to_rs_id, '/net/blanc/awalbroe/Data/'
	NCDF_VARPUT, outfile_id, rh_thres_id, 0.95
	NCDF_VARPUT, outfile_id, height_above_sea_level_id, 0
	;NCDF_VARPUT, outfile_id, cap_height_above_sea_level_id, 0
	NCDF_VARPUT, outfile_id, frequency_id, freq_dsba
	NCDF_VARPUT, outfile_id, elevation_angle_id, 90.0		; elevation angle is 90: zenith
	NCDF_VARPUT, outfile_id, height_grid_id, Z_new
	;NCDF_VARPUT, outfile_id, cloud_model_id, 0
	;NCDF_VARPUT, outfile_id, cscl_id, '0'
	NCDF_VARPUT, outfile_id, gas_absorption_model_id, 'r98'
	NCDF_VARPUT, outfile_id, cloud_absorption_model_id, '---'
	NCDF_VARPUT, outfile_id, linewidth_id, '---'
	NCDF_VARPUT, outfile_id, cont_corr_id, '---'
	NCDF_VARPUT, outfile_id, air_mass_correction_id, 'yn'
	NCDF_VARPUT, outfile_id, date_id, date_save
	NCDF_VARPUT, outfile_id, atmosphere_temperature_id, T_save
	NCDF_VARPUT, outfile_id, atmosphere_humidity_id, ABSHUM_save
	NCDF_VARPUT, outfile_id, atmosphere_pressure_id, P_save
	;NCDF_VARPUT, outfile_id, number_of_levels_in_rs_ascent_id, 0
	NCDF_VARPUT, outfile_id, highest_level_in_rs_ascent_id, REPLICATE(10000, n_ix_year)
	NCDF_VARPUT, outfile_id, atmosphere_temperature_sfc_id, T_s_save
	NCDF_VARPUT, outfile_id, atmosphere_humidity_sfc_id, ABSHUM_s_save
	NCDF_VARPUT, outfile_id, atmosphere_pressure_sfc_id, P_s_save
	;NCDF_VARPUT, outfile_id, k_index_id, 0
	;NCDF_VARPUT, outfile_id, ko_index_id, 0
	;NCDF_VARPUT, outfile_id, tt_index_id, 0
	;NCDF_VARPUT, outfile_id, li_index_id, 0
	;NCDF_VARPUT, outfile_id, si_index_id, 0
	;NCDF_VARPUT, outfile_id, cape_mu_id, 0
	NCDF_VARPUT, outfile_id, integrated_water_vapor_id, IWV_save
	;NCDF_VARPUT, outfile_id, wet_delay_id, 0
	NCDF_VARPUT, outfile_id, liquid_water_path_id, LWP_save
	;NCDF_VARPUT, outfile_id, cloud_base_id, 0
	;NCDF_VARPUT, outfile_id, cloud_top_id, 0
	;NCDF_VARPUT, outfile_id, liquid_water_path_single_cloud_id, 0
	;NCDF_VARPUT, outfile_id, optical_depth_id, 0
	;NCDF_VARPUT, outfile_id, optical_depth_wv_id, 0
	;NCDF_VARPUT, outfile_id, optical_depth_o2_id, 0
	;NCDF_VARPUT, outfile_id, optical_depth_liq_l91_id, 0
	;NCDF_VARPUT, outfile_id, aperture_correction_id, 0
	;NCDF_VARPUT, outfile_id, bandwidth_correction_id, 0
	NCDF_VARPUT, outfile_id, brightness_temperatures_id, TBs_save
	;NCDF_VARPUT, outfile_id, brightness_temperatures_instrument_id, 0
	;NCDF_VARPUT, outfile_id, mean_radiating_temperature_id, 0

	NCDF_CLOSE, outfile_id

ENDFOR

print, "Done...."

; always need 'END' at the end of a script
END
