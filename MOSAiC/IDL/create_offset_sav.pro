;
;******************************
PRO CREATE_OFFSET_SAV
;******************************

;  The tb_offset_file must be of IDL binary format (.sav) and contain the parameters defined in the following:
;  date_offset=DBLARR(n_date_offset): specifies the times AFTER which a certain offset correction is valid (Julian day format!)
;  freq_offset=DBLARR(n_freq_offset): frequency channels that offset correction is applied to (GHz)
;  ang_offset=DBLARR(n_ang_offset): elevation angles that offset correction is applied to (elevation angle)
;  tb_offset=DBLARR(n_date_offset, n_freq_offset, n_ang_offset): tb offset correction values (K) to be SUBTRACTED from
;  original values to ensure correct values



; path of offset data created with PAMTRA_fwd_sim.py:
path_data = "/net/blanc/awalbroe/Data/MOSAiC_radiometers_offsets/"
path_output = "/net/blanc/awalbroe/Data/"

nc_id = NCDF_open(path_data + "MOSAiC_mirac-p_radiometer_clear_sky_offset_correction.nc")

bias_id = NCDF_VARID(nc_id, 'bias')
frequency_id = NCDF_VARID(nc_id, 'frequency')
cal_start_id = NCDF_VARID(nc_id, 'calibration_period_start')
cal_end_id = NCDF_VARID(nc_id, 'calibration_period_end')

NCDF_VARGET, nc_id, bias_id, bias
NCDF_VARGET, nc_id, frequency_id, frequency
NCDF_VARGET, nc_id, cal_start_id, cal_start
NCDF_VARGET, nc_id, cal_end_id, cal_end


n_date_offset = N_ELEMENTS(cal_start)
n_freq_offset = N_ELEMENTS(frequency)
n_ang_offset = 1

date_offset = DBLARR(n_date_offset)
freq_offset = DBLARR(n_freq_offset)
ang_offset = DBLARR(n_ang_offset)
tb_offset = REFORM(DBLARR(n_date_offset, n_freq_offset, n_ang_offset), [n_date_offset, n_freq_offset, n_ang_offset])



; Compute Julian Day:
;array(['2019-09-20T00:00:00.000000000', '2019-10-19T06:30:00.000000000',
;       '2019-10-22T05:40:00.000000000', '2020-07-06T12:19:00.000000000',
;       '2020-08-12T09:37:00.000000000'], dtype='datetime64[ns]')
CAL_Y = [2019, 2019, 2019, 2020, 2020]
CAL_M = [9, 10, 10, 7, 8]
CAL_D = [20, 19, 22, 6, 12]
CAL_H = [0, 6, 5, 12, 9]
CAL_MIN = [0, 30, 40, 19, 37]
CAL_S = [0, 0, 0, 0, 0]

JD = JULDAY(CAL_M, CAL_D, CAL_Y, CAL_H, CAL_MIN, CAL_S)


; Set variables:
date_offset = JD
freq_offset = frequency
ang_offset(0) = 90.0
tb_offset = REFORM(TRANSPOSE(bias), [n_date_offset, n_freq_offset, n_ang_offset])

outfile = path_output + 'tb_offset_pol.sav'
SAVE, date_offset, freq_offset, ang_offset, tb_offset, FILENAME=outfile, /VERBOSE

END
