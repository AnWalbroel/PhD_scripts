pro read_binary_data_RPG, filename, data

filename_dim = filename+'.dim'
filename_bin = filename+'.binary'

;-------------------------------------------------------------------------------
; Now read the binary file, if applicable
;-------------------------------------------------------------------------------
print,'RDB:  Start reading Binary dump of data ball file...'

line  = ' '
meta1 = 'meta-data-1'
meta2 = 'meta-data-2'
nof_frequencies     =0L
nof_angles          =0L
nof_additional_data =0L
nof_gridded_data    =0L
nof_levels          =0L
nof_rtm_data        =0L
nof_entries         =0L
gas_absorpt_model   = ' '

openr,unit1,filename_dim ,/get_lun
openr,unit2,filename_bin ,/get_lun

readf,unit1,line
readf,unit1,meta1
readf,unit1,meta2
readf,unit1,nof_frequencies     
readf,unit1,nof_angles          
readf,unit1,nof_additional_data 
readf,unit1,nof_gridded_data    
readf,unit1,nof_levels          
readf,unit1,nof_rtm_data        
readf,unit1,nof_entries         
readf,unit1,gas_absorpt_model
;if gas_absorpt_model.Contains('max_IWV') Then begin
;  print,'OLD version of databall file, cannot be used.'
;  print,'Please insert a line after "# nof_entries" with this string "r98 # Gas absorption model used".'
;  stop
;endif
gas_absorpt_model = strtrim((strsplit(gas_absorpt_model, '#'))[0],2)
; gas_absorpt_model = strtrim((gas_absorpt_model.split('#'))[0],2)
close   ,unit1
free_lun,unit1

;-------------------------------------------------------------------------------
; create data arrays with correct dimensions and size
;-------------------------------------------------------------------------------
angles      = dblarr(nof_angles     ,/NOZERO) 			                    ;
frequencies = dblarr(nof_frequencies,/NOZERO)                         	;
Z           = dblarr(nof_levels     ,/NOZERO)		                       	;	Z = Reform(data_gridded[*,0,0])
data_additional = dblarr(         nof_additional_data, nof_entries,/NOZERO)
data_gridded    = dblarr(nof_levels, nof_gridded_data, nof_entries,/NOZERO)
data_rtm        = dblarr(nof_rtm_data,nof_frequencies,nof_angles, nof_entries,/NOZERO)
;-------------------------------------------------------------------------------
; Now read the data files
;-------------------------------------------------------------------------------
readu,unit2,angles          ;
readu,unit2,frequencies     ;
readu,unit2,Z               ;
readu,unit2,data_additional ;
readu,unit2,data_gridded    ;
readu,unit2,data_rtm        ;
;-------------------------------------------------------------------------------
close   ,unit2
free_lun,unit2
;----------------------------------------------------------------------------
print,'RDB:  Binary reading of data-ball dump-file completed!'
print,'RDB:  Gas absorption model used: ',gas_absorpt_model
print,'RDB:  Dimensions: '
print,'RDB:  Additional ',size(data_additional)
print,'RDB:  Gridded    ',size(data_gridded)
print,'RDB:  Data-block ',size(data_rtm)
print,'RDB:  --------------------------------------------------------------------------'

;----------------------------------------------------------------------------
data_gridded_indices = { index_grid_z  :  0, $ ;profile data vectors z-levels
                         index_grid_t  :  1, $ ;profile data vectors temp.
                         index_grid_r  :  2, $ ;profile data vectors rel.hum.
                         index_grid_a  :  3, $ ;profile data vectors abs.hum.
                         index_grid_l  :  4}   ;profile data vectors lwc-profile
;-------------------------------------------------------------------------------
additional_data_indices = { index_yy     :  0, $ 
                            index_mm     :  1, $ 
                            index_dd     :  2, $ 
                            index_hh     :  3, $ 
                            index_Delay  :  4, $ 
                            index_LWM    :  5, $ 
                            index_LWP    :  6, $ 
                            index_IWV    :  7, $ 
                            index_T_s    :  8, $ 
                            index_R_s    :  9, $ 
                            index_P_s    : 10, $ 
                            index_Z_s    : 11, $ 
                            index_T_c    : 12, $ 
                            index_elev   : 13, $ 
                            index_KI     : 14, $ 
                            index_KO     : 15, $ 
                            index_TT     : 16, $ 
                            index_LI     : 17, $ 
                            index_SI     : 18, $ 
                            index_CAPE   : 19, $ 
                            index_IR0    : 20, $ ; Standard KT19 IR radiometer
                            index_IR1    : 21, $ 
                            index_IR2    : 22, $ 
                            index_IR3    : 23, $ 
                            index_IR4    : 24, $ 
                            index_WS     : 25, $ ; Wind speed
                            index_WD     : 26, $ ; Wind direction
                            index_x1     : 27, $ ; Temperature inversion parameters
                            index_x2     : 28, $
                            index_x3     : 29, $
                            index_x4     : 30, $
                            index_x5     : 31, $
                            index_00     : 32  }
;----------------------------------------------------------------------------
data_data_additional_dim  = "Array[nof_additional_data, nof_entries]"
data_gridded_dim          = "Array[nof_levels, nof_gridded_data, nof_entries]"
data_rtm_dim              = "Array[nof_rtm_data, nof_frequencies, nof_angles, nof_entries]"
;----------------------------------------------------------------------------
data_rtm_indices = { index_rtm_A   : 0, $ ; Angle
                     index_rtm_F   : 1, $ ; Frequency
                     index_rtm_TB  : 2, $ ; TB
                     index_rtm_H   : 3, $ ; Scale height of absorption
                     index_rtm_TMR : 4, $ ; TMR
                     index_rtm_tau : 5}   ; Tau
;----------------------------------------------------------------------------
data = {    meta1                    : meta1                    ,$
            meta2                    : meta2                    ,$
            nof_frequencies          : nof_frequencies          ,$
            nof_angles               : nof_angles               ,$
            nof_additional_data      : nof_additional_data      ,$
            nof_gridded_data         : nof_gridded_data         ,$
            nof_levels               : nof_levels               ,$
            nof_rtm_data             : nof_rtm_data             ,$
            nof_entries              : nof_entries              ,$
                                                                
            angles                   : angles                   ,$
            frequencies              : frequencies              ,$
            Z                        : Z                        ,$
            data_additional          : data_additional          ,$
            data_gridded             : data_gridded             ,$
            data_rtm                 : data_rtm                 ,$
            
            data_gridded_indices     : data_gridded_indices     ,$
            additional_data_indices  : additional_data_indices  ,$
            data_rtm_indices         : data_rtm_indices         ,$
            
            data_data_additional_dim : data_data_additional_dim ,$
            data_gridded_dim         : data_gridded_dim         ,$
            data_rtm_dim             : data_rtm_dim             }

print,'RDB:',size(data_additional)
print,'RDB:',size(data_gridded)
print,'RDB:',size(data_rtm)

print,'RDB:  Max IWV: ',max(data_additional[7,*])

;----------------------------------------------------------------------------

end
