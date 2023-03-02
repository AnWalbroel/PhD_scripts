;
;******************************
PRO CHECK_TB_BIASCOV
;******************************

; check tb bias and cov files:
path_data = "/net/blanc/awalbroe/Data/MiRAC-P_retrieval_RPG/"
file_bias = path_data + 'tb_bias_nya_lhumpro_20170516'
file_cov = path_data + 'tb_cov_nya_lhumpro_20170516'

RESTORE, file = file_bias
RESTORE, file = file_cov

stop



END
