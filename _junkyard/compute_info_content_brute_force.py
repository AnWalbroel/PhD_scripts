		if aux_i['get_info_content']:
			i_cont = info_content(predictand_test.output, predictor_test.TB, prediction_syn, ax_samp=0, ax_comp=1,
									perturbation=1.01, perturb_type='multiply', aux_i=aux_i, 
									suppl_data={'lat': predictand_test.lat, 'lon': predictand_test.lon,
												'time': predictand_test.time, 'height': predictand_test.height,
												'rh': predictand_test.rh, 'temp': predictand_test.temp, 
												'pres': predictand_test.pres, 'temp_sfc': predictand_test.temp_sfc,
												'cwp': predictand_test.cwp, 'rwp': predictand_test.rwp,	# LWP == CWP
												'swp': predictand_test.swp, 'iwp': predictand_test.iwp,
												'q': predictand_test.q})


			# create new reference observation and state vector instead of the predictor_test.TB because that was created on
			# another height grid.
			i_cont.new_obs(False, what_data='samp')
			
			# First, rebuild the input_scaled vector, since the obs vector in i_cont consists of TBs only.
			new_input = i_cont.y			# (n_samples, n_TBs) shape

			# If chosen, add surface pressure to input vector:
			if "pres_sfc" in aux_i['predictors']:
				new_input = np.concatenate((new_input, np.reshape(predictor.pres, (aux_i['n_test'],1))), axis=1)

			# Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
			if ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
				new_input = np.concatenate((new_input, predictor.DOY_1, predictor.DOY_2), axis=1)

			# scale input and retrieve new states
			new_input_scaled = scaler.transform(new_input)
			i_cont.x_ret = model.predict(new_input_scaled)		# replace retrieved state vector by the new one


			# Loop through test data set: perturb each state vector component, generate new obs 
			# via simulations, apply retrieval, compute AK:
			n_comp = i_cont.x.shape[i_cont.ax_c]
			for i_s in range(aux_i['n_test']):

				i_cont.perturb('state', i_s, 'all')
				i_cont.new_obs(True, what_data='comp')

				# retrieve new state vectors based on the perturbed obs vectors for test data sample i:
				# First, rebuild the input_scaled vector, since the obs vector in i_cont consists of TBs only.
				new_input = i_cont.y_ip_mat			# (n_comp_x, n_TBs) shape

				# If chosen, add surface pressure to input vector:
				if "pres_sfc" in aux_i['predictors']:
					new_input = np.concatenate((new_input, np.reshape(predictor.pres, (aux_i['n_test'],1))), axis=1)

				# Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
				if ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
					new_input = np.concatenate((new_input, predictor.DOY_1, predictor.DOY_2), axis=1)

				# scale input and retrieve new states
				new_input_scaled = scaler.transform(new_input)
				i_cont.x_ret_ip_mat = model.predict(new_input_scaled)
				i_cont.compute_dx_ret_i('all')
				i_cont.compute_AK_i()
				i_cont.compute_DOF()
				i_cont.visualise_AK_i()


				# # OR: for each component step by step:
				# for i_c in range(n_comp):
					# i_cont.perturb('state', i_s, i_c)
					# i_cont.new_obs(True, what_data='single')

					# # retrieve new state vectors based on the perturbed obs vectors for test data sample i:
					# # First, rebuild the input_scaled vector, since the obs vector in i_cont consists of TBs only.
					# new_input = np.reshape(i_cont.y_ip, (1, i_cont.n_cy))	# (1, n_TBs) shape

					# # If chosen, add surface pressure to input vector:
					# if "pres_sfc" in aux_i['predictors']:
						# new_input = np.concatenate((new_input, np.reshape(predictor.pres, (aux_i['n_test'],1))), axis=1)

					# # Compute Day of Year in radians if the sin and cos of it shall also be used in input vector:
					# if ("DOY_1" in aux_i['predictors']) and ("DOY_2" in aux_i['predictors']):
						# new_input = np.concatenate((new_input, predictor.DOY_1, predictor.DOY_2), axis=1)

					# # scale input and retrieve new states
					# new_input_scaled = scaler.transform(new_input)
					# i_cont.x_ret_ip = model.predict(new_input_scaled)		# new ret state vector based on component j being perturbed
					# i_cont.compute_dx_ret_i(i_c)
					# i_cont.compute_col_of_AK_i()
				# i_cont.compute_DOF()
				# i_cont.visualise_AK_i()