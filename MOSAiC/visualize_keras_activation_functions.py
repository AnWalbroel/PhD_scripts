from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pdb

# # """
	# # Code is taken from http://proc-x.com/2017/12/visualize-activations-functions-using-keras/ !
# # """

# # def NNmodel(activationFunc='linear'):
	# # """
		# # Define a neural network which can be arbitrary but we keep it simple here
	# # """
	# # model = Sequential()
	# # model.add(Dense(1, input_shape=(1,), activation=activationFunc))

	# # model.add(Dense(1, activation='linear'))
	# # model.compile(loss='mse', optimizer='adagrad', metrics=['mse'])
	# # return model

# # def VisualActivation(activationFunc='relu', plot=True):
	# # x = (np.arange(100)-50)/10		# -5 to 5
	# # y = np.log(x+51/10)

	# # model = NNmodel(activationFunc=activationFunc)
	# # # optional to fit the model. If not fitted, initialized parameters will be used
	# # model.fit(x, x, epochs=1, batch_size=128, verbose=0)

	# # # define the computing process
	# # inX = model.input
	# # outputs = [layer.output for layer in model.layers]
	# # functions = [K.function([inX], [out]) for out in outputs]

	# # # compute based on original inputs
	# # activationLayer={}
	# # for i in range(100):
		# # test = x[i].reshape(-1,1)
		# # layer_outs = [func([test]) for func in functions]
		# # activationLayer[i] = layer_outs[0][0][0][0]

	# # # process results
	# # activationDf = pd.DataFrame.from_dict(activationLayer, orient='index')
	# # result = pd.concat([pd.DataFrame(x), activationDf], axis=1)
	# # result.columns=['X', 'Activated']
	# # result.set_index('X', inplace=True)
	# # if plot:
		# # result.plot(title=f)

	# # return result

def activation_functions(x, AF):
	"""
		Use mathematical definitions of the activation functions
		given in https://keras.io/api/layers/activations/ .
	"""

	if AF == 'sigmoid':
		y = 1 / (1 + np.exp(-x))
	elif AF == 'tanh':
		y = np.tanh(x)
	elif AF == 'softplus':
		y = np.log(np.exp(x) + 1)
	elif AF == 'softmax':
		y = np.exp(x) / (tf.reduce_sum(np.exp(x)))
	elif AF == 'selu':
		y = tf.keras.activations.selu(x)
	elif AF == 'elu':
		y = tf.keras.activations.elu(x, alpha=1.0)
	elif AF == 'softsign':
		y = x / (np.abs(x) + 1)
	elif AF == 'exponential':
		y = np.exp(x)
	elif AF == 'relu':
		y = tf.keras.activations.relu(x)

	return y



# # # # Now we can visualize them (assuming default settings):
# # # actFuncs = ['sigmoid', 'tanh', 'softplus', 'softmax', 'selu', 'elu', 
			# # # 'softsign', 'exponential']

# # # figure = plt.figure()
# # # for i, f in enumerate(actFuncs):
	# # # figure.add_subplot(3,3,i+1)
	# # # out = VisualActivation(activationFunc=f, plot=False)
	# # # plt.plot(out.index, out.Activated)
	# # # plt.title(f)

# # # plt.show()


# Code by A. Walbroel: ALTERNATIVELY: 
actFuncs = ['sigmoid', 'tanh', 'softplus', 'softmax', 'selu', 'relu', 
			'softsign', 'exponential']

figure = plt.figure()
figure.set_size_inches(15,10)
for i, f in enumerate(actFuncs):
	figure.add_subplot(3,3,i+1)

	x_stuff = np.arange(-7.6, 3.10000001, 0.01)

	y_stuff = activation_functions(x_stuff, f)

	plt.plot(x_stuff, y_stuff)
	plt.grid()
	plt.title(f)

figure.savefig("/net/blanc/awalbroe/Plots/NN_test/KERAS_ACTIVATION_FUNCTIONS.png", dpi=400)
