# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:20:29 2020

@author: tsrik
Dataset details https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
Code reference source: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

# "The CNN LSTM architecture involves using Convolutional Neural Network (CNN) layers for feature extraction on input data combined with LSTMs to support sequence prediction.
    The CNN LSTM model will read subsequences of the main sequence in as blocks, extract features from each block, then allow the LSTM to interpret the features extracted from each block.
    One approach to implementing this model is to split each window of 128 time steps into subsequences for the CNN model to process. 
    For example, the 128 time steps in each window can be split into 4 subsequences of 32 time steps." - from the above source

9 features, each in a file:
    -The “Inertial Signals” directory contains 9 files.
    --Gravitational acceleration data files for x, y and z axes: total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt.
    --Body acceleration data files for x, y and z axes: body_acc_x_train.txt, body_acc_y_train.txt, body_acc_z_train.txt.
    --Body gyroscope data files for x, y and z axes: body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt.
    
    Each of the 9 features are listed below
    Total Acceleration x
    Total Acceleration y
    Total Acceleration z
    Body Acceleration x
    Body Acceleration y
    Body Acceleration z
    Body Gyroscope x
    Body Gyroscope y
    Body Gyroscope z
6 classes or type-of-activity to get trained on and predict:
    1 WALKING
    2 WALKING_UPSTAIRS
    3 WALKING_DOWNSTAIRS
    4 SITTING
    5 STANDING
    6 LAYING

The data has been split into windows of 128 time steps, with a 50% overlap.
trainX = (7352, 128, 9) 7352 samples or windows, each 128 timesteps or readings, represented with 9 features where each feature data is in a separate file
testX(2947, 128, 9)

trainY (7352, 1), activity type performed (one of the six activities)
testY  (2947, 1)
"""

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI_HAR_Dataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI_HAR_Dataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

#load the data
'''   .
trainX = (7352, 128, 9) 7352 samples or windows, each 128 timesteps or readings, represented with 9 features where each feature data is in a separate file
testX(2947, 128, 9)
trainY (7352, 1), activity type performed (one of the six activities)
testY  (2947, 1)
'''
trainX, trainy, testX, testy = load_dataset()
verbose, epochs, batch_size = 2, 10, 32
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# reshape data into time steps of sub-sequences
# Unlike an LSTM that reads the data in directly in order to calculate internal state and state transitions, 
#   and unlike the CNN LSTM that is interpreting the output from CNN models, 
#   the ConvLSTM is using convolutions directly as part of reading input into the LSTM units themselves. 
#   .. in other words ConvLSTM uses convolutions as against matrix multiplications in LSTM
# The ConvLSTM2D class, by default, expects input data to have the shape:(samples, time, rows, cols, channels)
#   Where each time step of data is defined as an image of (rows * columns) data points.
#   Diff between CNN+LSTM and ConvLSTM https://qr.ae/pNnMpc

##the 128 time steps in each window can be split into 4 subsequences or blocks of 32 time steps 
#   We can use this same subsequence approach in defining the ConvLSTM2D input where the number of time steps is the number of subsequences in the window, the number of rows is 1 as we are working with one-dimensional data, and the number of columns represents the number of time steps in the subsequence, in this case 32.
#   For this chosen framing of the problem, the input for the ConvLSTM2D would therefore be:

#   Samples: n, for the number of windows in the dataset.
#   Time: 4, for the four subsequences that we split a window of 128 time steps into.
#   Rows: 1, for the one-dimensional shape of each subsequence.
#   Columns: 32, for the 32 time steps in an input subsequence.
#   Channels: 9, for the nine input variables.
# in summary: a 2D image of 1X32 is convolved at each timestep in the LSTM.. and this happens 4 times (timesteps).. thus completing 128 timesteps 


# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
print('Test accuracy:', accuracy)
