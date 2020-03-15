# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:20:29 2020

@author: tsrik
Reference Credits:
-Dataset details https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
-Code reference source: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

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


from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
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
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
print('Test accuracy:', accuracy)
