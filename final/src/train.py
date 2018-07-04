from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K

import os
import shutil
import csv
from sys import argv
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import image
from keras.models import Model, load_model
import heapq
from collections import Counter

def main():
	X_Train = np.load('X_Train.npy')
	Y_Train = np.load('Y_Train.npy')


	datagen = ImageDataGenerator(
		rotation_range=30,
		horizontal_flip=True)

	n_models = ['res1.h5', 'nonew3.h5', 'vgg192.h5', 'iv32.h5', 'tune4.h5', 'tune3.h5']

	if argv[1] in {'0','1','2'}:
		delete = np.load('new_whaleID.npy')
		X_Train = np.delete(X_Train, delete, axis=0)
		Y_Train = np.delete(Y_Train, delete, axis=0)

	nclass = Y_Train.shape[1]

	if argv[1] == '0':
		base_model = ResNet50(include_top=False, weights='imagenet')
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nclass, activation='softmax')(x)
		for layer in base_model.layers:
			layer.trainable = False

		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[0], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]
		batch_size = 128
		epochs = 20
		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=10*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

	elif argv[1] == '1':
		base_model = Xception(include_top=False, weights='imagenet')
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nclass, activation='softmax')(x)
		for layer in base_model.layers:
			layer.trainable = False

		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[1], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		batch_size = 128
		epochs = 8
		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

		model = load_model(n_models[1])
		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:116]:
			layer.trainable = False
		for layer in model.layers[116:]:
			layer.trainable = True

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)
	elif argv[1] == '2':
		base_model = VGG19(include_top=False, weights='imagenet')
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nclass, activation='softmax')(x)
		for layer in base_model.layers:
			layer.trainable = False

		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[2], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		batch_size = 64
		epochs = 10
		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=15*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

		model = load_model(n_models[2])
		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:16]:
			layer.trainable = False
		for layer in model.layers[16:]:
			layer.trainable = True

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

	elif argv[1] == '3':
		base_model = InceptionV3(include_top=False, weights='imagenet')

		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nclass, activation='softmax')(x)
		for layer in base_model.layers:
			layer.trainable = False

		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


		callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[3], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		batch_size = 128
		epochs = 20
		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=10*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

		model = load_model(n_models[3])
		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:249]:
			layer.trainable = False
		for layer in model.layers[249:]:
			layer.trainable = True

		callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[3], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

	elif argv[1] == '4':
		base_model = Xception(include_top=False, weights='imagenet')

		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(nclass, activation='softmax')(x)
		for layer in base_model.layers:
			layer.trainable = False

		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

		callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[4], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		batch_size = 128
		epochs = 25
		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks)


		model = load_model(n_models[4])

		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:126]:
			layer.trainable = False
		for layer in model.layers[126:]:
			layer.trainable = True

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks
			)

		model = load_model(n_models[4])

		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:116]:
			layer.trainable = False
		for layer in model.layers[116:]:
			layer.trainable = True

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks)
	elif argv[1] == '5':

		model = load_model(n_models[4])

		callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
		ModelCheckpoint(filepath=n_models[5], verbose=1, save_best_only=True, monitor='val_acc',mode='max')]

		model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

		for layer in model.layers[:106]:
			layer.trainable = False
		for layer in model.layers[106:]:
			layer.trainable = True

		batch_size = 128
		epochs = 25

		model.fit_generator(
			datagen.flow(X_Train, Y_Train, batch_size=batch_size), 
			steps_per_epoch=12*len(X_Train)//batch_size,
			epochs=epochs,
			validation_data=(X_Train, Y_Train),
			callbacks=callbacks)



if __name__ == '__main__':
	main()