from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.xception import Xception
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

class Data:
	def __init__(self):
		pass

	def parse(self, size, imagefilepath, csvpath):
		X = []
		Xname = []
		label = []
		df = pd.read_csv(csvpath)

		for n in range(len(df['Image'])):
			img_path = os.path.join(imagefilepath, df['Image'][n])
			img = image.load_img(img_path, target_size=(size, size))
			imarr = image.img_to_array(img)
			X.append(imarr)
			label.append(df['Id'][n])

		np.save('X_Train.npy', X)
		np.save('label.npy', label)

		X = np.load('X_Train.npy')
		label = np.load('label.npy')

	def mapid(self, csvpath):
		df = pd.read_csv(csvpath)
		self.idmap = {}
		for n in range(len(df['Image'])):
			self.idmap[df['Image'][n]] = df['Id'][n]

	def id2category(self, label, dictionary):
		Y = []
		for i in range(len(label)):
			Y.append(dictionary[label[i]])
		np.save('Y_Train.npy', Y)

	def preprocess(self, X, Y, nclass):
		print('rescale')
		X_Train = X/255.
		np.save('X_Train.npy', X_Train)

		print('to categories')

		from keras.utils import np_utils
		Y_Train = np_utils.to_categorical(Y, nclass)
		np.save('Y_Train.npy', Y_Train)


def main():
	imagefilepath = argv[1]
	csvpath = argv[2]

	w = Data()
	size = 224

	w.mapid(csvpath)
	w.parse(size, imagefilepath, csvpath)
	
	label = np.load('label.npy')
	
	dictionary = pickle.load(open('dictionary.pkl', 'rb'))
	w.id2category(label, dictionary)	
	
	rdictionary = {dictionary[key]:key for key in dictionary}

	X = np.load('X_Train.npy')
	Y = np.load('Y_Train.npy')
	nclass = len(dictionary)
	w.preprocess(X, Y, nclass)

	#find new_whaleID
	df = pd.read_csv(csvpath)
	Ids = df['Id'].values

	delete = []
	for i, Id  in enumerate(Ids):
		if Id == 'new_whale':
			delete.append(i)
	np.save('new_whaleID.npy', delete)

if __name__ == '__main__':
	main()
