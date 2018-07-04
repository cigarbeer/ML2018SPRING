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
import time
from keras import backend as K


class Data:
	def __init__(self):
		pass

	def parsetest(self, size):
		X_Test = []
		X_Testname = []
		df = pd.read_csv(argv[2])
		for n in range(len(df['Image'])):
			img_path = os.path.join(argv[1], df['Image'][n])
			img = image.load_img(img_path, target_size=(size, size))
			imarr = image.img_to_array(img)
			X_Test.append(imarr)
			X_Testname.append(df['Image'][n])
		np.save('X_Test.npy', X_Test)


	def preprocesstest(self):
		X_Test = np.load('X_Test.npy')
		X_Test = X_Test/255.
		np.save('X_Test.npy', X_Test)

	def predict(self, modelpath, X_Test, rdictionary, batch_size, name):
		model = load_model(modelpath)
		pred = model.predict(X_Test, verbose=1, batch_size=batch_size)
		np.save('pred%s.npy'%name, pred)

	def writefile(self):
		pred52 = np.load('pred52.npy')
		df = pd.read_csv(argv[2], index_col=False)


		for i in range(len(df['Image'])):
			df['Id'].iloc[i] = ' '.join(str(x) for x in pred52[i])


		df.to_csv(argv[3], index = False)


def main():

	tStart = time.time()
	
	w = Data()
	
	size = 224
	print('----------parsetest----------')
	w.parsetest(size)
	print('----------preprocesstest----------')
	w.preprocesstest()

	dictionary = pickle.load(open('dictionary.pkl', 'rb'))
	rdictionary = {dictionary[key]:key for key in dictionary}

	
	X_Test = np.load('X_Test.npy')

	

	models = ['res1.h5', 'nonew3.h5', 'iv32.h5', 'tune4.h5', 'tune3.h5', 'vgg192.h5']
	for model in models:	
		if model == 'vgg192.h5':
			batch_size = 64
		else:
			batch_size = 128
		print('----------predict %s----------'%model)
		w.predict(os.path.join('model', model), X_Test, rdictionary, batch_size, model.split('.')[0])
		print('----------finished predict %s----------'%model)
		K.clear_session()

	
	print('----------ensemble----------')
	new_whale = []

	predtune3 = np.load('predtune3.npy')
	predvgg192 = np.load('predvgg192.npy')
	prednonew3 = np.load('prednonew3.npy')
	prediv32 = np.load('prediv32.npy')
	predtune4 = np.load('predtune4.npy')
	predres1 = np.load('predres1.npy')
	
	for pre in predvgg192:
		new_whale.append(0)

	new_whale = np.array(new_whale)
	new_whale = new_whale.reshape((new_whale.shape[0],1))
	
	predvgg192 = np.concatenate((new_whale, predvgg192), axis=1)
	prednonew3 = np.concatenate((new_whale, prednonew3), axis=1)
	predres1 = np.concatenate((new_whale, predres1), axis=1)

	for n in range(len(predtune3)):
		predtune3[n][0] = 0
		prediv32[n][0] = 0
		predtune4[n][0] = 0


	for n in range(len(predtune3)):
		tune3max = np.argmax(predtune3[n])
		vgg192max = np.argmax(predvgg192[n])
		nonew3max = np.argmax(prednonew3[n])
		iv32 = np.argmax(prediv32[n])
		tune4max = np.argmax(predtune4[n])
		res1max = np.argmax(predres1[n])

		
		if tune3max == vgg192max and tune3max == nonew3max:
			predtune3[n][tune3max] = 0.9
			predvgg192[n][tune3max] = 0.9
			prednonew3[n][tune3max] = 0.9
		

		maxL = [tune3max, vgg192max, nonew3max, res1max]
		counter = Counter(maxL)
		
		for i in counter:
			if counter[i]>=3:
				predtune3[n][i] = 0.9
				predvgg192[n][i] = 0.9
				prednonew3[n][i] = 0.9
				prediv32[n][i] = 0.9
				predtune4[n][i] = 0.9
				predres1[n][i] = 0.9

	pred = (predtune3+prednonew3+predvgg192+prediv32+predtune4+predres1)/6

	for i in range(len(pred)):
		pred[i][0] = 0.55

	
	pred5 = []

	for p in pred:
		pred5.append(heapq.nlargest(5, range(len(p)), p.take))
	

	pred5 = np.array(pred5)
		
	pred52 = []
	for p in pred5:
		p2 = [rdictionary[n] for n in p]
		pred52.append(p2)
	np.save('pred52.npy', pred52)

	print('----------write file----------')
	w.writefile()

	tEnd = time.time()

	print('Time cost: %d'%(tEnd-tStart))



if __name__ == '__main__':
	main()