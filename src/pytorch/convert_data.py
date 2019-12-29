import numpy as np
import pickle
from PIL import Image
import os
from matplotlib import pyplot as plt
import random


def make_oral_cancer(val_splits):
	train_data = np.zeros([72000, 80, 80])
	train_label = np.zeros([72000])
	test_data = np.zeros([55000, 80, 80])
	test_label = np.zeros([55000])

	# train data
	i = 0
	for imagename in os.listdir('OralCancer_DataSet3/train/Cancer'):
		image = Image.open(os.path.join('OralCancer_DataSet3/train/Cancer', imagename)).convert('L')
		train_data[i,:,:] = np.asarray(image)
		train_label[i] = 0
		i += 1
		if i == 22000:
			break
	for imagename in os.listdir('OralCancer_DataSet3/train/Healthy'):
		image = Image.open(os.path.join('OralCancer_DataSet3/train/Healthy', imagename)).convert('L')
		train_data[i,:,:] = np.asarray(image)
		train_label[i] = 1
		i += 1
		if i == 72000:
			break
	# test data
	i = 0
	for imagename in os.listdir('OralCancer_DataSet3/test/Cancer'):
		image = Image.open(os.path.join('OralCancer_DataSet3/test/Cancer', imagename)).convert('L')
		test_data[i,:,:] = np.asarray(image)
		test_label[i] = 0
		i += 1
		if i == 20000:
			break
	for imagename in os.listdir('OralCancer_DataSet3/test/Healthy'):
		image = Image.open(os.path.join('OralCancer_DataSet3/test/Healthy', imagename)).convert('L')
		test_data[i,:,:] = np.asarray(image)
		test_label[i] = 1
		i += 1
		if i == 35000:
			break

	try:
		os.mkdir('OralCancer/')
	except:
		None

	os.chdir('OralCancer/')

	for split in range(val_splits):
		random.seed(split)
		perm = np.random.permutation(train_data.shape[0])
		train_data = train_data[perm,:,:]
		train_label = train_label[perm]

		perm = np.random.permutation(test_data.shape[0])
		test_data = test_data[perm,:,:]
		test_label = test_label[perm]

		dict = {}
		dict['train_data'] = train_data
		dict['train_label'] = train_label
		dict['test_data'] = test_data
		dict['test_label'] = test_label

		pickle.dump(dict,open('oral_cancer_split_'+ str(split) +'.pickle','wb'), protocol=4)


if __name__ == '__main__':
	make_oral_cancer(6)
