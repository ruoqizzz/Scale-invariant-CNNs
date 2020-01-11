import numpy as np
import pickle
from PIL import Image
import os
from matplotlib import pyplot as plt
import random
from scipy.ndimage import zoom


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))

        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2


        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        #  trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)


        if trim_top<0:
            out = img
            # print(zoom_factor)

        else:
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]


    # If zoom_factor == 1, just return the input array
    else:
        out = img

    return out




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
		if i == 55000:
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



def make_oral_caner_scale(val_splits):
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
		image = np.asarray(image)
		zoom_factor = 1 + (np.random.rand()*0.3)
		image_scaled = clipped_zoom(image, zoom_factor, order=1)
		test_data[i,:,:] = image_scaled
		test_label[i] = 0
		i += 1
		if i == 20000:
			break
	for imagename in os.listdir('OralCancer_DataSet3/test/Healthy'):
		image = Image.open(os.path.join('OralCancer_DataSet3/test/Healthy', imagename)).convert('L')
		image = np.asarray(image)
		zoom_factor = 1 + (np.random.rand()*0.3)
		image_scaled = clipped_zoom(image, zoom_factor, order=1)
		test_data[i,:,:] = image_scaled
		test_label[i] = 1
		i += 1
		if i == 55000:
			break

	try:
		os.mkdir('OralCancer_Scaled/')
	except:
		None

	os.chdir('OralCancer_Scaled/')

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

		pickle.dump(dict,open('oral_cancer_scaled_split_'+ str(split) +'.pickle','wb'), protocol=4)



if __name__ == '__main__':
	# make_oral_cancer(6)
	make_oral_caner_scale(6)
