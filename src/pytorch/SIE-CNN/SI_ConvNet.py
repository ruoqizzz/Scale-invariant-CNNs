import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import *
import torch.nn.functional
import numpy as np
from SS_CNN import *
from torchvision import transforms
		

class ScaleInvariance_Layer(nn.Module):
	"""docstring for ScaleInvariance_Layer"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
				 dilation=1, scale_range=np.linspace(0,1,9)):
		super(ScaleInvariance_Layer, self).__init__()
		self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale_range = scale_range


    def scale(self, input):
    	upsamples = []
    	for s in range(len(scale_range)):
    		# upsampling
    		ups = F.upsample_bilinear(input, scale_factor=s)
    		upsamples.append(ups)
    	return upsamples

    def forward(self, input):
    	outputs = []
    	self.upsamples = self.scale(input)
    	for i in range(len(upsamples)):
    		padding = int((self.kernel_size-1)/2)
    		out = F.conv2d(input, self.upsamples[i], None, self.stride, padding, self.dilation)
    		# Undo scaling
    		outputs.append(out.unsqueeze(-1))

    	strength, _ = torch.max(torch.cat(outputs, -1), -1)
    	return F.relu(strength)


class Net_scaleinvariant_mnist_scale(object):
	"""docstring for Net_scaleinvariant_mnist_scale"""
	def __init__(self):
		super(Net_scaleinvariant_mnist_scale, self).__init__()
		
		kernel_size = [11,11,11]
		pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        lays = [30,60,90]

		self.conv1 = ScaleInvariance_Layer(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
        								   padding=pads[0], scale_range=np.arange(7,19,2)/11.0)
		self.conv2 = ScaleInvariance_Layer(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1,
        								   padding=pads[1], scale_range=np.arange(7,19,2)/11.0)
		self.conv2 = ScaleInvariance_Layer(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1,
        								   padding=pads[2], scale_range=np.arange(7,19,2)/11.0)
		self.pool1 = nn.MaxPool2d(2)
		self.bn1 = nn.BatchNorm2d(lays[0])

		self.pool2 = nn.MaxPool2d(2)
		self.bn2 = nn.BatchNorm2d(lays[1])

		self.pool3 = nn.MaxPool2d(8, padding=2)
		self.bn3 = nn.BatchNorm2d(lays[2])
		self.bn3_mag = nn.BatchNorm2d(lays[2])

		self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
		self.fc1bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.7)
		self.fc2 = nn.Conv2d(256,10,1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.bn1(x)

		x = self.conv2(x)
		x = self.pool2(x)
		x = self.bn2(x)

		x = self.conv3(x)
		x = self.pool3(x)
		xm = self.bn3_mag(x)

		xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
		xm = self.fc1(xm)
		xm = self.relu(self.fc1bn(xm))
		xm = self.dropout(xm)
		xm = self.fc2(xm)
		xm = xm.view(xm.size()[0], xm.size()[1])
		return xm

