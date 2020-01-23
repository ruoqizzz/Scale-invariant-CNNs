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
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
				 padding=0, dilation=1, mode=1, scale_range=np.linspace(0,1,9)):
		super(ScaleInvariance_Layer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		# print(stride)
		self.stride = stride
		self.padding = padding
		self.scale_range = scale_range
		self.dilation = dilation
		self.convs = []
		self.mode = mode
		# default groups = 1
		self.bias = None
		# print("out_channels")
		# print(out_channels)
		# print("in_channels")
		# print(in_channels)
		self.weight = Parameter(torch.Tensor(
				self.out_channels, self.in_channels, *kernel_size))
		self.reset_parameters()

	def reset_parameters(self):
		n = self.in_channels
		for k in self.kernel_size:
			n *= k
		stdv = 1. / math.sqrt(n)
		self.weight.data.uniform_(-stdv, stdv)


	def _apply(self, func):
		super(ScaleInvariance_Layer, self)._apply(func)


	def scale(self, input):
		input_copy = input.clone()
		upsamples = []
		# print(self.scale_range)
		for s in self.scale_range:
			# upsampling
			# print(s)
			ups = nn.Upsample(scale_factor=s, mode='bilinear')
			s = ups(input_copy)
			upsamples.append(s)
		return upsamples

	def forward(self, input):
		outputs = []
		orig_size = list(input.data.shape[2:4])
		for i in range(len(self.scale_range)):
			# input_copy = input.copy()
			# ups = nn.Upsample(scale_factor=s, mode='bilinear')
			# size = [0,0]
			# size[0] = int(round(self.scale_range[i]*orig_size[0]))
			# size[1] = int(round(self.scale_range[i]*orig_size[1]))
			input_ups = F.upsample(input, scale_factor=self.scale_range[i], mode='bilinear')
			padding = tuple([self.padding,self.padding])
			input_conv = F.conv2d(input_ups, self.weight, None, self.stride, padding, self.dilation)
			out = F.upsample(input_conv, size = orig_size, mode='bilinear')
			outputs.append(out.unsqueeze(-1))

		strength, _ = torch.max(torch.cat(outputs, -1), -1)
		return F.relu(strength)


class Net_scaleinvariant_mnist_scale(nn.Module):
	"""docstring for Net_scaleinvariant_mnist_scale"""
	def __init__(self):
		super(Net_scaleinvariant_mnist_scale, self).__init__()
		
		kernel_sizes = [11,11,11]
		pads = (np.array(kernel_sizes) - 1) / 2
		pads = pads.astype(int)

		lays = [30,60,90]

		self.conv1 = ScaleInvariance_Layer(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
										   padding=pads[0], scale_range=np.arange(1.0,3.1,0.4))
		self.conv2 = ScaleInvariance_Layer(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1,
										   padding=pads[1], scale_range=np.arange(1.0,3.1,0.4))
		self.conv3 = ScaleInvariance_Layer(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1,
										   padding=pads[2], scale_range=np.arange(1.0,3.1,0.4))
		self.pool1 = nn.MaxPool2d(2)
		self.bn1 = nn.BatchNorm2d(lays[0])

		self.pool2 = nn.MaxPool2d(2)
		self.bn2 = nn.BatchNorm2d(lays[1])

		self.pool3 = nn.MaxPool2d(8, padding=2)
		self.bn3 = nn.BatchNorm2d(lays[2])
		self.bn3_mag = nn.BatchNorm2d(lays[2])

		# self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
		self.fc1 = nn.Linear(lays[2]*4, 256)
		self.fc1bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.7)
		# self.fc2 = nn.Conv2d(256,10,1)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.bn1(self.relu(x))

		x = self.conv2(x)
		x = self.pool2(x)
		x = self.bn2(self.relu(x))

		x = self.conv3(x)
		x = self.pool3(x)
		xm = self.bn3_mag(self.relu(x))

		# xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
		xm = torch.flatten(xm,1)
		xm = self.fc1(xm)
		xm = self.relu(xm)
		xm = self.dropout(xm)
		xm = self.fc2(xm)
		# xm = xm.view(xm.size()[0], xm.size()[1])
		xm = F.log_softmax(xm, dim=1)
		return xm


class Net_scaleinvariant_fmnist_scale(nn.Module):
	"""docstring for Net_scaleinvariant_mnist_scale"""
	def __init__(self):
		super(Net_scaleinvariant_mnist_scale, self).__init__()
		
		kernel_sizes = [11,11,11]
		pads = (np.array(kernel_sizes) - 1) / 2
		pads = pads.astype(int)

		lays = [30,60,90]

		self.conv1 = ScaleInvariance_Layer(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
										   padding=pads[0], scale_range=np.arange(1.0,1.55,0.1))
		self.conv2 = ScaleInvariance_Layer(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1,
										   padding=pads[1], scale_range=np.arange(1.0,1.55,0.1))
		self.conv3 = ScaleInvariance_Layer(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1,
										   padding=pads[2], scale_range=np.arange(1.0,1.55,0.1))
		self.pool1 = nn.MaxPool2d(2)
		self.bn1 = nn.BatchNorm2d(lays[0])

		self.pool2 = nn.MaxPool2d(2)
		self.bn2 = nn.BatchNorm2d(lays[1])

		self.pool3 = nn.MaxPool2d(8, padding=2)
		self.bn3 = nn.BatchNorm2d(lays[2])
		self.bn3_mag = nn.BatchNorm2d(lays[2])

		# self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
		self.fc1 = nn.Linear(lays[2]*4, 256)
		self.fc1bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.7)
		# self.fc2 = nn.Conv2d(256,10,1)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.bn1(self.relu(x))

		x = self.conv2(x)
		x = self.pool2(x)
		x = self.bn2(self.relu(x))

		x = self.conv3(x)
		x = self.pool3(x)
		xm = self.bn3_mag(self.relu(x))

		# xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
		xm = torch.flatten(xm,1)
		xm = self.fc1(xm)
		xm = self.relu(xm)
		xm = self.dropout(xm)
		xm = self.fc2(xm)
		# xm = xm.view(xm.size()[0], xm.size()[1])
		xm = F.log_softmax(xm, dim=1)
		return xm





class Net_scaleinvariant_oral_cancer(nn.Module):
	"""docstring for Net_scaleinvariant_mnist_scale"""
	def __init__(self):
		super(Net_scaleinvariant_oral_cancer, self).__init__()
		
		kernel_sizes = [11,11,11]
		pads = (np.array(kernel_sizes) - 1) / 2
		pads = pads.astype(int)

		lays = [16, 32, 48]

		self.conv1 = ScaleInvariance_Layer(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
										   padding=pads[0], scale_range=np.arange(11,17,2)/11.0)
		self.conv2 = ScaleInvariance_Layer(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1,
										   padding=pads[1], scale_range=np.arange(11,17,2)/11.0)
		self.conv3 = ScaleInvariance_Layer(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1,
										   padding=pads[2], scale_range=np.arange(11,17,2)/11.0)
		self.pool1 = nn.MaxPool2d(2)
		self.bn1 = nn.BatchNorm2d(lays[0])

		self.pool2 = nn.MaxPool2d(2)
		self.bn2 = nn.BatchNorm2d(lays[1])

		self.pool3 = nn.MaxPool2d(8, padding=2)
		self.bn3 = nn.BatchNorm2d(lays[2])

		self.bn3_mag = nn.BatchNorm2d(lays[2])
		self.fc1 = nn.Linear(lays[2]*9, 256)
		self.fc1bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.7)
		self.fc2 = nn.Linear(256, 2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.bn1(self.relu(x))

		x = self.conv2(x)
		x = self.pool2(x)
		x = self.bn2(self.relu(x))

		x = self.conv3(x)
		x = self.pool3(x)
		xm = self.bn3_mag(self.relu(x))

		xm = torch.flatten(xm,1)
		xm = self.fc1(xm)
		xm = F.relu(xm)
		xm = self.dropout(xm)
		xm = self.fc2(xm)
		xm = F.log_softmax(xm, dim=1)
		return xm
