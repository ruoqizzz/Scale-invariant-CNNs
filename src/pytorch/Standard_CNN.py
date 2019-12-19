import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import *
import torch.nn.functional
import numpy as np


from torch.optim.lr_scheduler import StepLR
import os

class standard_CNN_mnist_scale(nn.Module):
	def __init__(self):
		super(standard_CNN_mnist_scale, self).__init__()
		lays = [12, 32, 48]
		kernel_sizes = [11, 11, 11]
		pads = (np.array(kernel_sizes) - 1) / 2
		pads = pads.astype(int)

		self.conv1 = nn.Conv2d(1, lays[0], kernel_sizes[0], stride=1,padding=pads[0])
		self.pool1 = nn.MaxPool2d(2)
		self.bn1 = nn.BatchNorm2d(lays[0])

		self.conv2 = nn.Conv2d(lays[0], lays[1], kernel_sizes[1], stride=1,padding=pads[1])
		self.pool2 = nn.MaxPool2d(2)
		self.bn2 = nn.BatchNorm2d(lays[1])

		self.conv3 = nn.Conv2d(lays[1], lays[2], kernel_sizes[2], stride=1,padding=pads[2])
		self.pool3 = nn.MaxPool2d(8,padding=2)
		self.bn3 = nn.BatchNorm2d(lays[2])
		
		self.bn3_mag = nn.BatchNorm2d(lays[2])
		self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
		self.fc1bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.7)
		self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

	def forward(self,x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.bn2(x)
		x = self.conv3(x)
		xm = self.pool3(x)
		xm = self.bn3_mag(xm)
		xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
		xm = self.fc1(xm)
		xm = self.relu(self.fc1bn(xm))
		xm = self.dropout(xm)
		xm = self.fc2(xm)
		xm = xm.view(xm.size()[0], xm.size()[1])
		return xm


