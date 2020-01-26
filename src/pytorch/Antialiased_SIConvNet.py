import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from SI_ConvNet import *
from torch.nn.parameter import Parameter
import math
from utils import *
import torch.nn.functional

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Net_antialiased_scaleinvariant_oral_cancer(object):
    """docstring for Net_antialiased_scaleinvariant_oral_cancer"""
    def __init__(self, arg):
        super(Net_antialiased_scaleinvariant_oral_cancer, self).__init__()
        
        lays = [30, 60, 90, 120, 150]
        kernel_sizes = [11, 11, 11, 11, 11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        self.conv1 = ScaleInvariance_Layer(3, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                           padding=pads[0], scale_range=np.arange(0.5,1.0,0.1))
        self.conv2 = ScaleInvariance_Layer(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1,
                                           padding=pads[1], scale_range=np.arange(0.5,1.0,0.1))
        self.conv3 = ScaleInvariance_Layer(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1,
                                           padding=pads[2], scale_range=np.arange(0.5,1.0,0.1))
        self.conv4 = ScaleInvariance_Layer(lays[2], lays[3], [kernel_sizes[3], kernel_sizes[3]], 1,
                                           padding=pads[3], scale_range=np.arange(0.5,1.0,0.1))
        self.conv5 = ScaleInvariance_Layer(lays[3], lays[4], [kernel_sizes[4], kernel_sizes[4]], 1,
                                           padding=pads[4], scale_range=np.arange(0.5,1.0,0.1))

        # self.pool1 = nn.MaxPool2d(2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down1 = Downsample(channels=lays[0], filt_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(lays[0])

        # self.pool2 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down2 = Downsample(channels=lays[1], filt_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(lays[1])

        # self.pool3 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down3 = Downsample(channels=lays[2], filt_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(lays[2])

        # self.pool4 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down4 = Downsample(channels=lays[3], filt_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(lays[3])

        # self.pool5 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.down5 = Downsample(channels=lays[4], filt_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(lays[4])

        self.fc1 = nn.Linear(600, 256)
        
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Linear(256, 2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(self.relu(x))

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(self.relu(x))

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(self.relu(x))

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(self.relu(x))

        x = self.conv5(x)
        x = self.pool5(x)
        xm = self.bn5(self.relu(x))

        xm = torch.flatten(xm,1)
        xm = self.fc1(xm)
        xm = self.relu(xm)
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = F.log_softmax(xm, dim=1)

        return xm

