import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from ScaleSteerableInvariant_Network import *
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


class Net_antialiased_steerinvariant_mnist_scale(nn.Module):
    def __init__(self):
        super(Net_antialiased_steerinvariant_mnist_scale, self).__init__()


        lays = [12, 32, 48]
        kernel_sizes = [11, 11, 11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)

        # Good configuration saved
        #
        # self.conv1 = ScaleConv_steering(1, 20, [kernel_sizes[0], kernel_sizes[0]], 1,
        #                                 padding=pads[0], sigma_phi_range=[np.pi/16],
        #                                  mode=1)
        # self.conv2 = ScaleConv_steering(20, 50, [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi/16],
        #                                 basis_scale = [0.2], mode=1)
        # self.conv3 = ScaleConv_steering(50, 100, [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
        #                                 k_range=[2,3],phase_range=[0,np.pi/4,np.pi/2,3*np.pi/4],
        #                                 # sigma_phi_range=[np.pi / 16],
        #                                 basis_scale = [0.1], mode=1)

        # For less data size
        lays = [30,60,90]

        # For 100 percent data size
        # lays = [30,60,90]

        self.conv1 = ScaleConv_steering(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 16],
                                        k_range = [0.5,1,2], ker_size_range=np.arange(7,19,2),
                                        # stride = 2,
                                        phi_range = np.linspace(0, np.pi, 9),
                                        phase_range = [-np.pi/4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        ker_size_range=np.arange(7,19,2),
                                        # stride=2,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi/16],
                                        mode=1,drop_rate=2)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range = [0.5,1,2], sigma_phi_range=[np.pi/16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        # stride=4,
                                        ker_size_range=np.arange(7,19,2),
                                        # phase_range=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        # sigma_phi_range=[np.pi / 16],
                                        mode=1,drop_rate=4)


        # self.pool1 = nn.MaxPool2d(2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        # Typically, blur kernel M is 3 or 5
        self.down1 = Downsample(channels=lays[0], filt_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(lays[0])

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        # Typically, blur kernel M is 3 or 5
        self.down2 = Downsample(channels=lays[1], filt_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(lays[1])

        self.pool3 = nn.MaxPool2d(kernel_size=8, stride=1, padding=2)
        # Typically, blur kernel M is 3 or 5
        self.down3 = Downsample(channels=lays[2], filt_size=3, stride=8)
        

        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn3_mag = nn.BatchNorm2d(lays[2])

        self.fc1 = nn.Conv2d(lays[2]*4, 256, 1)
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

    def forward(self, x):
        # if self.orderpaths == True:

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.down1(x)
        # print("after down1:")
        # print(x.shape)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.down2(x)
        # print("after down2:")
        # print(x.shape)
        x = self.bn2(x)

        x = self.conv3(x)
        xm = self.pool3(x)
        xm = self.down3(xm)
        # print("after down3:")
        # print(xm.shape)
        
        xm = self.bn3_mag(xm)
        # print(xm.shape)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm


