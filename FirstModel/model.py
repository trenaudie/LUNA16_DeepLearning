import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import namedtuple 
from tqdm import tqdm
from torchvision import datasets, transforms
import os, subprocess
from glob import glob
import functools
import csv
from itertools import islice
import pandas as pd
import SimpleITK as sitk
from logging import Logger
import logging
from Utils.util import importer

layers = [
    nn.Linear(3072,512),
    nn.Tanh(),
    nn.Linear(512,2),
    nn.LogSoftmax(dim = 1)
    ]
model = nn.Sequential(*layers)


class FirstModel(nn.Module):
    """Enter the number of channels at first"""
    def __init__(self, n_chans):
        """Init function, requires nchans"""
        self.conv1 = nn.Conv3d(3,n_chans, padding = 1)
        self.pool1 = nn.MaxPool2d(2,1) #out (B, C, H/2, W/2)
        self.conv2 = nn.Conv3d(n_chans,n_chans*2, padding = 1) #(B, C*2, H/2,W/2)
        self.pool2 = nn.MaxPool2d(2,1) #out (B, C, H/2, W/2)
    
    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        print(out.shape)
        out = self.pool1(out)
        print(out.shape)
        out = nn.functional.relu(self.conv2(out))
        print(out.shape)
        out = self.pool2(out)
        print(out.shape)
        return out

