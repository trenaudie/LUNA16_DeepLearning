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


class Dataset_Images(object):
    def __init__(self, train = True):
        self.tensorcifar = datasets.CIFAR10("./data/cifar/",train = True, download = False, transform = transforms.ToTensor())
    def __getitem__(self,index): 
        return self.tensorcifar[index]
    def __len__(self):
        return len(self.tensorcifar)