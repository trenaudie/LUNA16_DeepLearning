
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import namedtuple 
from tqdm import tqdm
from torchvision import datasets, transforms
from glob import glob


import pandas as pd
import SimpleITK as sitk
import logging

#my packages
from Utils.util import importer
from FirstModel.dsets import Dataset_Images
from model import FirstModel


log = logging.Logger(format = '%(filename)s - %(message)s', level=logging.INFO)

class TrainingApp(object):
    def __init__(self, args) -> None:
        log.info("initializing TrainingApp")
        #add cmd line parsing possibility
        if args is None: 
            #get args from cmd line
            pass
        self.model = nn.Sequential([
            nn.Linear(3072,512),
            nn.Tanh(),
            nn.Linear(512,2),
            nn.LogSoftmax(dim = 1)
            ])
        
        #args
        self.lr = 1e-5
        self.optim = torch.optim.Adam(self.model.parameters(),self.lr )
        self.lossfn = nn.NLLLoss()
        self.epochs = 2
        
        
        self.train_dl = self.initDL_Train()
        self.test_dl = self.initDL_Test()
        self.model = self.initModel()

    def initModel(self):
        self.model = FirstModel(32)
    
    def initOptim(self):
        self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        return self.optim

    def initDL_Train(self):
        batch_size = self.batch_size
        ds_train = Dataset_Images(train=True)

        #make train dl and test dl 
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size = batch_size,
            shuffle = True
        )
        #add cuda support for the right batch size
        return dl_train

    def initDL_Test(self):
        ratio = 0.2
        batch_size_test = self.batch_size * ratio
        ds_test = Dataset_Images(train = False)
        #make train dl and test dl 
        dl_test = torch.utils.data.DataLoader(
            ds_test,
            batch_size = batch_size_test,
            shuffle = False)
        #add cuda support for the right batch size
        return dl_test

    def run(self):
        training_losses = []
        test_losses = []
        for i in range(self.epochs):
            
            for (Xtrain, ytrain), (Xtest,ytest) in zip(self.train_dl,self.test_dl):

                ypred = self.model(Xtrain)
                loss_train = self.lossfn(ypred, ytrain)
                training_losses.append(loss_train.item())

                with torch.no_grad():
                    ypred_test = self.model(Xtest)
                    loss_test = self.lossfn(ypred_test, ytest)
                    test_losses.append(loss_test.item())

                self.optim.zero_grad()
                loss_train.backward()
                self.optim.step()

        correct = 0
        total = 0
        for (Xval,Yval) in tqdm(self.test_dl):
            batch_size_test = self.test_dl.batchsize #watchout
            #quick shape test
            c = 0
            if (Xval.view(batch_size_test,-1).shape[1] != self.model[0].weight.shape[1]) :
                c += 1
                print("Error %s", c)
                print(Xval.view(batch_size_test,-1).shape)
                #for the final validation batch, we don't have perfect same size, so we remove it
                continue

            Yval_pred = self.model(Xval.view(batch_size_test,-1))
            _,pred = torch.max(Yval_pred,dim = 1)
            total += batch_size_test
            correct += int((pred == Yval).sum())

        print("Accuracy : %s", correct/total)
        plt.plot(training_losses, label = 'training loss')
        plt.plot(test_losses, label = 'validation loss')
        plt.legend()
        return training_losses,test_losses
    
