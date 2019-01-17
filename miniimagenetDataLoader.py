import os
import cv2
import numpy as np
import torch.utils.data
import torchnet as tnt

from tqdm import tqdm
import math
import psutil
import gc

import torchvision.transforms as transforms

from option import Options
from dataset.mini_imagenet import MiniImagenet
from dataset.mini_imagenet import MiniImagenetPairs
from dataset.mini_imagenet import MiniImagenetOneShot

class miniImagenetDataLoader():

    def __init__(self,type=MiniImagenet, opt=Options().parse()):
        self.type = type
        self.opt = opt
        self.train_mean = np.array([120.45/255.0,115.74/255.0,104.65/255.0]) # RGB
        self.train_std = np.array([127.5/255.0,127.5/255.0,127.5/255.0])

    def get(self, rnd_seed=None):

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        #kwargs = {}

        #################################
        # TRANSFORMATIONS: transformations for the TRAIN dataset
        #################################
        train_transform = transforms.Compose([            
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.train_mean,self.train_std), 
        ])

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='train', 
                                        size = (self.opt.imageSize,self.opt.imageSize),  
                                        transform=train_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='train', 
                                        size = (self.opt.imageSize,self.opt.imageSize),  
                                        transform=train_transform, target_transform=None,
                                        numTrials=self.opt.numTrials)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='train', 
                                        size = (self.opt.imageSize,self.opt.imageSize),
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.numTrials)

        train_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        eval_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.train_mean,self.train_std), # loaded in PIL RGB.
        ])

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        size = (self.opt.imageSize,self.opt.imageSize),  
                                        transform=eval_test_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        size = (self.opt.imageSize,self.opt.imageSize),
                                        transform=eval_test_transform, target_transform=None,
                                        numTrials=self.opt.numTrials)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        size = (self.opt.imageSize,self.opt.imageSize),
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.numTrials)

        val_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        size = (self.opt.imageSize,self.opt.imageSize),  
                                        transform=eval_test_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        size = (self.opt.imageSize,self.opt.imageSize),
                                        transform=eval_test_transform, target_transform=None,
                                        numTrials=self.opt.numTrials)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        size = (self.opt.imageSize,self.opt.imageSize),
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.numTrials)

        test_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

