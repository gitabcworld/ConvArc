import os
import cv2
import numpy as np
import torch.utils.data

from tqdm import tqdm
import math
import psutil
import gc

from torch.autograd import Variable
import torchvision.transforms as transforms

from option import Options
from dataset.mini_imagenet import MiniImagenet
from dataset.mini_imagenet import MiniImagenetPairs
from dataset.mini_imagenet import MiniImagenetOneShot

class miniImagenetDataLoader():

    def __init__(self,type=MiniImagenet, fcn = None, opt=Options().parse()):
        self.type = type
        self.opt = opt
        self.fcn = fcn
        self.train_mean = np.array([120.45/255.0,115.74/255.0,104.65/255.0]) # RGB
        self.train_std = np.array([127.5/255.0,127.5/255.0,127.5/255.0])

    def getlstTransforms(self, train='train'):
        lst_transforms = []

        if not(self.opt.imageSize is None):
            lst_transforms.append(transforms.Resize((self.opt.imageSize,self.opt.imageSize)))
        
        if train == 'train':
            lst_transforms.append(transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4))
            lst_transforms.append(transforms.RandomAffine(degrees=(0,10), translate=(0.1, 0.1), scale=(0.8, 1.2)))
            lst_transforms.append(transforms.RandomVerticalFlip())
            lst_transforms.append(transforms.RandomHorizontalFlip())
        
        lst_transforms.append(transforms.ToTensor())
        #lst_transforms.append(transforms.Lambda(lambda x: (x.float()-torch.from_numpy(self.train_mean).float())/torch.from_numpy(self.train_std).float()))
        lst_transforms.append(transforms.Normalize(self.train_mean,self.train_std))
        
        if self.opt.fcn_applyOnDataLoader:
            lst_transforms.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
            if self.opt.cuda:
                lst_transforms.append(transforms.Lambda(lambda x: Variable(x.cuda(), requires_grad=True)))
            else:
                lst_transforms.append(transforms.Lambda(lambda x: Variable(x, requires_grad=True)))
            lst_transforms.append(transforms.Lambda(lambda x: self.fcn(x)))
        
        return lst_transforms

    def get(self, rnd_seed=None):

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        #kwargs = {}

        #################################
        # TRANSFORMATIONS: transformations for the TRAIN dataset
        #################################
        train_transform = transforms.Compose(self.getlstTransforms(train='train'))

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='train',
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=train_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='train', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=train_transform, target_transform=None,
                                        numTrials=self.opt.batchSize)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='train', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.batchSize)

        train_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        eval_test_transform = transforms.Compose(self.getlstTransforms(train='eval_test'))

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=eval_test_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=eval_test_transform, target_transform=None,
                                        numTrials=self.opt.batchSize)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='val', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.batchSize)

        val_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == MiniImagenet:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=eval_test_transform, target_transform=None)
        elif self.type == MiniImagenetPairs:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=eval_test_transform, target_transform=None,
                                        numTrials=self.opt.batchSize)
        elif self.type == MiniImagenetOneShot:
            datasetParams = self.type(root=self.opt.dataroot, train='test', 
                                        datasetCompactSize = self.opt.datasetCompactSize,
                                        size = self.opt.imageSize,  
                                        transform=train_transform, target_transform=None,
                                        n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                        numTrials=self.opt.batchSize)

        test_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

