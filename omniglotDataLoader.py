import os
import cv2
import numpy as np
import torch.utils.data
import torchnet as tnt

from tqdm import tqdm
import math
import psutil
import gc

import torch
from util import cvtransforms as T
import torchvision.transforms as transforms

from option import Options
from dataset.omniglot import OmniglotOneShot
from dataset.omniglot import Omniglot
from dataset.omniglot import Omniglot_Pairs

'''
Benchmark configurations:
fullBanknote: The resized image of the full banknote is the input.
fullBanknoteROI: Cropped ROIs of the banknote. The cropped ROIs are random.
fullBanknotePairsROI: Pairs of Cropped ROIs of the banknote are given. 
                    The cropped ROIs are random and are at the same possition in both
                    images of the pair. If the images have different sizes the smaller
                    one is filled with padding at bottom and right.
fullBanknoteTripletsROI: Triplets of Cropped ROIs of the banknote are given. 
                    The cropped ROIs are random and are at the same possition in the
                    images of the triplet. If the images have different sizes the smaller
                    one is filled with padding at bottom and right.
BanknoteROI: Rois of the banknote. The ROIs are already cropped at a fixed position and
                    the full banknote is not loaded.
BanknotePairsROI: Pairs of Rois banknotes. The ROIs are already cropped at a fixed position and
                    the full banknote is not loaded.
BanknoteTripletsROI: Triplets of ROIs banknotes. The ROIs are already cropped at a fixed
                    position. The full banknote is not loaded.
'''

class omniglotDataLoader():

    def __init__(self,type=Omniglot, opt=Options().parse(), train_mean=None, train_std=None):
        self.type = type
        self.opt = opt
        if train_mean is None and train_std is None:
            self.train_mean = None
            self.train_std = None
        else:
            self.train_mean = train_mean
            self.train_std = train_std

    def __get_mean_std__(self):

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        cv2_scale = lambda x: cv2.resize(x, dsize=(self.opt.imageSize, self.opt.imageSize),
                                        interpolation=cv2.INTER_AREA).astype(np.uint8)
        np_reshape = lambda x: np.reshape(x, (self.opt.imageSize, self.opt.imageSize, 1))
        np_repeat = lambda x: np.repeat(x, self.opt.nchannels, axis=2)
        np_mul = lambda x: (x * 255.0).astype(np.uint8)

        train_transform = transforms.Compose([
            #cv2_scale,
            np_reshape,
            np_repeat,
            T.AugmentationAleju(channel_is_first_axis=False,
                                scale_to_percent=self.opt.scale,
                                hflip=self.opt.hflip, vflip=self.opt.vflip,
                                rotation_deg=self.opt.rotation_deg, shear_deg=self.opt.shear_deg,
                                translation_x_px=self.opt.translation_px,
                                translation_y_px=self.opt.translation_px),
            np_mul,
            transforms.ToTensor(),
        ])

        train_loader_mean_std = torch.utils.data.DataLoader(
            Omniglot(root=self.opt.dataroot,
                                train='train', transform=train_transform, target_transform=None,
                                partitionType=self.opt.partitionType),
            batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        print('Calculate mean and std for training set....')
        pbar = tqdm(enumerate(train_loader_mean_std))
        tmp = []
        for batch_idx, (data, labels) in pbar:
            tmp.append(data.data.cpu().numpy())
            pbar.set_description(
                '[{}/{} ({:.0f}%)]\t'.format(
                    batch_idx * len(data), len(train_loader_mean_std.dataset),
                    100. * batch_idx / len(train_loader_mean_std)))
            # Memory problems if we acumulate the images.
            free_mem = psutil.virtual_memory().available / (1024.0 ** 3)
            # print('Free mem: %f' % free_mem)
            if int(math.floor(free_mem)) == 0:
                break

        tmp = np.vstack(tmp)
        train_mean = tmp.mean(axis=0)
        train_std = tmp.std(axis=0)

        # Free memory
        tmp = []
        data = []
        labels = []

        train_loader_mean_std = None
        gc.collect()

        return train_mean, train_std

    def get(self, rnd_seed = 42):

        if self.train_mean is None and self.train_std is None:
            train_mean, train_std = self.__get_mean_std__()
            self.train_mean = train_mean
            self.train_std = train_std

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

        cv2_scale = lambda x: cv2.resize(x, dsize=(self.opt.imageSize, self.opt.imageSize),
                                         interpolation=cv2.INTER_AREA).astype(np.uint8)
        np_reshape = lambda x: np.reshape(x, (self.opt.imageSize, self.opt.imageSize, 1))
        np_repeat = lambda x: np.repeat(x, self.opt.nchannels, axis=2)
        np_mul = lambda x: (x*255.0).astype(np.uint8)

        #################################
        # TRANSFORMATIONS: transformations for the TRAIN dataset
        #################################
        train_transform = transforms.Compose([
            #cv2_scale,
            np_reshape,
            np_repeat,
            T.AugmentationAleju(channel_is_first_axis=False,
                                scale_to_percent = self.opt.scale,
                                hflip=self.opt.hflip, vflip=self.opt.vflip,
                                rotation_deg=self.opt.rotation_deg, shear_deg=self.opt.shear_deg,
                                translation_x_px=self.opt.translation_px,
                                translation_y_px=self.opt.translation_px),
            np_mul,
            #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            #T.Normalize(self.train_mean, self.train_std),
            transforms.ToTensor(),
            T.Normalize(torch.from_numpy(self.train_mean).float(), torch.from_numpy(self.train_std).float()),
            #T.Normalize(train_mean, train_std),
        ])

        if self.type == OmniglotOneShot:
            datasetParams = self.type(root=self.opt.dataroot,
                                      reduced_dataset=self.opt.reduced_dataset,
                                      train='train', rnd_seed=rnd_seed, transform=train_transform, target_transform=None,
                                      partitionType=self.opt.partitionType,
                                      n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                      numTrials=self.opt.batchSize)
        elif self.type == Omniglot:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', rnd_seed=rnd_seed, transform=train_transform, target_transform=None,
                                      partitionType=self.opt.partitionType)
        elif self.type == Omniglot_Pairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', rnd_seed=rnd_seed, transform=train_transform, target_transform=None,
                                      partitionType=self.opt.partitionType,
                                      numTrials=self.opt.batchSize)

        train_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        eval_test_transform = transforms.Compose([
            #cv2_scale,
            np_reshape,
            np_repeat,
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #T.Normalize(self.train_mean, self.train_std),
            transforms.ToTensor(),
            T.Normalize(torch.from_numpy(self.train_mean).float(), torch.from_numpy(self.train_std).float()),
        ])

        if self.type == OmniglotOneShot:
            datasetParams = self.type(root=self.opt.dataroot,
                       reduced_dataset=self.opt.reduced_dataset,
                       train='val', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                      partitionType=self.opt.partitionType,
                                      n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                      numTrials=self.opt.batchSize)
        elif self.type == Omniglot:
            datasetParams = self.type(root=self.opt.dataroot,
                                train='val', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                partitionType=self.opt.partitionType)
        elif self.type == Omniglot_Pairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                train='val', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                partitionType=self.opt.partitionType,
                                numTrials=self.opt.batchSize)

        val_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == OmniglotOneShot:
            datasetParams = self.type(root=self.opt.dataroot,
                       reduced_dataset=self.opt.reduced_dataset,
                       train='test', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                      partitionType=self.opt.partitionType,
                                      n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                      numTrials=self.opt.batchSize)
        elif self.type == Omniglot:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                      partitionType=self.opt.partitionType)
        elif self.type == Omniglot_Pairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', rnd_seed=rnd_seed, transform=eval_test_transform, target_transform=None,
                                      partitionType=self.opt.partitionType,
                                      numTrials=self.opt.batchSize)

        test_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

