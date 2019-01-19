import cv2
import numpy as np
import torch.utils.data
import torchnet as tnt

from tqdm import tqdm
import math
import psutil
import gc

from util import cvtransforms as T
import torchvision.transforms as transforms

from option import Options
from dataset.banknote_pytorch import FullBanknoteROIPairs, FullBanknoteROI, \
                                        ROIBanknotePairs, ROIBanknote

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
list_benchmarks = ['fullBanknoteROIPairs','roiBanknotePairs']

class omniglotBenchMark():
    def __init__(self,type = FullBanknoteROI, opt = Options()):

        self.type = type
        self.opt = opt.parse()

    def __get_mean_std__(self):

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        cv2_scale = lambda x: cv2.resize(x, dsize=(self.opt.imageSize, self.opt.imageSize),
                                         interpolation=cv2.INTER_AREA).astype(np.uint8)
        np_reshape = lambda x: np.reshape(x, (self.opt.imageSize, self.opt.imageSize, 1))
        np_repeat = lambda x: np.repeat(x, self.opt.nchannels, axis=2)

        train_transform = tnt.transform.compose([
            cv2_scale,
            np_reshape,
            np_repeat,
            transforms.ToTensor(),
        ])

        if self.type == FullBanknoteROIPairs:
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknoteROI(root=self.opt.dataroot,
                                  train='train', transform=train_transform, target_transform=None),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)
        else:
            train_loader_mean_std = torch.utils.data.DataLoader(
                ROIBanknote(root=self.opt.dataroot,
                                train='train', transform=train_transform, target_transform=None),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        print('Calculate mean and std for training set....')
        pbar = tqdm(enumerate(train_loader_mean_std))
        tmp = []
        for batch_idx, (data, labels) in pbar:
            tmp.append(data)
            pbar.set_description(
                '[{}/{} ({:.0f}%)]\t'.format(
                    batch_idx * len(data), len(train_loader_mean_std.dataset),
                    100. * batch_idx / len(train_loader_mean_std)))
            # Memory problems if we acumulate the images.
            free_mem = psutil.virtual_memory().available / (1024.0 ** 3)
            # print('Free mem: %f' % free_mem)
            if int(math.floor(free_mem)) == 0:
                break

        train_mean = torch.cat(tmp).mean()
        train_std = torch.cat(tmp).std()

        # Free memory
        tmp = []
        data = []
        labels = []

        # Free memory
        train_loader_mean_std.dataset.clear()
        train_loader_mean_std = None
        gc.collect()

        return train_mean, train_std

    def get(self):

        train_mean, train_std = self.__get_mean_std__()
        #train_mean = 0.07793916504149288
        #train_std = 0.23499299757513684

        self.train_mean = train_mean
        self.train_std = train_std

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

        cv2_scale = lambda x: cv2.resize(x, dsize=(self.opt.imageSize, self.opt.imageSize),
                                         interpolation=cv2.INTER_AREA).astype(np.uint8)
        np_reshape = lambda x: np.reshape(x, (self.opt.imageSize, self.opt.imageSize, 1))
        np_repeat = lambda x: np.repeat(x, self.opt.nchannels, axis=2)
        np_mul = lambda x: x*255.0

        #################################
        # TRANSFORMATIONS: transformations for the TRAIN dataset
        #################################
        train_transform = tnt.transform.compose([
            cv2_scale,
            np_reshape,
            np_repeat,
            T.AugmentationAleju(channel_is_first_axis=False,
                                scale_to_percent = self.opt.scale,
                                hflip=self.opt.hflip, vflip=self.opt.vflip,
                                rotation_deg=self.opt.rotation_deg, shear_deg=self.opt.shear_deg,
                                translation_x_px=self.opt.translation_px,
                                translation_y_px=self.opt.translation_px),
            np_mul,
            T.Normalize(train_mean, train_std),
            transforms.ToTensor(),
        ])

        if self.type == ROIBanknote:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', transform=train_transform, target_transform=None,
                                      isWithinAlphabets=self.opt.isWithinAlphabets)
        elif self.type == ROIBanknotePairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)
        elif self.type == FullBanknoteROI:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', transform=train_transform, target_transform=None)
        elif self.type == FullBanknoteROIPairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='train', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)

        train_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        eval_test_transform = tnt.transform.compose([
            cv2_scale,
            np_reshape,
            np_repeat,
            T.Normalize(train_mean, train_std),
            transforms.ToTensor(),
        ])

        if self.type == ROIBanknote:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='val', transform=train_transform, target_transform=None,
                                      isWithinAlphabets=self.opt.isWithinAlphabets)
        elif self.type == ROIBanknotePairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='val', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)
        elif self.type == FullBanknoteROI:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='val', transform=train_transform, target_transform=None)
        elif self.type == FullBanknoteROIPairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='val', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)

        val_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == ROIBanknote:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', transform=train_transform, target_transform=None,
                                      isWithinAlphabets=self.opt.isWithinAlphabets)
        elif self.type == ROIBanknotePairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)
        elif self.type == FullBanknoteROI:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', transform=train_transform, target_transform=None)
        elif self.type == FullBanknoteROIPairs:
            datasetParams = self.type(root=self.opt.dataroot,
                                      train='test', transform=train_transform, target_transform=None,
                                      numTrials=self.opt.batchSize)

        test_loader = torch.utils.data.DataLoader(
            datasetParams,
            batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

