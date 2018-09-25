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
from optionBanknote import Options

from dataset.FullBanknote import FullBanknote
from dataset.FullBanknote import FullBanknoteROI
from dataset.FullBanknote import FullBanknotePairsROI
from dataset.FullBanknote import FullBanknoteTripletsROI

#from dataset.FullBanknote import FullBanknoteTripletsROI

from dataset.ROIBanknote import ROIBanknote
#from dataset.ROIBanknote import ROIBanknotePairs
#from dataset.ROIBanknote import ROIBanknoteTriplets

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
list_benchmarks = ['fullBanknote',
                   'fullBanknoteROI','fullBanknotePairsROI','fullBanknoteTripletsROI',
                   'BanknoteROI', 'BanknotePairsROI', 'BanknoteTripletsROI']

class banknoteBenchMark():
    def __init__(self,type = 'fullBanknoteROI'):

        self.type = type
        self.opt = Options().parse()

    def __get_mean_std__(self):

        if self.type == 'fullBanknote':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
            #cv2_scale = lambda x: cv2.resize(x, dsize=(self.opt.imageSize, self.opt.imageSize),
            #                     interpolation=cv2.INTER_AREA).astype(np.uint8)

            train_transform = tnt.transform.compose([
                transforms.ToTensor(),
            ])
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknote(root=self.opt.dataroot, train='train',
                             transform=train_transform, target_transform=None,
                             partition=0, dpi=600, counterfeitLabel=True,
                             size_mode='fixed', use_memory=False),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        if self.type == 'fullBanknoteROI':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                transforms.ToTensor(),
            ])
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknoteROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=100, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nImages=500, percentagePairs=[0.5, 0.5],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        if self.type == 'fullBanknotePairsROI':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                transforms.ToTensor(),
            ])
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknotePairsROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=100, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nPairs=500, percentagePairs=[0.5, 0.25,0.25],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

        if self.type == 'fullBanknoteTripletsROI':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
            train_transform = tnt.transform.compose([
                transforms.ToTensor(),
            ])
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknoteTripletsROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=100, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nTriplets=250,
                                use_memory=False),
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

        if self.type == 'fullBanknote':
            a = 0
        if self.type == 'fullBanknoteROI':
            train_mean = torch.cat(tmp).transpose(0, 2).contiguous().view(3, -1).mean(1)
            train_std = torch.cat(tmp).transpose(0, 2).contiguous().view(3, -1).std(1)
        if self.type == 'fullBanknotePairsROI':
            train_mean = torch.cat(tmp).transpose(0, 3).contiguous().view(3, -1).mean(1)
            train_std = torch.cat(tmp).transpose(0, 3).contiguous().view(3, -1).std(1)
        if self.type == 'fullBanknoteTripletsROI':
            train_mean = torch.cat(tmp).transpose(0, 3).contiguous().view(3, -1).mean(1)
            train_std = torch.cat(tmp).transpose(0, 3).contiguous().view(3, -1).std(1)

        # Free memory
        tmp = []
        data = []
        labels = []

        train_mean = train_mean.numpy()
        train_std = train_std.numpy()

        # Free memory
        train_loader_mean_std.dataset.clear()
        train_loader_mean_std = None
        gc.collect()

        return train_mean, train_std

    def get(self):

        train_mean, train_std = self.__get_mean_std__()

        if self.type == 'fullBanknote':
            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                T.AugmentationAleju(channel_is_first_axis = False,
                                    hflip = self.opt.hflip, vflip = self.opt.vflip,
                                    rotation_deg = self.opt.rotation_deg,
                                    shear_deg = self.opt.shear_deg,
                                    translation_x_px = self.opt.translation_px,
                                    translation_y_px = self.opt.translation_px),
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            train_loader = torch.utils.data.DataLoader(
                FullBanknote(root=self.opt.dataroot, train='train',
                             transform=train_transform, target_transform=None,
                             partition=0, dpi=600, counterfeitLabel=True,
                             size_mode='fixed', use_memory=False),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

            eval_test_transform = tnt.transform.compose([
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            val_loader = torch.utils.data.DataLoader(
                FullBanknote(root=self.opt.dataroot, train='val',
                             transform=eval_test_transform, target_transform=None,
                             partition=0, dpi=600, counterfeitLabel=True,
                             size_mode='fixed', use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                FullBanknote(root=self.opt.dataroot, train='test',
                             transform=eval_test_transform, target_transform=None,
                             partition=0, dpi=600, counterfeitLabel=True,
                             size_mode='fixed', use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == 'fullBanknoteROI':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                T.AugmentationAleju(channel_is_first_axis=False,
                                    hflip=self.opt.hflip, vflip=self.opt.vflip,
                                    rotation_deg=self.opt.rotation_deg,
                                    shear_deg=self.opt.shear_deg,
                                    translation_x_px=self.opt.translation_px,
                                    translation_y_px=self.opt.translation_px),
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            train_loader = torch.utils.data.DataLoader(
                FullBanknoteROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nImages=1000, percentagePairs=[0.5, 0.5, 0.0],
                                use_memory=True),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

            eval_test_transform = tnt.transform.compose([
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            val_loader = torch.utils.data.DataLoader(
                FullBanknoteROI(root=self.opt.dataroot, train='val',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nImages=1000, percentagePairs=[0.5, 0.5, 0.0],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                FullBanknoteROI(root=self.opt.dataroot, train='test',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nImages=1000, percentagePairs=[0.5, 0.5, 0.0],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == 'fullBanknotePairsROI':

            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                T.AugmentationAleju(channel_is_first_axis=False,
                                    hflip=self.opt.hflip, vflip=self.opt.vflip,
                                    rotation_deg=self.opt.rotation_deg,
                                    shear_deg=self.opt.shear_deg,
                                    translation_x_px=self.opt.translation_px,
                                    translation_y_px=self.opt.translation_px),
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            train_loader = torch.utils.data.DataLoader(
                FullBanknotePairsROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nPairs=500, percentagePairs=[0.5, 0.25, 0.25],
                                use_memory=True),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

            eval_test_transform = tnt.transform.compose([
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            val_loader = torch.utils.data.DataLoader(
                FullBanknotePairsROI(root=self.opt.dataroot, train='val',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nPairs=1000, percentagePairs=[0.5, 0.25, 0.25],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                FullBanknotePairsROI(root=self.opt.dataroot, train='test',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nPairs=1000, percentagePairs=[0.5, 0.25, 0.25],
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        if self.type == 'fullBanknoteTripletsROI':

            self.opt.batchSize = 25
            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}

            train_transform = tnt.transform.compose([
                T.AugmentationAleju(channel_is_first_axis=False,
                                    hflip=self.opt.hflip, vflip=self.opt.vflip,
                                    rotation_deg=self.opt.rotation_deg,
                                    shear_deg=self.opt.shear_deg,
                                    translation_x_px=self.opt.translation_px,
                                    translation_y_px=self.opt.translation_px),
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            train_loader = torch.utils.data.DataLoader(
                FullBanknoteTripletsROI(root=self.opt.dataroot, train='train',
                                transform=train_transform, target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nTriplets=250,
                                use_memory=True),
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)

            eval_test_transform = tnt.transform.compose([
                T.Normalize(train_mean, train_std),
                transforms.ToTensor(),
            ])

            val_loader = torch.utils.data.DataLoader(
                FullBanknoteTripletsROI(root=self.opt.dataroot, train='val',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nTriplets=1000,
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                FullBanknoteTripletsROI(root=self.opt.dataroot, train='test',
                                transform=eval_test_transform,target_transform=None,
                                partition=0, dpi=600, counterfeitLabel=False,
                                roiSize=64, nCroppedRois=1,
                                nTriplets=1000,
                                use_memory=False),
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

