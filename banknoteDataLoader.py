import cv2
import numpy as np
import torch.utils.data
from tqdm import tqdm
import math
import psutil
import gc
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from models.customResnet50 import CustomResNet50
import pdb

from option import Options
from dataset.banknote_pytorch import FullBanknotePairs, FullBanknote, FullBanknoteOneShot, FullBanknoteTriplets

#torch.utils.data.dataloader.default_collate = (lambda default_collate = torch.utils.data.dataloader.default_collate: \
#                                                    lambda batch: batch if all(map(torch.is_tensor, batch)) \
#                                                    and any([tensor.size() != batch[0].size() for tensor in batch]) else default_collate(batch))()

class banknoteDataLoader():
    def __init__(self,type=FullBanknotePairs, opt=Options().parse(), fcn = None, train_mean=None, train_std=None):
        self.type = type
        self.opt = opt
        self.fcn = fcn

        if train_mean is None and train_std is None:
            self.train_mean = None
            self.train_std = None
        else:
            self.train_mean = train_mean
            self.train_std = train_std

    def getlstTransforms(self, train = 'train'):
        lst_transforms = []

        if not(self.opt.imageSize is None):
            lst_transforms.append(transforms.Resize((self.opt.imageSize,self.opt.imageSize)))
        
        if train == 'train':
            lst_transforms.append(transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4))
            lst_transforms.append(transforms.RandomAffine(degrees=(0,10), translate=(0.1, 0.1), scale=(0.8, 1.2)))
            if self.opt.imageSize is None:
                lst_transforms.append(transforms.RandomCrop(size=224))
        if not(train == 'train'):
            lst_transforms.append(transforms.CenterCrop(size=224))

        lst_transforms.append(transforms.ToTensor())

        if not(self.train_mean == None) and not(self.train_std == None):
            lst_transforms.append(transforms.Normalize(torch.from_numpy(self.train_mean),torch.from_numpy(self.train_std)))

        if self.opt.fcn_applyOnDataLoader:
            lst_transforms.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
            if self.opt.cuda:
                lst_transforms.append(transforms.Lambda(lambda x: Variable(x.cuda(), requires_grad=True)))
            else:
                lst_transforms.append(transforms.Lambda(lambda x: Variable(x, requires_grad=True)))
            lst_transforms.append(transforms.Lambda(lambda x: self.fcn(x)))
        return lst_transforms

    def get_mean_std(self):

        lst_transforms = []
        if not(self.opt.imageSize is None):
            lst_transforms.append(transforms.Resize((self.opt.imageSize,self.opt.imageSize)))
        
        lst_transforms.append(transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4))
        lst_transforms.append(transforms.RandomAffine(degrees=(0,10), translate=(0.1, 0.1), scale=(0.8, 1.2)))
        lst_transforms.append(transforms.ToTensor())
        train_transform = transforms.Compose(lst_transforms)

        kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        if self.type == FullBanknotePairs or self.type == FullBanknote:
            train_loader_mean_std = torch.utils.data.DataLoader(
                FullBanknote(setType=self.opt.setType, root=self.opt.dataroot, train='train', size = self.opt.imageSize,
                                    mode = 'generator_processor', path_tmp_data = self.opt.path_tmp_data,
                                    transform=train_transform, target_transform=None),
                #batch_size=self.opt.batchSize, shuffle=True, collate_fn = torch.utils.data.dataloader.default_collate, **kwargs)
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

    def get(self, dataPartition = ['train','val','test'] ,rnd_seed = 42):

        if self.train_mean is None and self.train_std is None and not(self.opt.imageSize is None):
            train_mean, train_std = self.get_mean_std()
            self.train_mean = train_mean
            self.train_std = train_std

        #kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True} if self.opt.cuda else {}
        if self.opt.cuda:
            kwargs = {'num_workers': self.opt.nthread, 'pin_memory': True}    
        else:
            kwargs = {'num_workers': self.opt.nthread}
            #kwargs = {}

        train_transform = transforms.Compose(self.getlstTransforms(train = 'train'))

        if len(dataPartition) > 0 and not(dataPartition[0] is None):
            if self.type == FullBanknote:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[0],
                                            size = self.opt.imageSize,  
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=train_transform, target_transform=None)
            elif self.type == FullBanknotePairs or self.type == FullBanknoteTriplets:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[0], 
                                            size = self.opt.imageSize, numTrials=self.opt.batchSize,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=train_transform, target_transform=None)
            elif self.type == FullBanknoteOneShot:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[0], 
                                            size = self.opt.imageSize,  
                                            transform=train_transform, target_transform=None,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            sameClass = self.opt.datasetBanknoteOneShotSameClass,
                                            n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                            numTrials=self.opt.batchSize)

            train_loader = torch.utils.data.DataLoader(
                datasetParams,
                batch_size=self.opt.batchSize, shuffle=True, **kwargs)
        else:
            train_loader = None

        eval_test_transform = transforms.Compose(self.getlstTransforms(train = 'val_test')) 

        if len(dataPartition) > 0 and not(dataPartition[1] is None):
            if self.type == FullBanknote:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[1], 
                                            size = self.opt.imageSize,  
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=eval_test_transform, target_transform=None)
            elif self.type == FullBanknotePairs or self.type == FullBanknoteTriplets:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[1],
                                            size = self.opt.imageSize, numTrials=self.opt.batchSize,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=eval_test_transform, target_transform=None)
            elif self.type == FullBanknoteOneShot:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[1],
                                            size = self.opt.imageSize,  
                                            transform=train_transform, target_transform=None,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            sameClass = self.opt.datasetBanknoteOneShotSameClass,
                                            n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                            numTrials=self.opt.batchSize)

            val_loader = torch.utils.data.DataLoader(
                datasetParams,
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)
        else:
            val_loader = None

        if len(dataPartition) > 0 and not(dataPartition[2] is None):
            if self.type == FullBanknote:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[2],
                                            size = self.opt.imageSize,  
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=eval_test_transform, target_transform=None)
            elif self.type == FullBanknotePairs or self.type == FullBanknoteTriplets:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[2],
                                            size = self.opt.imageSize, numTrials=self.opt.batchSize,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            transform=eval_test_transform, target_transform=None)
            elif self.type == FullBanknoteOneShot:
                datasetParams = self.type(setType=self.opt.setType, root=self.opt.dataroot, train=dataPartition[2],
                                            size = self.opt.imageSize,  
                                            transform=train_transform, target_transform=None,
                                            mode = self.opt.mode, path_tmp_data = self.opt.path_tmp_data,
                                            sameClass = self.opt.datasetBanknoteOneShotSameClass,
                                            n_way = self.opt.one_shot_n_way, n_shot = self.opt.one_shot_n_shot,
                                            numTrials=self.opt.batchSize)

            test_loader = torch.utils.data.DataLoader(
                datasetParams,
                batch_size=self.opt.batchSize, shuffle=False, **kwargs)
        else:
            test_loader = None

        return train_loader, val_loader, test_loader



