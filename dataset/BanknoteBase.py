from __future__ import print_function

import os
import os.path
import sys

import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import math
import random
import cv2
from dataset.dataModel import DataModel

class BanknoteBase(data.Dataset):

    def __init__(self, root, train='train',
                 transform=None, target_transform=None,
                 partition=0, dpi=600):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set
        self.dpi = dpi
        self.partition = partition

        models = [d for d in os.listdir(os.path.join(self.root, 'dataset'))
                if os.path.isdir(os.path.join(os.path.join(self.root, 'dataset'), d))]
        # remove CUROUD1 - CUROUD2. which only contains 2 images each.
        models.remove('CUROUD1')
        models.remove('CUROUD2')

        # Limit the dataset to the counterfeit images.
        models = ['CUEURB1','CUEURB2','CUEURC1','CUEURC2','CUEURD1','CUEURD2','IDESPC1','IDESPC2']

        # Load the data DB
        self.data = {}
        total_size_mem = 0
        total_num_images = 0
        for model in models:
            path_info_file = self.root + '/dataset/' + model + '/'
            name_path_info_file = model + '_minres_400.pkl'
            path_complete = path_info_file + name_path_info_file

            with open(path_complete, 'rb') as f:
                dataDB = pickle.load(f)

            self.data[model] = {}
            self.data[model] = dataDB['dataset_Images']

            # Load the partitions of data. If there is not partition create it.
            dataModel = DataModel(model=model).getModelData()
            pixHeight = int((dpi * dataModel['Height']) / 25.4)
            pixWidth = int((dpi * dataModel['Width']) / 25.4)
            self.data[model]['size'] = (pixHeight,pixWidth)

        # Train contains 70% and Test 30%.
        # Split train set, so the splits will be Train 60%, Validation 10%, Test 30%.
        print('Creating validation set....')
        random.seed(1447)
        for model in self.data.keys():
            nTrain = len(self.data[model]['inputs']['train'][partition])
            nVal = int(math.ceil(nTrain*0.85))
            idxTrain = range(nTrain)
            random.shuffle(idxTrain)
            validationIndx = np.array(idxTrain[nVal:])
            idxTrain = np.array(idxTrain[:nVal])
            # update lists with the validation set.
            self.data[model]['inputs']['val'] = [[] for _ in range(10)]
            self.data[model]['labels']['val'] = [[] for _ in range(10)]
            self.data[model]['inputs']['val'][partition] = \
                np.array(self.data[model]['inputs']['train'][partition])[validationIndx]
            self.data[model]['labels']['val'][partition] = \
                np.array(self.data[model]['labels']['train'][partition])[validationIndx]
            # remove from training set the validation set.
            self.data[model]['labels']['train'][partition] = \
                np.array(self.data[model]['labels']['train'][partition])[idxTrain]
            self.data[model]['inputs']['train'][partition]= \
                np.array(self.data[model]['inputs']['train'][partition])[idxTrain]
            # update test to be arrays
            self.data[model]['labels']['test'][partition] = \
                np.array(self.data[model]['labels']['test'][partition])
            self.data[model]['inputs']['test'][partition] = \
                np.array(self.data[model]['inputs']['test'][partition])

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
