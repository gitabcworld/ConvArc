from __future__ import print_function

import os
import os.path
import sys

import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import cv2
from dataset.dataModel import DataModel
from BanknoteBase import BanknoteBase

class ROIBanknoteBase(BanknoteBase):
    def __init__(self, root, train='train',
                 transform=None, target_transform=None,
                 partition=0, dpi=600,
                 counterfeitLabel=True, fixed_size=(64, 64),
                 use_memory=True):
        BanknoteBase.__init__(self, root, train, transform, target_transform,
                              partition, dpi)

        self.size = fixed_size
        self.counterfeitLabel = counterfeitLabel
        self.use_memory = use_memory
        self.memory_imgs = {}

        # counterfeitLabel:
        #   * True: the labels of the images are counterfeit 0 or genuine 1.
        #   * False: the genuine labels are rearranged to be between [0 .. num models]
        #            and the counterfeit models are added as different labels. So in the end the labels
        #            ends as [0 .. num models] + counterfeit_models
        self.dictIndexModel = {i: self.data.keys()[i] for i in range(len(self.data.keys()))}
        self.dictModelIndex = {self.data.keys()[i]: i for i in range(len(self.data.keys()))}

        # find models with counterfeit images
        counterfeitModels = []
        for model in self.data.keys():
            if len(np.unique(self.data[model]['labels'][train][partition])) > 1:
                counterfeitModels.append(model)

        maxIndexModel = len(self.data.keys())
        self.dictIndexModelCounterfeit = {i: counterfeitModels[i - maxIndexModel] for i in \
                                          range(maxIndexModel, maxIndexModel + len(counterfeitModels))}
        self.dictModelIndexCounterfeit = {key: index for index, key in enumerate(self.dictIndexModelCounterfeit.keys())}

        # Add nRois for each model
        for model in self.data.keys():
            self.data[model]['rois'] = {}
            self.data[model]['rois']['rois_coord'] = DataModel(model=model).getRoiData()
            self.data[model]['rois']['nrois'] = len(self.data[model]['rois']['rois_coord'])

        for model in self.data.keys():
            self.data[model]['rois']['inputs'] = {}
            for i, strPathInput in enumerate(self.data[model]['inputs'][self.train][self.partition]):
                # change the word in the path dataset by datasetRois
                strPathRoiInput = strPathInput.replace('dataset', 'datasetRois')
                label = self.data[model]['labels'][train][partition][i]
                pathsRois = []
                for nroi in range(self.data[model]['rois']['nrois']):
                    _, file_extension = os.path.splitext(strPathRoiInput)
                    pathsRois.append(strPathRoiInput[:-1 * len(file_extension)] + '_roi_%d_lbl_%d'
                                     % (nroi, label) + strPathInput[-1 * len(file_extension):])
                self.data[model]['rois']['inputs'][strPathInput] = pathsRois

    # resize Roi images
    def resizeFixed(self, img):
        height, width, channels = img.shape
        interpol_method = cv2.INTER_AREA if self.size[0] * self.size[1] > height * width else cv2.INTER_LINEAR
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=interpol_method)
        return img

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class ROIBanknote(ROIBanknoteBase):
    def __init__(self, root, train='train',
                 transform=None, target_transform=None,
                 partition=0, dpi=600,
                 counterfeitLabel=True, fixed_size=(64, 64),
                 use_memory=True):
        ROIBanknoteBase.__init__(self, root, train,
                                 transform, target_transform,
                                 partition, dpi,
                                 counterfeitLabel, fixed_size,
                                 use_memory)

        # Count all images
        self.nImages = 0
        for model in self.data.keys():
            for strPathInput in self.data[model]['inputs'][self.train][self.partition]:
                self.nImages += len(self.data[model]['rois']['inputs'][strPathInput])

    def getImagePathLabel(self, index):

        num_images_by_model = \
            [len(self.data[model]['inputs'][self.train][self.partition]) * self.data[model]['rois']['nrois']
             for model in self.data.keys()]
        model_idx = [i for i, val in enumerate(np.cumsum(num_images_by_model) <= index) if val == True]
        model_idx = 0 if model_idx == [] else model_idx[-1] + 1
        model = self.data.keys()[model_idx]
        if model_idx > 0:
            idx_img_tmp = index - np.cumsum(num_images_by_model)[model_idx - 1]
            idx_img = int(idx_img_tmp / self.data[model]['rois']['nrois'])
            idx_roi = idx_img_tmp % self.data[model]['rois']['nrois']
        else:
            idx_img = int(index / self.data[model]['rois']['nrois'])
            idx_roi = index % self.data[model]['rois']['nrois']

        strInput = self.data[model]['inputs'][self.train][self.partition][idx_img]
        return self.data[model]['rois']['inputs'][strInput][idx_roi], \
               self.data[model]['labels'][self.train][self.partition][idx_img]

    def __getitem__(self, index):

        path, target = self.getImagePathLabel(index)

        img = None
        if self.use_memory:
            if index in self.memory_imgs.keys():
                img = self.memory_imgs[index]

        if img is None:
            img = cv2.imread(path)
            height, width, channels = img.shape
            # Resize the image to the working dpis
            img = self.resizeFixed(img)
            if self.use_memory:
                self.memory_imgs[index] = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.nImages

