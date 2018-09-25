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
from tqdm import tqdm
import itertools
import torchnet as tnt
from dataset.dataModel import DataModel
import torchvision.transforms as transforms
import util.cvtransforms as T
import torch

cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
np_reshape = lambda x: np.reshape(x, (32, 32, 1))


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

class FullBanknoteBase(BanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True, size_mode = None,
                     use_memory = True):
        BanknoteBase.__init__(self, root, train, transform, target_transform,
                              partition, dpi)

        self.size_mode = size_mode
        self.counterfeitLabel = counterfeitLabel
        self.use_memory = use_memory
        self.memory_imgs = {}

        # counterfeitLabel:
        #   * True: the labels of the images are counterfeit 0 or genuine 1.
        #   * False: the genuine labels are rearranged to be between [0 .. num models]
        #            and the counterfeit models are added as different labels. So in the end the labels
        #            ends as [0 .. num models] + counterfeit_models
        self.dictIndexModel = {i: self.data.keys()[i] for i in range(len(self.data.keys()))}
        self.dictModelIndex = {self.data.keys()[i]:i for i in range(len(self.data.keys()))}

        # find models with counterfeit images
        self.counterfeitModels = []
        for model in self.data.keys():
            if len(np.unique(self.data[model]['labels'][train][partition]))>1:
                self.counterfeitModels.append(model)

        maxIndexModel = len(self.data.keys())
        self.dictIndexModelCounterfeit = {i: self.counterfeitModels[i-maxIndexModel] for i in \
                                     range(maxIndexModel,maxIndexModel+len(self.counterfeitModels))}
        self.dictModelIndexCounterfeit = {key:index for key,index in zip(self.dictIndexModelCounterfeit.values(),
                                                                         self.dictIndexModelCounterfeit.keys())}

        # different return image sizes depending on the mode.
        # 1. padding: add zeros to the smaller images to make equal as the bigger image
        # 2. decrease: shrink bigger images until all dataset is equal to the smaller image.
        # 3. increase: resize all small images to be equal to the bigger image.
        # 4. fixed: resize all images to the same size.
        if self.size_mode:
            temp_sizes = np.array([self.data[model]['size'] for model in self.data.keys()])
            if self.size_mode == 'padding' or self.size_mode == 'increase':
                self.size = self.data[self.data.keys()[np.argmax(temp_sizes[:, 0] * temp_sizes[:, 1])]]['size']
            elif self.size_mode == 'decrease':
                self.size = self.data[self.data.keys()[np.argmin(temp_sizes[:, 0] * temp_sizes[:, 1])]]['size']
            elif self.size_mode == 'fixed':
                self.size = (640, 640)

    # different return image sizes depending on the mode.
    # 1. padding: add zeros to the smaller images to make equal as the bigger image
    # 2. decrease: shrink bigger images until all dataset is equal to the smaller image.
    # 3. increase: resize all small images to be equal to the bigger image.
    def resizeBySizeMode(self,img):
        height, width, channels = img.shape
        interpol_method = cv2.INTER_AREA if self.size[0] * self.size[1] > height * width else cv2.INTER_LINEAR
        if not (self.size_mode == 'padding'):
            img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=interpol_method)
        else:
            # Add padding
            hpadding = int((self.size[1] - width) / 2)
            vpadding = int((self.size[0] - height) / 2)
            if hpadding < 0:
                hpadding = 0
            if vpadding < 0:
                vpadding = 0
            color = [0, 0, 0]
            img = cv2.copyMakeBorder(img, vpadding, vpadding, hpadding, hpadding,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=color)
            img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class FullBanknote(FullBanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True, size_mode = None,
                     use_memory = True):
        FullBanknoteBase.__init__(self, root, train,
                                    transform, target_transform,
                                    partition, dpi,
                                    counterfeitLabel, size_mode,
                                    use_memory)

        # Count all images
        self.nImages = \
            np.sum([len(self.data[model]['inputs'][self.train][self.partition]) for model in self.data.keys()])

    def getImagePathLabel(self, index):

        num_images_by_model = [len(self.data[model]['inputs'][self.train][self.partition]) for model in self.data.keys()]
        model_idx = [i for i, val in enumerate(np.cumsum(num_images_by_model) <= index) if val == True]
        model_idx = 0 if model_idx == [] else model_idx[-1]+1
        model = self.data.keys()[model_idx]
        if model_idx > 0:
            idx_img = index - np.cumsum(num_images_by_model)[model_idx-1]
        else:
            idx_img = index
        return self.data[model]['inputs'][self.train][self.partition][idx_img], \
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
            if self.size_mode:
                img = self.resizeBySizeMode(img)
            if self.use_memory:
                self.memory_imgs[index] = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.nImages


'''
This class adds the padding to the full banknote and then selects a random ROI
params:
        - counterfeitLabel: True, returns 0 or 1 labels. False, returns num classes
        which is equal to numClassesGenuine + numClassesCounterfeit.
        - roisize: size of the ROI
        - nCroppedRois: random num ROIs cropped at each document.
        - nPairs: num of image Pairs of the dataset.
        - percentagePairs: 3 percentages that sum 1, default = [0.5, 0.25, 0.25].
        The first percentage is the % of num of genuine pairs. The second percentage is the
        % of genuine-counterfeit pairs. The third percentage is pairs of different classes.
        
'''
class FullBanknoteROIPairs(FullBanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True,
                     roiSize=32,
                     nCroppedRois = 1,
                     nPairs = 1500,
                     percentagePairs = [0.5,0.25,0.25],
                     use_memory=False):
        FullBanknoteBase.__init__(self, root, train,
                                    transform, target_transform,
                                    partition, dpi,
                                    counterfeitLabel, size_mode = 'padding',
                                    use_memory = use_memory)
        self.nCroppedRois = nCroppedRois
        self.roiSize = roiSize
        self.percentagePairs = percentagePairs
        self.nPairs = nPairs

        # Count all images
        self.nImages = \
            np.sum([len(self.data[model]['inputs'][self.train][self.partition]) for model in self.data.keys()])

        if counterfeitLabel:
            self.numClasses = 2
        else:
            self.numClasses = len(self.dictModelIndex.keys()) + len(self.dictModelIndexCounterfeit.keys())

        # Generate nPairs
        self.generate_pairs(self.nPairs)


    def generate_pairs(self, nPairs):
        # pair tuples selected
        self.data_idx = []

        # Select N/2 positive pairs
        modelsWithGenuine = self.dictModelIndex.keys()
        idx_model_genuine = np.random.choice(len(modelsWithGenuine), int(nPairs*self.percentagePairs[0]), replace=True)
        for idx_model in idx_model_genuine:
            model = self.dictModelIndex.keys()[idx_model]
            prob = np.ones(len(self.data[model]['inputs'][self.train][self.partition]))
            prob = prob * self.data[model]['labels'][self.train][self.partition]
            prob = prob / np.sum(prob)
            idx_selected = np.random.choice(len(self.data[model]['inputs'][self.train][self.partition]),
                                            2, replace=False,p=prob)
            self.data_idx.append([(model,idx_selected[0]),(model,idx_selected[1])])

        # Select N/4 negative pairs same class. Counterfeits.
        modelsWithCounterfeit = self.dictModelIndexCounterfeit.keys()
        idx_model_counterfeit = np.random.choice(len(modelsWithCounterfeit), int(nPairs*self.percentagePairs[1]), replace=True)
        for idx_model in idx_model_counterfeit:
            model = self.dictModelIndexCounterfeit.keys()[idx_model]
            prob_genuine = np.ones(len(self.data[model]['inputs'][self.train][self.partition]))
            prob_genuine = prob_genuine * self.data[model]['labels'][self.train][self.partition]
            prob_genuine = prob_genuine / np.sum(prob_genuine)
            idx_genuine = np.random.choice(len(self.data[model]['inputs'][self.train][self.partition]),
                                            1, replace=False,p=prob_genuine)
            prob_counterfeit = np.ones(len(self.data[model]['inputs'][self.train][self.partition]))
            inverse_labels = (np.invert(self.data[model]['labels'][self.train][self.partition] == 1)).astype(np.int)
            prob_counterfeit = prob_counterfeit * inverse_labels
            prob_counterfeit = prob_counterfeit / np.sum(prob_counterfeit)
            idx_counterfeit = np.random.choice(len(self.data[model]['inputs'][self.train][self.partition]),
                             1, replace=False, p=prob_counterfeit)
            self.data_idx.append([(model,idx_genuine),(model,idx_counterfeit)])

        # Select N/4 negative pairs different class
        for i in range(int(nPairs*self.percentagePairs[2])):
            idx_selected = np.random.choice(len(modelsWithGenuine), 2, replace=False)
            model1 = modelsWithGenuine[idx_selected[0]]
            model2 = modelsWithGenuine[idx_selected[1]]
            idx1 = random.randint(0, len(self.data[model1]['inputs'][self.train][self.partition]) - 1)
            idx2 = random.randint(0, len(self.data[model2]['inputs'][self.train][self.partition]) - 1)
            self.data_idx.append([(model1, idx1), (model2, idx2)])

        # Count all images
        self.nImages = len(self.data_idx)

        # Load all images if needed for memory purposes
        if self.use_memory:
            print('Loading into memory ...')
            for i in tqdm(range(self.nImages)):
                self.__getitem__(i)



    def __getitem__(self, index):

        #for index in range(self.nImages):
        #    print('index: %d.' % index)
        (model1,idx1),(model2,idx2) = self.data_idx[index]
        path1 = self.data[model1]['inputs'][self.train][self.partition][idx1]
        path1 = path1[0] if type(path1) == np.ndarray else path1
        path2 = self.data[model2]['inputs'][self.train][self.partition][idx2]
        path2 = path2[0] if type(path2) == np.ndarray else path2
        target1 = self.data[model1]['labels'][self.train][self.partition][idx1]
        target1 = target1[0] if type(target1) == np.ndarray else target1
        target2 = self.data[model2]['labels'][self.train][self.partition][idx2]
        target2 = target2[0] if type(target2) == np.ndarray else target2

        #print('Memory images: [%d]' % (len(self.memory_imgs.keys())))

        if path1 in self.memory_imgs.keys():
            img1 = self.memory_imgs[path1]
        else:
            img1 = cv2.imread(path1)
            self.memory_imgs[path1] = img1

        if path2 in self.memory_imgs.keys():
            img2 = self.memory_imgs[path2]
        else:
            img2 = cv2.imread(path2)
            self.memory_imgs[path2] = img2

        # Resize to the established dpi
        img1 = cv2.resize(img1, (self.data[model1]['size'][1], self.data[model1]['size'][0]),
                          interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (self.data[model2]['size'][1], self.data[model2]['size'][0]),
                          interpolation=cv2.INTER_AREA)

        def addPadding(img,left_padding, right_padding, top_padding, bottom_padding):
            left_padding = 0 if left_padding < 0 else left_padding
            right_padding = 0 if right_padding < 0 else right_padding
            top_padding = 0 if top_padding < 0 else top_padding
            bottom_padding = 0 if bottom_padding < 0 else bottom_padding
            color = [0, 0, 0]
            if left_padding > 0 or right_padding > 0 or top_padding > 0 or bottom_padding > 0:
                img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=color)
            return img

        def addPaddingBottomRight(img,left_padding, right_padding, top_padding, bottom_padding):
            left_padding = 0 if left_padding < 0 else left_padding
            right_padding = 0 if right_padding < 0 else right_padding
            top_padding = 0 if top_padding < 0 else top_padding
            bottom_padding = 0 if bottom_padding < 0 else bottom_padding
            color = [0, 0, 0]
            if left_padding > 0 or right_padding > 0 or top_padding > 0 or bottom_padding > 0:
                img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=color)
            return img

        # Add padding to the smaller image
        height1, width1, channels1 = img1.shape
        height2, width2, channels2 = img2.shape
        # Add padding at bottom and right if needed
        img1 = addPadding(img1, 0, max(width1, width2) - width1, 0, max(height1, height2) - height1)
        img2 = addPadding(img2, 0, max(width1, width2) - width2, 0, max(height1, height2) - height2)

        # Cut nCroppedRois at the same position of the two images.
        concat_img = np.concatenate((img1, img2), 2)
        if self.transform is None:
            rois_arr = np.zeros((self.nCroppedRois, 2, self.roiSize, self.roiSize, channels1),dtype=np.uint8)
        else:
            rois_arr = torch.zeros((self.nCroppedRois, 2, channels1, self.roiSize, self.roiSize))

        for i in range(self.nCroppedRois):
            isCorrectPatch = False
            while not isCorrectPatch:
                concat_rois = T.RandomCrop(self.roiSize)(concat_img)
                # TODO: check ROIs if one of the rois contains a black region.
                isCorrectPatch = True

            if self.transform is not None:
                rois_arr[i, 0] = self.transform(concat_rois[:, :, 0:3])
                rois_arr[i, 1] = self.transform(concat_rois[:, :, 3:6])
            else:
                rois_arr[i, 0] = concat_rois[:, :, 0:3]
                rois_arr[i, 1] = concat_rois[:, :, 3:6]

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        # Transform counterfeit to classes
        if self.counterfeitLabel:
            labels = (model1, target1, model2, target2)
        else:
            target1 = self.dictModelIndex[model1] if target1 else self.dictModelIndexCounterfeit[model1]
            target2 = self.dictModelIndex[model2] if target2 else self.dictModelIndexCounterfeit[model2]
            labels = (model1, target1, model2, target2)

        return rois_arr, labels

    def __len__(self):
        return self.nImages


class FullBanknoteROI(FullBanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True,
                     roiSize=32,
                     nCroppedRois = 1,
                     nTrials = 1500,
                     percentagePairs = [0.5,0.25,0.25],
                     use_memory=False):
        FullBanknoteBase.__init__(self, root, train,
                                    transform, target_transform,
                                    partition, dpi,
                                    counterfeitLabel, size_mode = 'padding',
                                    use_memory = use_memory)
        self.nCroppedRois = nCroppedRois
        self.roiSize = roiSize
        self.percentagePairs = percentagePairs
        self.nTrials = nTrials

        # Count all images
        self.nImages = \
            np.sum([len(self.data[model]['inputs'][self.train][self.partition]) for model in self.data.keys()])

        if counterfeitLabel:
            self.numClasses = 2
        else:
            self.numClasses = len(self.dictModelIndex.keys()) + len(self.dictModelIndexCounterfeit.keys())

        # Generate nPairs
        self.generate_samples(self.nTrials)


    def generate_samples(self, nTrials):
        # pair tuples selected
        self.data_idx = []

        # Select N/2 positive
        modelsWithGenuine = self.dictModelIndex.keys()
        idx_model_genuine = np.random.choice(len(modelsWithGenuine), int(nTrials*self.percentagePairs[0]), replace=True)
        for idx_model in idx_model_genuine:
            model = self.dictModelIndex.keys()[idx_model]
            prob = np.ones(len(self.data[model]['inputs'][self.train][self.partition]))
            prob = prob * self.data[model]['labels'][self.train][self.partition]
            prob = prob / np.sum(prob)
            idx_selected = np.random.choice(len(self.data[model]['inputs'][self.train][self.partition]),
                                            1, replace=False,p=prob)
            self.data_idx.append((model,idx_selected))

        # Select N/2 negative. Counterfeits.
        modelsWithCounterfeit = self.dictModelIndexCounterfeit.keys()
        idx_model_counterfeit = np.random.choice(len(modelsWithCounterfeit), int(nTrials*self.percentagePairs[1]), replace=True)
        for idx_model in idx_model_counterfeit:
            model = self.dictModelIndexCounterfeit.keys()[idx_model]
            prob_counterfeit = np.ones(len(self.data[model]['inputs'][self.train][self.partition]))
            inverse_labels = (np.invert(self.data[model]['labels'][self.train][self.partition] == 1)).astype(np.int)
            prob_counterfeit = prob_counterfeit * inverse_labels
            prob_counterfeit = prob_counterfeit / np.sum(prob_counterfeit)
            idx_counterfeit = np.random.choice(len(self.data[model]['inputs'][self.train][self.partition]),
                             1, replace=False, p=prob_counterfeit)
            self.data_idx.append((model,idx_counterfeit))

        # Count all images
        self.nImages = len(self.data_idx)

        # Load all images if needed for memory purposes
        if self.use_memory:
            print('Loading into memory ...')
            for i in tqdm(range(self.nImages)):
                self.__getitem__(i)

    def __getitem__(self, index):

        (model1,idx1),(model2,idx2) = self.data_idx[index]
        path1 = self.data[model1]['inputs'][self.train][self.partition][idx1]
        path1 = path1[0] if type(path1) == np.ndarray else path1
        target1 = self.data[model1]['labels'][self.train][self.partition][idx1]
        target1 = target1[0] if type(target1) == np.ndarray else target1

        if path1 in self.memory_imgs.keys():
            img1 = self.memory_imgs[path1]
        else:
            img1 = cv2.imread(path1)
            self.memory_imgs[path1] = img1

        # Resize to the established dpi
        img1 = cv2.resize(img1, (self.data[model1]['size'][1], self.data[model1]['size'][0]),
                          interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            img1 = self.transform(img1)

        if self.target_transform is not None:
            target1 = self.target_transform(target1)

        # Transform counterfeit to classes
        if not self.counterfeitLabel:
            target1 = self.dictModelIndex[model1] if target1 else self.dictModelIndexCounterfeit[model1]
        labels = (model1, target1)

        return img1, labels

    def __len__(self):
        return self.nImages


class ROIBanknoteBase(BanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True, fixed_size = (64, 64),
                     use_memory = True):
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
        self.dictModelIndex = {self.data.keys()[i]:i for i in range(len(self.data.keys()))}

        # find models with counterfeit images
        counterfeitModels = []
        for model in self.data.keys():
            if len(np.unique(self.data[model]['labels'][train][partition]))>1:
                counterfeitModels.append(model)

        maxIndexModel = len(self.data.keys())
        self.dictIndexModelCounterfeit = {i: counterfeitModels[i-maxIndexModel] for i in \
                                     range(maxIndexModel,maxIndexModel+len(counterfeitModels))}
        self.dictModelIndexCounterfeit = {key:index for index,key in enumerate(self.dictIndexModelCounterfeit.keys())}

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
                    pathsRois.append(strPathRoiInput[:-1*len(file_extension)] + '_roi_%d_lbl_%d'
                                     % (nroi,label) + strPathInput[-1*len(file_extension):])
                self.data[model]['rois']['inputs'][strPathInput] = pathsRois

    # resize Roi images
    def resizeFixed(self,img):
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
                     counterfeitLabel = True, fixed_size = (64, 64),
                     use_memory = True):
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
            [len(self.data[model]['inputs'][self.train][self.partition])*self.data[model]['rois']['nrois']
             for model in self.data.keys()]
        model_idx = [i for i, val in enumerate(np.cumsum(num_images_by_model) <= index) if val == True]
        model_idx = 0 if model_idx == [] else model_idx[-1]+1
        model = self.data.keys()[model_idx]
        if model_idx > 0:
            idx_img_tmp = index - np.cumsum(num_images_by_model)[model_idx-1]
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

# TODO: NOT DONE! Check the code!
class ROIBanknotePairs(ROIBanknoteBase):

    def __init__(self, root, train='train',
                     transform=None, target_transform=None,
                     partition=0, dpi=600,
                     counterfeitLabel = True, fixed_size = (64, 64),
                     use_memory = True):
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
            [len(self.data[model]['inputs'][self.train][self.partition])*self.data[model]['rois']['nrois']
             for model in self.data.keys()]
        model_idx = [i for i, val in enumerate(np.cumsum(num_images_by_model) <= index) if val == True]
        model_idx = 0 if model_idx == [] else model_idx[-1]+1
        model = self.data.keys()[model_idx]
        if model_idx > 0:
            idx_img_tmp = index - np.cumsum(num_images_by_model)[model_idx-1]
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

