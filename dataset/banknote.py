import os
import csv
import cv2
import sys
import pickle
import math
import random

import numpy as np
import torch
from numpy.random import choice
from scipy.misc import imresize as resize
from torch.autograd import Variable
from tqdm import tqdm

from dataset.image_augmenter import ImageAugmenter
from dataset.dataModel import DataModel

use_cuda = False


class Banknote(object):
    def __init__(self, path=os.path.join('/home/aberenguel/Dataset/', 'banknote'),
                 batch_size=128, image_size=32, partition = 0, dpi = 600):
        """
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        data_split: in number of alphabets, e.g. [30, 10] means out of 50 Omniglot characters,
                    30 is for training, 10 for validation and the remaining(10) for testing
        within_alphabet: for verfication task, when 2 characters are sampled to form a pair,
                        this flag specifies if should they be from the same alphabet/language
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
        def getNumImages(data):
            return len(data['inputs']['train'][partition]) + len(data['inputs']['test'][partition])

        def getPathImages(data):
            return data['inputs']['train'][partition] + data['inputs']['test'][partition]

        #models = ["IDESPC1","IDESPC2","CUEURB1","CUEURB2",
        #          "CUEURC1","CUEURC2","CUEURD1","CUEURD2",
        #          "CUEURF1","CUEURF2"]
        models = [d for d in os.listdir(os.path.join(path, 'dataset'))
                if os.path.isdir(os.path.join(os.path.join(path, 'dataset'), d))]

        # remove CUROUD1 - CUROUD2. which only contains 2 images each.
        models.remove('CUROUD1')
        models.remove('CUROUD2')

        self.same_image_size = True
        self.image_size_width = None
        self.image_size_height = None
        self.channels = 3
        maxMemory = 4

        self.data = {}
        total_size_mem = 0
        total_num_images = 0
        for model in models:
            path_info_file = path + '/dataset/' + model + '/'
            name_path_info_file = model + '_minres_400.pkl'
            path_complete = path_info_file + name_path_info_file

            with open(path_complete, 'rb') as f:
                dataDB = pickle.load(f)

            self.data[model] = {}
            self.data[model] = dataDB['dataset_Images']

            # Load the partitions of data. If there is not partition create it.
            dataModel = DataModel(model=model).getModelData()
            pixHeight = (dpi * dataModel['Height'])/ 25.4
            pixWidth = (dpi * dataModel['Width'])/ 25.4
            print('Model: %s, num imgs: %d. Height: %d, Widht: %d. Total Size Mem: %d MB' %
                  (model,getNumImages(self.data[model]), pixHeight, pixWidth,
                   (pixHeight*pixWidth*self.channels)*getNumImages(self.data[model])/(1024*1024) ))
            total_size_mem += (pixHeight*pixWidth*self.channels)*getNumImages(self.data[model])/(1024*1024)
            total_num_images += getNumImages(self.data[model])
        print('+++++++++++++++++++++++++++++++++++++++++++++++')
        print('Num Models: %d, Total imgs: %d, Total size mem dataset: %d GB'
              % (len(models),total_num_images,total_size_mem/1024))
        print('+++++++++++++++++++++++++++++++++++++++++++++++')

        # Load images.
        if os.path.isfile(os.path.join('./data/', models[0] + '.npy')):
            self.data_img = {}
            for model in models:
                pathSaveData = os.path.join('./data/', model + '.npy')
                print('Loading %s' % (pathSaveData))
                self.data_img[model] = np.load(pathSaveData)
                #self.data_img = self.data_img.tolist()

            # check if the images have the same size
            self.same_image_size = True
            for model in self.data_img.keys():
                channels1, height1, width1 = self.data_img[self.data_img.keys()[0]][0].shape
                channels2, height2, width2 = self.data_img[model][0].shape
                if not (channels1 == channels2 and height1 == height2 and width1 == width2):
                    self.same_image_size = False
                    break
        else:

            # Convert all images to the smaller image
            if self.same_image_size:
                # calculate the image size for all images.
                # We get the smaller image size.
                self.image_size_width = sys.maxint
                self.image_size_height = sys.maxint
                for model in models:
                    dataModel = DataModel(model=model).getModelData()
                    pixHeight = (dpi * dataModel['Height']) / 25.4
                    pixWidth = (dpi * dataModel['Width']) / 25.4
                    if (pixHeight*pixWidth < self.image_size_height*self.image_size_width):
                        self.image_size_height = pixHeight
                        self.image_size_width = pixWidth
                print('Selected same size for all images. Height: %d, Width: %d'
                      % (self.image_size_height,self.image_size_width))

                # fit the dataset in memory. Calculate % of resize.
                for resize_factor in np.arange(1.0, 0, -0.1):
                    total_size_mem = 0
                    for model in models:
                        pixHeight = self.image_size_height
                        pixWidth = self.image_size_width
                        pixHeight *= resize_factor
                        pixWidth *= resize_factor
                        total_size_mem += (pixHeight * pixWidth * self.channels) \
                                          * getNumImages(self.data[model]) / (1024.0 * 1024.0)
                    total_size_GB = int(total_size_mem / 1024.0)
                    print('Resize factor: %f, Total size: %d GB' % (resize_factor, total_size_GB))
                    if total_size_GB <= maxMemory:
                        break

                self.image_size_height = int(self.image_size_height * resize_factor)
                self.image_size_width = int(self.image_size_width * resize_factor)

                # resize the images
                self.data_img = {}
                for model in models:
                    print('Resizing images model: %s' % (model))
                    im_array = np.zeros((getNumImages(self.data[model]), self.channels,
                                         self.image_size_height, self.image_size_width),
                                        dtype='uint8')
                    for i, path_img in enumerate(getPathImages(self.data[model])):
                        im = cv2.imread(path_img)
                        imResized = cv2.resize(im, (self.image_size_width,self.image_size_height),
                                               interpolation=cv2.INTER_AREA)
                        imTransposed = imResized.transpose((2,0,1))
                        im_array[i] = imTransposed
                    self.data_img[model] = im_array

            else:
                # fit the dataset in memory. Calculate % of resize.
                for resize_factor in np.arange(1,0,-0.1):
                    total_size_mem = 0
                    for model in models:
                        dataModel = DataModel(model=model).getModelData()
                        pixHeight = (dpi * dataModel['Height']) / 25.4
                        pixWidth = (dpi * dataModel['Width']) / 25.4
                        pixHeight *= resize_factor
                        pixWidth *= resize_factor
                        total_size_mem += (pixHeight * pixWidth * self.channels) * \
                                          getNumImages(self.data[model]) / (1024.0 * 1024.0)
                    total_size_GB = int(total_size_mem / 1024.0)
                    print('Resize factor: %f, Total size: %d GB' % (resize_factor,total_size_GB))
                    if total_size_GB <= maxMemory:
                        break

                total_images = np.sum([len(self.data[d]) for d in self.data.keys()])
                # resize the images
                self.data_img = {}
                for model in models:

                    print('Resizing images model: %s' % (model))
                    dataModel = DataModel(model=model).getModelData()
                    pixHeight = (dpi * dataModel['Height']) / 25.4
                    pixWidth = (dpi * dataModel['Width']) / 25.4
                    pixHeight = int(pixHeight * resize_factor)
                    pixWidth = int(pixWidth * resize_factor)

                    im_array = np.zeros((getNumImages(self.data[model]), self.channels, pixHeight, pixWidth), dtype='uint8')
                    for i, path_img in enumerate(getPathImages(self.data[model])):
                        im = cv2.imread(path_img)
                        imResized = cv2.resize(im, (pixWidth,pixHeight), interpolation=cv2.INTER_AREA)
                        imTransposed = imResized.transpose((2,0,1))
                        im_array[i] = imTransposed
                    self.data_img[model] = im_array

            # Save in small chunks by model. Python 2.7 has a problem with big arrays.
            for model in self.data_img.keys():
                pathSaveData = os.path.join('./data/', model + '.npy')
                np.save(pathSaveData,self.data_img[model])

        print('Calculating mean pixel for dataset...')
        self.mean_pixel = [np.array([self.data_img[model][:, 0, :, :].mean() / 255,
                                      self.data_img[model][:, 1, :, :].mean() / 255,
                                      self.data_img[model][:, 2, :, :].mean() / 255])
                            for model in self.data_img.keys()]
        self.mean_pixel = np.mean(self.mean_pixel,0)

        self.image_size = image_size
        self.batch_size = batch_size

        flip = True
        scale = 0.2
        rotation_deg = 2
        shear_deg = 2
        translation_px = 5

        if self.same_image_size:
            channels, height, width = self.data_img[self.data_img.keys()[0]][0].shape
            self.image_size_height = height
            self.image_size_width = width
            self.augmentor = ImageAugmenter(self.image_size_width, self.image_size_height,
                                        hflip=flip, vflip=flip,
                                        scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                        translation_x_px=translation_px, translation_y_px=translation_px,
                                        channel_is_first_axis = True)
        else:
            self.augmentor = []
            for model in self.data_img.keys():
                channels, height, width = self.data_img[model][0].shape
                self.augmentor.append(ImageAugmenter(width, height,
                                    hflip=flip, vflip=flip,
                                    scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                    translation_x_px=translation_px, translation_y_px=translation_px,
                                    channel_is_first_axis=True))

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
            # re-structure images
            tmp_train = self.data_img[model][:nTrain]
            tmp_val = tmp_train[validationIndx]
            tmp_train = tmp_train[idxTrain]
            tmp_test = self.data_img[model][nTrain:]
            self.data_img[model] = {}
            self.data_img[model]['train'] = tmp_train
            self.data_img[model]['val'] = tmp_val
            self.data_img[model]['test'] = tmp_test

        a = 0


    def fetch_batch(self, part):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                Dissimilar A 		Dissimilar B
                Similar A 			Similar B

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - batch_size

        """
        pass


class BanknoteVerif(Banknote):
    def __init__(self, path=os.path.join('/home/aberenguel/Dataset/', 'banknote'), batch_size=128, image_size=32,
                 partition=0, dpi=600):
        Banknote.__init__(self, path, batch_size, image_size, partition, dpi)

        # each document has different number of documents.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a document by its size. p is that probability
        # Having N models, p = Nx3 where 3 = (train, validation, test)
        self.p = np.array([ (len(self.data[model]['inputs']['train'][partition]), \
            len(self.data[model]['inputs']['val'][partition]), \
            len(self.data[model]['inputs']['test'][partition])) for model in self.data.keys()]).astype(np.float)
        self.p /= np.sum(self.p,0)

        self.partition = partition

    def getModelsWithNegatives(self, part):
        lst_model = []
        for model in self.data.keys():
            if(np.sum(self.data[model]['labels'][part][self.partition] == 0) > 0):
                lst_model.append(model)
        return lst_model

    def fetch_batch(self, part, batch_size = None):

        if batch_size is None:
            batch_size = self.batch_size

        X, Y = self._fetch_batch(part, batch_size)

        channels = 3

        X = Variable(torch.from_numpy(X)).view(2*batch_size, channels, self.image_size_height, self.image_size_width)

        X1 = X[:batch_size]  # (B, c, h, w)
        X2 = X[batch_size:]  # (B, c, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, c, h, w)

        Y = Variable(torch.from_numpy(Y))

        if use_cuda:
            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def _fetch_batch(self, part, batch_size = None):

        if batch_size is None:
            batch_size = self.batch_size

        n_imgs, channels, height, width = self.data_img[self.data.keys()[0]][part].shape
        num_alphbts = len(self.data.keys())

        # half of the patches will have a negative sample of the model
        lst_models_counterfeit = self.getModelsWithNegatives(part)
        idx_model_counterfeit = choice(len(lst_models_counterfeit), batch_size / 2, replace=True)

        X = np.zeros((2 * batch_size, channels, height, width), dtype='uint8')

        offset = 0
        '''
        # Positive samples => size: batch_size / 2
        for i in range(batch_size / 2):
            model = self.data.keys()[choice(range(num_alphbts))]
            idx = np.array(range(len(self.data[model]['labels'][part][self.partition])))
            idx_genuine = idx[self.data[model]['labels'][part][self.partition] == 1]
            X[i + (batch_size / 2)] = self.data_img[model][part][choice(idx_genuine)]
            X[i + batch_size + (batch_size / 2)] = self.data_img[model][part][choice(idx_genuine)]
        offset += i
        '''

        # Negative samples => size: batch_size / 2
        # Same model. Samples genuine - counterfeit
        for i in range(len(idx_model_counterfeit)): # size: batch_size / 2
            model = lst_models_counterfeit[idx_model_counterfeit[i]]
            all_idx = np.array(range(len(self.data[model]['labels'][part][self.partition])))
            idx_genuine = all_idx[self.data[model]['labels'][part][self.partition] == 1]
            idx_counterfeit = all_idx[self.data[model]['labels'][part][self.partition] == 0]
            X[i] = self.data_img[model][part][choice(idx_genuine)]
            X[i + batch_size] = self.data_img[model][part][choice(idx_counterfeit)]
        offset += i

        # Negative samples => size: batch_size / 2
        # Different model. Samples genuine - counterfeit
        for i in range(batch_size / 2): # size: batch_size / 2
            model1 = self.data.keys()[choice(range(num_alphbts))]
            model2 = model1
            while model2 == model1:
                model2 = self.data.keys()[choice(range(num_alphbts))]

            idx1 = np.array(range(len(self.data[model1]['labels'][part][self.partition])))
            idx2 = np.array(range(len(self.data[model2]['labels'][part][self.partition])))
            X[i + offset] = self.data_img[model1][part][choice(idx1)]
            X[i + batch_size + offset] = self.data_img[model2][part][choice(idx2)]

        y = np.zeros((batch_size, 1), dtype='int32')
        y[:batch_size / 2] = 0
        y[batch_size / 2:] = 1

        if part == 'train':
            #X = self.augmentor.augment_batch(X)
            X = X / 255.0
        else:
            X = X / 255.0

        channels = 3
        for i in range(channels):
            X[:, i, :, :] -= self.mean_pixel[i]
        X = X[:, np.newaxis]
        X = X.astype("float32")

        return X, y

