from __future__ import print_function

import os
import os.path
import sys
import io
import yaml
import numpy as np
import pickle

import torch.utils.data as data
import math
import random
import cv2
from tqdm import tqdm
import itertools
from dataset.dataModel import DataModel
import torchvision.transforms as transforms
import torch
from PIL import Image as pil_image

class BanknoteBase(data.Dataset):

    def __init__(self, root='./banknote', train='train', size = None,
                transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set
        self.size = size
        partition = 0

        path_info_file_partition = os.path.join(self.root,'tmp/data_minres_400_partition_'+ str(partition) + '.yaml')
        # check if exists already the dataset in YAML format
        if os.path.exists(path_info_file_partition):
            print('Loading banknote dataset...')
            with open(path_info_file_partition, 'r') as stream:
                self.data = yaml.load(stream)
        else:
            models = [d for d in os.listdir(os.path.join(self.root, 'images'))
                    if os.path.isdir(os.path.join(os.path.join(self.root, 'images'), d))]
            # remove CUROUD1 - CUROUD2. which only contains 2 images each.
            models.remove('CUROUD1')
            models.remove('CUROUD2')

            # Limit the dataset to the counterfeit images.
            models = ['CUEURB1','CUEURB2','CUEURC1','CUEURC2','CUEURD1','CUEURD2','IDESPC1','IDESPC2']
            # Only to debug, delete this line.
            #models = ['CUEURB1','CUEURB2']

            # Load the data DB
            self.data = {}
            total_size_mem = 0
            total_num_images = 0
            for model in models:
                path_info_file = self.root + '/images/' + model + '/'
                name_path_info_file = model + '_minres_400.pkl'
                path_complete = path_info_file + name_path_info_file

                ## open the banknote dataset info. Hack to open python2.7 pickle.
                with open(path_complete, 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    dataDB = u.load()
                
                self.data[model] = {}
                self.data[model] = dataDB['dataset_Images']

            # select the partition of the data
            for model in self.data.keys():
                for datasetType in ['train','test']:
                    self.data[model]['inputs'][datasetType] = self.data[model]['inputs'][datasetType][partition]
                    self.data[model]['labels'][datasetType] = self.data[model]['labels'][datasetType][partition]
                    
            # replace all the paths for the paths of this dataroot
            for model in self.data.keys():
                for datasetType in ['train','test']:
                    for i, path in enumerate(self.data[model]['inputs'][datasetType]):
                        self.data[model]['inputs'][datasetType][i] = os.path.join(os.path.join(self.root, 'images'),path[path.find(model):])

            # for each data model calculate the dpis of each image.
            print('Calculating size and dpi information')
            for model in self.data.keys(): 
                self.data[model]['sizes'] = {}
                for datasetType in ['train','test']:
                    # Create the dictionary entry of the dpi
                    self.data[model]['sizes'][datasetType] = []
                    dataModel = DataModel(model=model).getModelData()
                    heightmm = dataModel['Height']
                    widthmm = dataModel['Width']
                    print('Model: ' + model + ' partitionType: ' + datasetType)
                    for path in tqdm(self.data[model]['inputs'][datasetType]):
                        img = cv2.imread(path)
                        height, width, channels = img.shape
                        dpi_x = (width*24.4)/widthmm
                        dpi_y = (height*24.4)/heightmm
                        self.data[model]['sizes'][datasetType].append([height,width,dpi_x,dpi_y]) 
                

            # If there is not validation partition create it.

            # Train contains 70% and Test 30%.
            # Split train set, so the splits will be Train 60%, Validation 10%, Test 30%.
            print('Creating validation set....')
            random.seed(1447)
            for model in self.data.keys():
                nTrain = len(self.data[model]['inputs']['train'])
                nVal = int(math.ceil(nTrain*0.85))
                idxTrain = range(nTrain)
                validationIndx = np.array(idxTrain[nVal:])
                idxTrain = np.array(idxTrain[:nVal])
                # update lists with the validation set.
                self.data[model]['inputs']['val'] = \
                    (np.array(self.data[model]['inputs']['train'])[validationIndx]).tolist()
                self.data[model]['labels']['val'] = \
                    (np.array(self.data[model]['labels']['train'])[validationIndx]).tolist()
                self.data[model]['sizes']['val'] = \
                    (np.array(self.data[model]['sizes']['train'])[validationIndx]).tolist()
                # remove from training set the validation set.
                self.data[model]['labels']['train'] = \
                    (np.array(self.data[model]['labels']['train'])[idxTrain]).tolist()
                self.data[model]['inputs']['train']= \
                    (np.array(self.data[model]['inputs']['train'])[idxTrain]).tolist()
                self.data[model]['sizes']['train']= \
                    (np.array(self.data[model]['sizes']['train'])[idxTrain]).tolist()
                # update test to be arrays
                self.data[model]['labels']['test'] = \
                    (np.array(self.data[model]['labels']['test'])).tolist()
                self.data[model]['inputs']['test'] = \
                    (np.array(self.data[model]['inputs']['test'])).tolist()
                self.data[model]['sizes']['test'] = \
                    (np.array(self.data[model]['sizes']['test'])).tolist()
            

            # Load the reference images
            for model in models:
                reference = [os.path.join(os.path.join(self.root, 'reference/'+model), d) 
                                for d in os.listdir(os.path.join(self.root, 'reference/'+model))][0]
                self.data[model]['reference'] = reference

            # Load the ROI images
            for model in models:
                # get the number of rois
                nRois = len(DataModel(model=model).getRoiData())
                self.data[model]['rois'] = {}
                for datasetType in ['train','test','val']:
                    for i, path in enumerate(self.data[model]['inputs'][datasetType]):
                        filename_ext = os.path.basename(path)
                        extension = os.path.splitext(filename_ext)[1]
                        filename = os.path.splitext(filename_ext)[0]
                        self.data[model]['rois'][filename] = []
                        label = self.data[model]['labels'][datasetType][i]
                        for nroi in range(nRois):
                            path_roi = os.path.join(os.path.join(os.path.join(self.root,'rois'),model), \
                                                    filename + '_roi_' + str(nroi) + '_lbl_' + str(label) + extension)
                            self.data[model]['rois'][filename].append(path_roi)

            # Write YAML file
            print('Write banknote dataset...')
            with io.open(path_info_file_partition, 'w', encoding='utf8') as outfile:
                yaml.dump(self.data, outfile, default_flow_style=False, allow_unicode=True)

        # Get the list of models with counterfeit images.
        self.counterfeitModels = {}
        for model in self.data.keys():
            for datasetType in ['train','test','val']:
                num_negative_imgs = np.array(self.data[model]['labels'][datasetType]).sum()
                if num_negative_imgs > 0:
                    if not (model in self.counterfeitModels.keys()):
                        self.counterfeitModels[model] = {}
                    self.counterfeitModels[model][datasetType] = num_negative_imgs

        # Finally get only the datasetType which we are working with.
        for model in self.counterfeitModels:
            self.counterfeitModels[model] = self.counterfeitModels[model][self.train]
        for model in self.data:
            self.data[model]['labels'] = self.data[model]['labels'][self.train]
            self.data[model]['inputs'] = self.data[model]['inputs'][self.train]
            self.data[model]['sizes'] = self.data[model]['sizes'][self.train]

        self.encode_labels = {}
        self.decode_labels = {}
        for i, model in enumerate(self.data):
            self.encode_labels[model] = i+1
            self.decode_labels[i+1] = model


    def load_img(self, path, info_dpi, size = None):
        img = pil_image.open(path)
        img = img.convert('RGB')
        # Resize all the images to 600 dpis
        sizeX, sizeY, dpiX, dpiY = info_dpi
        sizeX = int((600.0/dpiX)*sizeX)
        sizeY = int((600.0/dpiY)*sizeY)
        if not(size is None):
            img = img.resize((size,size), pil_image.ANTIALIAS) # 2-tuple resize: (width, height)
        else:
            img = img.resize((sizeX,sizeY), pil_image.ANTIALIAS) 
        img = np.array(img, dtype='float32')
        return img

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class FullBanknote(BanknoteBase):

    def __init__(self, root, train='train',  size = None,
                     transform=None, target_transform=None):
        BanknoteBase.__init__(self, root, train, size, transform, target_transform)
        
    def __getitem__(self, index):

        model_cumsum = np.array([0]+[len(self.data[model]['labels']) for model in self.data.keys()]).cumsum()
        idx_model = [index<cumsum for i, cumsum in enumerate(model_cumsum[1:])].index(True)
        idx_image = index - model_cumsum[idx_model]
        path = self.data[list(self.data.keys())[idx_model]]['inputs'][idx_image]
        label = self.data[list(self.data.keys())[idx_model]]['labels'][idx_image]

        filename_ext = os.path.basename(path)
        filename = os.path.splitext(filename_ext)[0]
        paths_rois = self.data[list(self.data.keys())[idx_model]]['rois'][filename]

        img = pil_image.open(path)
        #img = np.array(img, dtype='float32')

        if self.transform is not None:
            img = self.transform(img)
        # squeeze in the 0 dim
        img = img[0]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        num_images = 0
        for model in self.data.keys():
            num_images += len(self.data[model]['labels'])
        return num_images




'''
This class adds the padding to the full banknote and then selects a random ROI
params:
        - nPairs: num of image Pairs of the dataset.
        - percentagePairs: 3 percentages that sum 1, default = [0.5, 0.25, 0.25].
        The first percentage is the % of num of genuine pairs. The second percentage is the
        % of genuine-counterfeit pairs. The third percentage is pairs of different classes.
        
'''
class FullBanknotePairs(BanknoteBase):

    def __init__(self, root, train='train', size = None, numTrials = 1500,
                     transform=None, target_transform=None):
        BanknoteBase.__init__(self, root, train, size,
                                    transform, target_transform)
        self.numTrials = numTrials
        self.percentagePairs = [0.5,0.25,0.25]

        # Count all images
        self.nImages = \
            np.sum([len(self.data[model]['inputs']) for model in self.data.keys()])

        self.numClasses = 2

        # Generate nPairs
        self.generate_pairs(self.numTrials)

    def generate_pairs(self, nPairs):
        # pair tuples selected
        self.data_idx = []

        # Select N/2 positive pairs
        modelsWithGenuine = list(self.data.keys())
        selected_model_genuine = np.random.choice(modelsWithGenuine, int(nPairs*self.percentagePairs[0]), replace=True)
        for model in selected_model_genuine:
            idx_positive_class_genuine = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==1]
            idx_selected = np.random.choice(idx_positive_class_genuine,2, replace=False)
            self.data_idx.append([(model,idx_selected[0]),(model,idx_selected[1])])

        # Select N/4 negative pairs same class. Counterfeits.
        modelsWithCounterfeit = list(self.counterfeitModels.keys())
        select_model_counterfeit = np.random.choice(modelsWithCounterfeit, int(nPairs*self.percentagePairs[1]), replace=True)
        for model in select_model_counterfeit:           
            idx_positive_class_genuine = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==1]
            idx_positive_class_counterfeit = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==0]
            self.data_idx.append([(model,np.random.choice(idx_positive_class_genuine,1)[0]),(model,np.random.choice(idx_positive_class_counterfeit,1)[0])])

        # Select N/4 negative pairs different class
        modelsWithGenuine = list(self.data.keys())
        for i in range(int(nPairs*self.percentagePairs[2])):
            models_selected = np.random.choice(modelsWithGenuine, 2, replace=False)
            idx_positive_class1_genuine = np.array(range(len(self.data[models_selected[0]]['labels'])))[np.array(self.data[models_selected[0]]['labels'])==1]
            idx_positive_class2_genuine = np.array(range(len(self.data[models_selected[1]]['labels'])))[np.array(self.data[models_selected[1]]['labels'])==1]
            self.data_idx.append([(models_selected[0], np.random.choice(idx_positive_class1_genuine,1)[0]), (models_selected[1], np.random.choice(idx_positive_class2_genuine,1)[0])])

        # Count all images
        self.nImages = len(self.data_idx)


    def __getitem__(self, index):

        #for index in range(self.nImages):
        #    print('index: %d.' % index)
        (model1,idx1),(model2,idx2) = self.data_idx[index]
        path1 = self.data[model1]['inputs'][idx1]
        path1 = path1[0] if type(path1) == np.ndarray else path1
        path2 = self.data[model2]['inputs'][idx2]
        path2 = path2[0] if type(path2) == np.ndarray else path2
        target1 = self.data[model1]['labels'][idx1]
        target1 = target1[0] if type(target1) == np.ndarray else target1
        target2 = self.data[model2]['labels'][idx2]
        target2 = target2[0] if type(target2) == np.ndarray else target2

        # If we have paths in self.data then load the image
        img1 = self.load_img(path=path1, info_dpi= self.data[model1]['sizes'][idx1] , size=self.size)
        img1 = pil_image.fromarray(np.uint8(img1))
        
        # If we have paths in self.data then load the image
        img2 = self.load_img(path=path2, info_dpi= self.data[model2]['sizes'][idx2] , size=self.size)
        img2 = pil_image.fromarray(np.uint8(img2))        

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            # Case the FCN is done inside the DataLoader
            if len(img1.shape)>3:
                img1 = img1[0]
                img2 = img2[0]

        '''
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
        '''

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        data = torch.stack((img1,img2))
        labels = (model1, target1, model2, target2)
        isSimilar = False
        if model1 == model2 and target1 == target2:
            isSimilar = True

        return data, int(isSimilar)

    def __len__(self):
        return self.nImages


class FullBanknoteOneShot(BanknoteBase):

    def __init__(self, root, train='train',
                 size=None, # normally size = 84
                 transform=None, target_transform=None,
                 sameClass = False, # If Same Class there will be only positive and negative examples of the same Class
                 n_way = 20,
                 n_shot = 1,
                 numTrials = 32):

        BanknoteBase.__init__(self, root, train, size,
                                    transform, target_transform)
        
        self.n_way = n_way
        self.n_shot = n_shot
        self.numTrials = numTrials
        self.sameClass = sameClass

    def __getitem__(self, index):

        # set the choice function to random
        np.random.seed(None)

        # Only classes with counterfeit samples can be selected for the positive class.
        classes_positive = list(self.counterfeitModels.keys())

        # Get n_shot positive examples. 
        positive_class = np.random.choice(classes_positive,1)[0]
        # Get indices of positive images from the positive class.
        idx_positive_class_genuine = np.array(range(len(self.data[positive_class]['labels'])))[np.array(self.data[positive_class]['labels'])==1]
        idx_positive_class_counterfeit = np.array(range(len(self.data[positive_class]['labels'])))[np.array(self.data[positive_class]['labels'])==0]

        # all classes can be negative, remove the positive class.
        classes_negative = list(self.data.keys())
        if positive_class in classes_negative:
            classes_negative.remove(positive_class)
        classes_negative = np.random.choice(classes_negative,self.n_way-1)

        # Num of samples
        indexes_perm = np.random.permutation((self.n_way) * self.n_shot)

        # list with the indices and tuples
        list_idxs = [[] for i in range((self.n_way*self.n_shot)+1)]

        # all n_way elements genuine and add n_shot counterfeits. 
        if self.sameClass:
            counter_added_elements = 0
            idxs = np.random.choice(idx_positive_class_genuine,((self.n_way-1)*self.n_shot)+1)
            for idx in idxs[:-1]:
                list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx]]
                counter_added_elements = counter_added_elements + 1                                                 
            
            # Add the reference positive class
            list_idxs[-1] = [positive_class,self.data[positive_class]['inputs'][idxs[-1]],self.data[positive_class]['labels'][idxs[-1]]]

            # Add the negative image
            idxs = np.random.choice(idx_positive_class_counterfeit,self.n_shot)
            for idx in idxs[:-1]:
                list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx]]
                counter_added_elements = counter_added_elements + 1                                                 

        # the elements are from the other classes + genuine and the probability=0.5 of having one counterfeit.
        else:
            probability_counterfeit = 0.5
            replace_negative_class_with_counterfeit = random.random() < probability_counterfeit
            # we replace some other class with the same positive class, but we will take it into account and only load
            # counterfeits.
            if replace_negative_class_with_counterfeit:
                classes_negative[np.random.choice(range(len(classes_negative)),1)[0]] = positive_class

            counter_added_elements = 0
            # Add first the negative classes
            for i in range(len(classes_negative)):
                # Find if it is a counterfeit from the same positive class
                if classes_negative[i] == positive_class:
                    idxs = np.random.choice(idx_positive_class_counterfeit,self.n_shot)
                    for idx in idxs:
                        list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx]]
                        counter_added_elements = counter_added_elements + 1
                else:
                    idxs = np.random.choice(range(len(self.data[classes_negative[i]]['inputs'])),self.n_shot)
                    for idx in idxs:
                        list_idxs[indexes_perm[counter_added_elements]] = [classes_negative[i],
                                                                            self.data[classes_negative[i]]['inputs'][idx],
                                                                            self.data[classes_negative[i]]['labels'][idx]]
                        counter_added_elements = counter_added_elements + 1

            # Add now the positive class 
            idxs = np.random.choice(idx_positive_class_genuine,self.n_shot+1)
            for idx in idxs[:-1]:
                list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx]]
                counter_added_elements = counter_added_elements + 1

            # Add positive class to compare with
            list_idxs[-1] = [positive_class,self.data[positive_class]['inputs'][idxs[-1]],self.data[positive_class]['labels'][idxs[-1]]]


        # Iterate over the selected samples and load the images
        data = []
        labels_model = []
        labels_genuine_counterfeit = []
        for elem in list_idxs:
            if type(elem[1]).__name__ == 'str':
                img1 = self.load_img(path=elem[1],size=self.size)
                img1 = pil_image.fromarray(np.uint8(img1))
            else:
                img1 = elem[1]
            img1 = self.transform(img1)
            # Case the FCN is done inside the DataLoader
            if len(img1.shape)>3:
                img1 = img1[0]
            data.append(img1)
            # Encode labels
            if elem[2] == 1:
                labels_model.append(self.encode_labels[elem[0]])
            else:
                labels_model.append(-1*self.encode_labels[elem[0]])
            labels_genuine_counterfeit.append(elem[2])

        data = torch.stack(data)
        labels = torch.from_numpy(np.array(labels_model))
        
        '''
        import cv2
        [cv2.imwrite('/home/aberenguel/tmp/cedar/im_' + str(i) + '.png',
                     trial[i].transpose(0, 1).transpose(1, 2).cpu().numpy() * 255) for i in range(trial.shape[0])]
        '''

        return data, labels

    def __len__(self):
        # num characters * num samples of each character
        return self.numTrials

