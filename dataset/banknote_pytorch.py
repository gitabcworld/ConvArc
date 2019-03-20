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
import pdb
import h5py
import yaml
import time
import shutil

class BanknoteBase(data.Dataset):

    def __init__(self, setType='set1', root='./banknote', train='train', size = None, 
                mode = 'generator', path_tmp_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tmp_data/'),
                transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set
        self.size = size
        self.setType = setType
        partition = 0

        # create the structure for generator / processor 
        self.mode = mode
        self.path_temp = path_tmp_data
        self.path_temp = os.path.join(self.path_temp,train)
        if not os.path.exists(self.path_temp):
            os.makedirs(self.path_temp)
        self.path_temp_epoch = os.path.join(self.path_temp,'epoch_0')
        ##

        path_info_file_partition = os.path.join(self.root,'tmp/data_minres_400_partition_'+ str(partition) + '_' + setType + '.yaml')
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
            if self.setType == 'set1':
                models = ['CUEURB1','CUEURB2','CUEURC1','CUEURC2','CUEURD1','CUEURD2','IDESPC1','IDESPC2']
            else:
                models.remove('CUEURB1')
                models.remove('CUEURB2')
                models.remove('CUEURC1')
                models.remove('CUEURC2')
                models.remove('CUEURD1')
                models.remove('CUEURD2')
                models.remove('IDESPC1')
                models.remove('IDESPC2')

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
                num_negative_imgs = (np.array(self.data[model]['labels'][datasetType])==0).astype(np.int).sum()
                if num_negative_imgs > 0:
                    if not (model in self.counterfeitModels.keys()):
                        self.counterfeitModels[model] = {}
                    self.counterfeitModels[model][datasetType] = num_negative_imgs

        # Finally get only the datasetType which we are working with.
        counterfeitModels_tmp = {}
        data_tmp = {}
        
        # UNCOMMENT TO SEE PARTITON STATISTICS
        #for key in self.data.keys():
        #    total = np.array([len(self.data[key]['labels'][datasetType]) for datasetType in ['train','val','test']]).sum()
        #    for datasetType in ['train','val','test']:
        #        print('[%s] Total imgs: %d. %s: %d. Counterfeit: %f' % (key,total, datasetType, len(self.data[key]['labels'][datasetType]),
        #                (np.array(self.data[key]['labels'][datasetType])==0).astype(np.int).sum()/len(self.data[key]['labels'][datasetType])))

        for model in self.data.keys():
            data_tmp[model] = {}
            data_tmp[model]['labels'] = []
            data_tmp[model]['inputs'] = []
            data_tmp[model]['sizes'] = []
            #if model in self.counterfeitModels.keys():
            #    counterfeitModels_tmp[model] = 0
            
            for datasetType in ['train','test','val']:
                if not(self.train is None):
                    if datasetType in self.train:
                        data_tmp[model]['labels'].append(self.data[model]['labels'][datasetType])
                        data_tmp[model]['inputs'].append(self.data[model]['inputs'][datasetType])
                        data_tmp[model]['sizes'].append(self.data[model]['sizes'][datasetType])
                        
                        #if model in self.counterfeitModels.keys():
                        #    counterfeitModels_tmp[model] += self.counterfeitModels[model][datasetType]

            # flatten the lists if needed
            self.data[model]['labels'] = [item for sublist in data_tmp[model]['labels'] for item in sublist]
            self.data[model]['inputs'] = [item for sublist in data_tmp[model]['inputs'] for item in sublist]
            self.data[model]['sizes'] = [item for sublist in data_tmp[model]['sizes'] for item in sublist]

        counterfeitModels_tmp = {}
        data_tmp = {}

        '''             
        for model in self.counterfeitModels:
            self.counterfeitModels[model] = self.counterfeitModels[model][self.train]
        for model in self.data:
            self.data[model]['labels'] = self.data[model]['labels'][self.train]
            self.data[model]['inputs'] = self.data[model]['inputs'][self.train]
            self.data[model]['sizes'] = self.data[model]['sizes'][self.train]
        '''

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

    def makeDeterministicTransforms(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def resetDeterministicTransforms(self):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        np.random.seed(None)
        random.seed(None)

    def getFolderEpochList(self):
        d = self.path_temp
        return [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

    def set_path_tmp_epoch_iteration(self, epoch, iteration):
        self.path_temp_epoch = os.path.join(self.path_temp,'epoch_'+str(epoch))
        self.path_temp_epoch = os.path.join(self.path_temp_epoch,'iteration_'+str(iteration))
        if self.mode == 'generator':
            if not os.path.exists(self.path_temp_epoch):
                os.makedirs(self.path_temp_epoch)

    def remove_path_tmp_epoch(self, epoch, iteration=None):
        self.path_temp_epoch = os.path.join(self.path_temp,'epoch_'+str(epoch))
        if not(iteration is None):
            self.path_temp_epoch = os.path.join(self.path_temp_epoch,'iteration_'+str(iteration))
        if os.path.isdir(self.path_temp_epoch):
	        shutil.rmtree(self.path_temp_epoch)


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class FullBanknote(BanknoteBase):

    def __init__(self, setType, root, train='train',  size = None,
                    mode = 'generator', path_tmp_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tmp_data/'),
                    transform=None, target_transform=None):
        BanknoteBase.__init__(self, setType, root, train, size, 
                                mode, path_tmp_data,
                                transform, target_transform)
    
    def generate_data(self, index):

        model_cumsum = np.array([0]+[len(self.data[model]['labels']) for model in self.data.keys()]).cumsum()
        idx_model = [index<cumsum for i, cumsum in enumerate(model_cumsum[1:])].index(True)
        idx_image = index - model_cumsum[idx_model]
        path = self.data[list(self.data.keys())[idx_model]]['inputs'][idx_image]
        label = self.data[list(self.data.keys())[idx_model]]['labels'][idx_image]

        filename_ext = os.path.basename(path)
        filename = os.path.splitext(filename_ext)[0]
        paths_rois = self.data[list(self.data.keys())[idx_model]]['rois'][filename]

        img = pil_image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __getitem__(self, index):

        # load images and create a hdf5 package file and a txt file for syncronization.
        path_sync = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_synchro.yaml')
        path_data = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_data.hdf5')
        path_labels = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_labels.yaml')

        if self.mode == 'generator_processor':

            img, label = self.generate_data(index)

        if self.mode == 'generator':

            img, label = self.generate_data(index)

            # Save the data information
            hf = h5py.File(path_data,'w')
            hf.create_dataset('batch_' + str(index), data=img.numpy())
            hf.close()
            # Save the label information
            with open(path_labels, 'w') as outfile:
                labels_dict = {}
                labels_dict['label'] = label
                yaml.dump(labels_dict, outfile, default_flow_style=False)
            # save the control synchronization file
            with open(path_sync, 'w') as outfile:
                noop_dict = {}
                noop_dict['info'] = 'ready'
                yaml.dump(noop_dict, outfile, default_flow_style=False)

        if self.mode == 'processor':

            while not os.path.exists(path_sync):
                time.sleep(0.5)

            if os.path.isfile(path_data):
                hf = h5py.File(path_data, 'r')
                img = hf.get('batch_' + str(index))
                img = torch.from_numpy(np.array(img))
                hf.close()
            else:
                raise ValueError("%s isn't a file!" % path_data)
            
            if os.path.isfile(path_labels):
                y = yaml.load(open(path_labels))
                label = y['label']
            else:
                raise ValueError("%s isn't a file!" % path_labels)

            # Remove the files
            os.remove(path_sync)
            os.remove(path_data)
            os.remove(path_labels)

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

    def __init__(self, setType, root, train='train', size = None, numTrials = 1500,
                        mode = 'generator', path_tmp_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tmp_data/'),
                        transform=None, target_transform=None):
        BanknoteBase.__init__(self, setType, root, train, size,
                                    mode, path_tmp_data,
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

        nPositivePairs = int(nPairs*self.percentagePairs[0])
        nNegativePairsSameClass = int(nPairs*self.percentagePairs[1])
        nNegativePairsDifferentClass = int(nPairs*self.percentagePairs[2])
        nDiff = nPairs - (nPositivePairs+nNegativePairsSameClass+nNegativePairsDifferentClass)
        if nDiff > 0:
            groupToAdd = random.randint(0, 3)
            if groupToAdd == 0:
                nPositivePairs = nPositivePairs + 1
            if groupToAdd == 1:
                nNegativePairsSameClass = nNegativePairsSameClass + 1
            if groupToAdd == 2:
                nNegativePairsDifferentClass = nNegativePairsDifferentClass + 1

        # Select N/2 positive pairs
        modelsWithGenuine = list(self.data.keys())
        selected_model_genuine = np.random.choice(modelsWithGenuine, nPositivePairs, replace=True)
        for model in selected_model_genuine:
            idx_positive_class_genuine = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==1]
            idx_selected = np.random.choice(idx_positive_class_genuine,2, replace=False)
            self.data_idx.append([(model,idx_selected[0]),(model,idx_selected[1])])

        # Select N/4 negative pairs same class. Counterfeits.
        modelsWithCounterfeit = list(self.counterfeitModels.keys())
        select_model_counterfeit = np.random.choice(modelsWithCounterfeit, nNegativePairsSameClass, replace=True)
        for model in select_model_counterfeit:           
            idx_positive_class_genuine = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==1]
            idx_positive_class_counterfeit = np.array(range(len(self.data[model]['labels'])))[np.array(self.data[model]['labels'])==0]
            self.data_idx.append([(model,np.random.choice(idx_positive_class_genuine,1)[0]),(model,np.random.choice(idx_positive_class_counterfeit,1)[0])])

        # Select N/4 negative pairs different class
        modelsWithGenuine = list(self.data.keys())
        for i in range(nNegativePairsDifferentClass):
            models_selected = np.random.choice(modelsWithGenuine, 2, replace=False)
            idx_positive_class1_genuine = np.array(range(len(self.data[models_selected[0]]['labels'])))[np.array(self.data[models_selected[0]]['labels'])==1]
            idx_positive_class2_genuine = np.array(range(len(self.data[models_selected[1]]['labels'])))[np.array(self.data[models_selected[1]]['labels'])==1]
            self.data_idx.append([(models_selected[0], np.random.choice(idx_positive_class1_genuine,1)[0]), (models_selected[1], np.random.choice(idx_positive_class2_genuine,1)[0])])

        # Count all images
        self.nImages = len(self.data_idx)

    def generate_data(self, index):

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
            # Make the same transformation
            rand_number = int(np.random.uniform()*1000)
            self.makeDeterministicTransforms(seed=rand_number)
            img1 = self.transform(img1)
            self.makeDeterministicTransforms(seed=rand_number)
            img2 = self.transform(img2)
            self.resetDeterministicTransforms()
            # Case the FCN is done inside the DataLoader
            if len(img1.shape)>3:
                img1 = img1[0]
                img2 = img2[0]

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        ret_data = torch.stack((img1,img2))
        labels = (model1, target1, model2, target2)
        return ret_data, labels

    def __getitem__(self, index):

        # load images and create a hdf5 package file and a txt file for syncronization.
        path_sync = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_synchro.yaml')
        path_data = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_data.hdf5')
        path_labels = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_labels.yaml')

        if self.mode == 'generator_processor':

            ret_data, labels = self.generate_data(index)
            model1, target1, model2, target2 = labels

        if self.mode == 'generator':

            ret_data, labels = self.generate_data(index)
            model1, target1, model2, target2 = labels

            # Save the data information
            try:
                hf = h5py.File(path_data,'w')
                hf.create_dataset('batch_' + str(index), data=ret_data.numpy())
                hf.close()
                    
                # Save the label information
                with open(path_labels, 'w') as outfile:
                    labels_dict = {}
                    labels_dict['model1'] = model1
                    labels_dict['target1'] = target1
                    labels_dict['model2'] = model2
                    labels_dict['target2'] = target2
                    yaml.dump(labels_dict, outfile, default_flow_style=False)
                # save the control synchronization file
                with open(path_sync, 'w') as outfile:
                    noop_dict = {}
                    noop_dict['info'] = 'ready'
                    yaml.dump(noop_dict, outfile, default_flow_style=False)
            except Exception as e: 
                print(e) # Do nothing and continue                

        if self.mode == 'processor':

            while not os.path.exists(path_sync):
                time.sleep(0.5)

            if os.path.isfile(path_data):
                hf = h5py.File(path_data, 'r')
                data = hf.get('batch_' + str(index))
                ret_data = torch.from_numpy(np.array(data))
                hf.close()
            else:
                raise ValueError("%s isn't a file!" % path_data)
            
            if os.path.isfile(path_labels):
                y = yaml.load(open(path_labels))
                labels = (y['model1'], y['target1'], y['model2'], y['target2'])
                model1, target1, model2, target2 = labels
            else:
                raise ValueError("%s isn't a file!" % path_labels)

            # Remove the files
            os.remove(path_sync)
            os.remove(path_data)
            os.remove(path_labels)


        isSimilar = False
        if model1 == model2 and target1 == target2:
            isSimilar = True

        return ret_data, int(isSimilar)

    def __len__(self):
        return self.nImages


class FullBanknoteTriplets(BanknoteBase):

    def __init__(self, setType, root, train='train', size = None, numTrials = 1500, 
                     mode = 'generator', path_tmp_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tmp_data/'),
                     transform=None, target_transform=None):
        BanknoteBase.__init__(self, setType, root, train, size, 
                                    mode, path_tmp_data,
                                    transform, target_transform)
        self.numTrials = numTrials

    def generate_triplet(self):
        if np.random.rand(1) > 0.5: # Generate a triplet that the negative is counterfeit
            modelsWithCounterfeit = list(self.counterfeitModels.keys())
            model_counterfeit = np.random.choice(modelsWithCounterfeit, 1)[0]
            idx_positive_class_genuine = np.array(range(len(self.data[model_counterfeit]['labels'])))[np.array(self.data[model_counterfeit]['labels'])==1]
            idx_positive_class_counterfeit = np.array(range(len(self.data[model_counterfeit]['labels'])))[np.array(self.data[model_counterfeit]['labels'])==0]
            data_idx = [(model_counterfeit,np.random.choice(idx_positive_class_genuine,1)[0]), # positive
                        (model_counterfeit,np.random.choice(idx_positive_class_genuine,1)[0]), # anchor
                        (model_counterfeit,np.random.choice(idx_positive_class_counterfeit,1)[0])] # negative
        else:   # Generate a triplet where the negative comes from a different class.
            modelsWithGenuine = list(self.data.keys())
            models_selected = np.random.choice(modelsWithGenuine, 2, replace=False)
            idx_class1_genuine = np.array(range(len(self.data[models_selected[0]]['labels'])))[np.array(self.data[models_selected[0]]['labels'])==1]
            idx_class2_genuine = np.array(range(len(self.data[models_selected[1]]['labels'])))[np.array(self.data[models_selected[1]]['labels'])==1]
            data_idx = [(models_selected[0], np.random.choice(idx_class1_genuine,1)[0]), # positive
                        (models_selected[0], np.random.choice(idx_class1_genuine,1)[0]), # anchor
                        (models_selected[1], np.random.choice(idx_class2_genuine,1)[0])] # negative
        return data_idx

    def generate_data(self):

        (model1,idx1),(model2,idx2),(model3,idx3) = self.generate_triplet()
        path1 = self.data[model1]['inputs'][idx1]
        path1 = path1[0] if type(path1) == np.ndarray else path1
        path2 = self.data[model2]['inputs'][idx2]
        path2 = path2[0] if type(path2) == np.ndarray else path2
        path3 = self.data[model3]['inputs'][idx3]
        path3 = path3[0] if type(path3) == np.ndarray else path3
        target1 = self.data[model1]['labels'][idx1]
        target1 = target1[0] if type(target1) == np.ndarray else target1
        target2 = self.data[model2]['labels'][idx2]
        target2 = target2[0] if type(target2) == np.ndarray else target2
        target3 = self.data[model3]['labels'][idx3]
        target3 = target3[0] if type(target3) == np.ndarray else target3

        # Positive img. 
        img1 = self.load_img(path=path1, info_dpi= self.data[model1]['sizes'][idx1] , size=self.size)
        img1 = pil_image.fromarray(np.uint8(img1))
        
        # Anchor img.
        img2 = self.load_img(path=path2, info_dpi= self.data[model2]['sizes'][idx2] , size=self.size)
        img2 = pil_image.fromarray(np.uint8(img2))        

        # Negative img.
        img3 = self.load_img(path=path3, info_dpi= self.data[model3]['sizes'][idx3] , size=self.size)
        img3 = pil_image.fromarray(np.uint8(img3))        

        if self.transform is not None:
            # Make the same transformation
            rand_number = int(np.random.uniform()*1000)
            self.makeDeterministicTransforms(seed=rand_number)
            img1 = self.transform(img1)
            self.makeDeterministicTransforms(seed=rand_number)
            img2 = self.transform(img2)
            self.makeDeterministicTransforms(seed=rand_number)
            img3 = self.transform(img3)
            self.resetDeterministicTransforms()
            # Case the FCN is done inside the DataLoader
            if len(img1.shape)>3:
                img1 = img1[0]
                img2 = img2[0]
                img3 = img3[3]

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)
            target3 = self.target_transform(target3)

        ret_data = torch.stack((img1,img2,img3))
        labels = (model1, target1, model2, target2, model3, target3) 
        return ret_data, labels


    def __getitem__(self, index):

        # load images and create a hdf5 package file and a txt file for syncronization.
        path_sync = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_synchro.yaml')
        path_data = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_data.hdf5')
        path_labels = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_labels.yaml')

        if self.mode == 'generator_processor':

            ret_data, labels = self.generate_data()
            model1, target1, model2, target2, model3, target3 = labels

        if self.mode == 'generator':

            ret_data, labels = self.generate_data()
            model1, target1, model2, target2, model3, target3 = labels

            # Save the data information
            hf = h5py.File(path_data,'w')
            hf.create_dataset('batch_' + str(index), data=ret_data.numpy())
            hf.close()
            # Save the label information
            with open(path_labels, 'w') as outfile:
                labels_dict = {}
                labels_dict['model1'] = model1
                labels_dict['target1'] = target1
                labels_dict['model2'] = model2
                labels_dict['target2'] = target2
                labels_dict['model3'] = model2
                labels_dict['target3'] = target3
                yaml.dump(labels_dict, outfile, default_flow_style=False)
            # save the control synchronization file
            with open(path_sync, 'w') as outfile:
                noop_dict = {}
                noop_dict['info'] = 'ready'
                yaml.dump(noop_dict, outfile, default_flow_style=False)
        
        if self.mode == 'processor':
            while not os.path.exists(path_sync):
                time.sleep(0.5)

            if os.path.isfile(path_data):
                hf = h5py.File(path_data, 'r')
                data = hf.get('batch_' + str(index))
                ret_data = torch.from_numpy(np.array(data))
                hf.close()
            else:
                raise ValueError("%s isn't a file!" % path_data)
            
            if os.path.isfile(path_labels):
                y = yaml.load(open(path_labels))
                labels = (y['model1'], y['target1'], y['model2'], y['target2'], y['model3'], y['target3'])
            else:
                raise ValueError("%s isn't a file!" % path_labels)

            # Remove the files
            os.remove(path_sync)
            os.remove(path_data)
            os.remove(path_labels)

        return ret_data, labels

    def __len__(self):
        return self.numTrials



class FullBanknoteOneShot(BanknoteBase):

    def __init__(self, setType, root, train='train',
                 size=None, # normally size = 84
                 transform=None, target_transform=None,
                 mode = 'generator', path_tmp_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../tmp_data/'),
                 sameClass = False, # If Same Class there will be only positive and negative examples of the same Class
                 n_way = 20,
                 n_shot = 1,
                 numTrials = 32):

        BanknoteBase.__init__(self, setType, root, train, size,
                                    mode, path_tmp_data,
                                    transform, target_transform)
        
        self.n_way = n_way
        self.n_shot = n_shot
        self.numTrials = numTrials
        self.sameClass = sameClass

    def generate_data(self, index):

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
                                                                            self.data[positive_class]['labels'][idx],
                                                                            self.data[positive_class]['sizes'][idx]]
                counter_added_elements = counter_added_elements + 1                                                 
            
            # Add the reference positive class
            list_idxs[-1] = [positive_class,
                                self.data[positive_class]['inputs'][idxs[-1]],
                                self.data[positive_class]['labels'][idxs[-1]],
                                self.data[positive_class]['sizes'][idxs[-1]]]

            # Add the negative image
            idxs = np.random.choice(idx_positive_class_counterfeit,self.n_shot)
            for idx in idxs[:-1]:
                list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx],
                                                                            self.data[positive_class]['sizes'][idx]]
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
                                                                            self.data[positive_class]['labels'][idx],
                                                                            self.data[positive_class]['sizes'][idx]]
                        counter_added_elements = counter_added_elements + 1
                else:
                    idxs = np.random.choice(range(len(self.data[classes_negative[i]]['inputs'])),self.n_shot)
                    for idx in idxs:
                        list_idxs[indexes_perm[counter_added_elements]] = [classes_negative[i],
                                                                            self.data[classes_negative[i]]['inputs'][idx],
                                                                            self.data[classes_negative[i]]['labels'][idx],
                                                                            self.data[classes_negative[i]]['sizes'][idx]]
                        counter_added_elements = counter_added_elements + 1

            # Add now the positive class 
            idxs = np.random.choice(idx_positive_class_genuine,self.n_shot+1)
            for idx in idxs[:-1]:
                list_idxs[indexes_perm[counter_added_elements]] = [positive_class,
                                                                            self.data[positive_class]['inputs'][idx],
                                                                            self.data[positive_class]['labels'][idx],
                                                                            self.data[positive_class]['sizes'][idx]]
                counter_added_elements = counter_added_elements + 1

            # Add positive class to compare with
            list_idxs[-1] = [positive_class,
                                    self.data[positive_class]['inputs'][idxs[-1]],
                                    self.data[positive_class]['labels'][idxs[-1]],
                                    self.data[positive_class]['sizes'][idxs[-1]]]


        # Iterate over the selected samples and load the images
        data = []
        labels_model = []
        labels_genuine_counterfeit = []
        # for the positive class we want the same cropping and transformation. 
        # save first a random seed.
        rand_seed = int(np.random.uniform()*1000)
            
        for elem in list_idxs:
            # load image
            if type(elem[1]).__name__ == 'str':
                img1 = self.load_img(path=elem[1], info_dpi= elem[3], size=self.size)
                img1 = pil_image.fromarray(np.uint8(img1))
            else:
                img1 = elem[1]
            
            # if is the positive class crop the same zone and apply the same transforms
            if elem[0] == positive_class:
                self.makeDeterministicTransforms(seed=rand_seed)
            # apply transforms
            img1 = self.transform(img1)
            if elem[0] == positive_class:
                self.resetDeterministicTransforms()

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
        return data, labels


    def __getitem__(self, index):

        # load images and create a hdf5 package file and a txt file for syncronization.
        path_sync = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_synchro.yaml')
        path_data = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_data.hdf5')
        path_labels = os.path.join(self.path_temp_epoch,'batch_'+str(index)+'_labels.yaml')

        if self.mode == 'generator_processor':
            data, labels = self.generate_data(index)

        if self.mode == 'generator':

            data, labels = self.generate_data(index)

            # Save the data information
            hf = h5py.File(path_data,'w')
            hf.create_dataset('batch_' + str(index), data=data.numpy())
            hf.close()
            # Save the label information
            hf = h5py.File(path_labels,'w')
            hf.create_dataset('batch_' + str(index), data=labels.numpy())
            hf.close()
            # save the control synchronization file
            with open(path_sync, 'w') as outfile:
                noop_dict = {}
                noop_dict['info'] = 'ready'
                yaml.dump(noop_dict, outfile, default_flow_style=False)

        if self.mode == 'processor':

            while not os.path.exists(path_sync):
                time.sleep(0.5)

            if os.path.isfile(path_data):
                hf = h5py.File(path_data, 'r')
                data = hf.get('batch_' + str(index))
                data = torch.from_numpy(np.array(data))
                hf.close()
            else:
                raise ValueError("%s isn't a file!" % path_data)
            
            if os.path.isfile(path_labels):
                hf = h5py.File(path_labels, 'r')
                labels = hf.get('batch_' + str(index))
                labels = torch.from_numpy(np.array(labels))
                hf.close()
            else:
                raise ValueError("%s isn't a file!" % path_labels)

            # Remove the files
            os.remove(path_sync)
            os.remove(path_data)
            os.remove(path_labels)

        '''
        import cv2
        [cv2.imwrite('/home/aberenguel/tmp/cedar/im_' + str(i) + '.png',
                     trial[i].transpose(0, 1).transpose(1, 2).cpu().numpy() * 255) for i in range(trial.shape[0])]
        '''

        return data, labels

    def __len__(self):
        # num characters * num samples of each character
        return self.numTrials

