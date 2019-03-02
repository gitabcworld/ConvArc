import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from logger import Logger
from sklearn.metrics import accuracy_score
import shutil
import multiprocessing
import cv2
from tqdm import tqdm

from option import Options, tranform_options

# Omniglot dataset
from omniglotDataLoader import omniglotDataLoader
from dataset.omniglot import Omniglot

# Mini-imagenet dataset
from miniimagenetDataLoader import miniImagenetDataLoader
from dataset.mini_imagenet import MiniImagenet

# Banknote dataset
from banknoteDataLoader import banknoteDataLoader
from dataset.banknote_pytorch import FullBanknote

# Features
from features.HoGFeature import HoGFeature
from features.UCIFeature import UCIFeature
from features.SIFTFeature import SIFTFeature
from features.LBPFeature import LBPFeature
from features.HaralickFeature import HaralickFeature
from features.BhavaniFeature import BahavaniFeature

# statistics
from util.show_results import show_results

from classifiers.xgboost import XGBoostClassifier

def train(index = 14):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    opt.cuda = False

    # set the mode of the dataset to generator_processor
    # which generates and processes the images without saving them.
    opt.mode = 'generator_processor'

    # Load Dataset
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=None)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot, opt=opt, fcn=None,train_mean=None,
                                        train_std=None)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknote, opt=opt, fcn=None, train_mean=None,
                                        train_std=None)
    else:
        pass

    # Use the same seed to split the train - val - test
    if os.path.exists(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy')):
        rnd_seed = np.load(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy'))
    else:    
        rnd_seed = np.random.randint(0, 100000)
        np.save(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy'), rnd_seed)

    # Get the DataLoaders from train - val - test
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed,dataPartition = ['train+val',None,'test'])
    
    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        #opt.name = "train_{}_{}_{}_{}_{}_wrn".format(str_model_fn, opt.numGlimpses, opt.glimpseSize, opt.numStates,
        opt.name = "{}_{}_{}_{}_{}_{}_wrn".format(opt.naive_full_type,
                                        "fcn" if opt.apply_wrn else "no_fcn",
                                        opt.arc_numGlimpses,
                                        opt.arc_glimpseSize, opt.arc_numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("[{}]. Will start training {} with parameters:\n{}\n\n".format(multiprocessing.current_process().name,
                                                                         opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join(opt.save, opt.name)
    if not os.path.isdir(models_path):
	    os.makedirs(models_path)
    else:
        shutil.rmtree(models_path)

    # create logger
    logger = Logger(models_path)

    # create object features
    #nameFeatures = 'HoGFeature'
    #nameFeatures = 'UCIFeature'
    nameFeatures = opt.wrn_name_type
    objFeatures = eval(nameFeatures + '()')

    objClassifier = XGBoostClassifier()

    train_features = []
    train_labels = []
    
    # if the training features exists go to testing
    if not(os.path.exists(os.path.join(opt.save,'train_features.npy')) and os.path.exists(os.path.join(opt.save,'train_labels.npy'))):

        ## EXTRACT FEATURES TRAIN
        for i in range(opt.train_num_batches):
            for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
                # transform batch of data and label tensors to numpy
                data = data.numpy().transpose(0,2,3,1)
                labels = labels.numpy().tolist()
                for i in range(len(data)):
                    features = objFeatures.extract(data[i])
                    train_features.append(features)
                train_labels.append(labels)
        
        # save the features
        train_features = np.stack(train_features)
        train_labels = [item for sublist in train_labels for item in sublist]
        np.save(os.path.join(opt.save,'train_features.npy'),train_features)
        np.save(os.path.join(opt.save,'train_labels.npy'),train_labels)
    else:
        train_features = np.load(os.path.join(opt.save,'train_features.npy'))
        train_labels = np.load(os.path.join(opt.save,'train_labels.npy'))

    ## TRAIN
    objClassifier.train(train_features,train_labels)
    objClassifier.save(opt.save)

    ## EXTRACT FEATURES TEST
    for j in range(opt.test_num_batches):
        test_features = []
        test_labels = []
        for batch_idx, (data, labels) in enumerate(tqdm(test_loader)):
            # transform batch of data and label tensors to numpy
            data = data.numpy().transpose(0,2,3,1)
            labels = labels.numpy().tolist()
            for i in range(len(data)):
                features = objFeatures.extract(data[i])
                test_features.append(features)
            test_labels.append(labels)

        ## PREDICT
        test_features = np.stack(test_features)
        test_labels = [item for sublist in test_labels for item in sublist]
        preds = objClassifier.predict(test_features)
        test_features_set1 = test_features
        test_labels_set1 = test_labels
        preds_set1 = preds

        show_results(test_labels,preds,'TEST SET 1. Iter: ' + str(j),show=False)

    ## Get the set2 and try
    opt.setType='set2'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=None)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot, opt=opt, fcn=None,train_mean=None,
                                        train_std=None)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknote, opt=opt, fcn=None, train_mean=None,
                                        train_std=None)
    else:
        pass
    
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed, dataPartition = [None,None,'train+val+test'])
    ## EXTRACT FEATURES TEST
    for j in range(opt.test_num_batches):
        test_features = []
        test_labels = []
        for batch_idx, (data, labels) in enumerate(tqdm(test_loader)):
            # transform batch of data and label tensors to numpy
            data = data.numpy().transpose(0,2,3,1)
            labels = labels.numpy().tolist()
            for i in range(len(data)):
                features = objFeatures.extract(data[i])
                test_features.append(features)
            test_labels.append(labels)

        ## PREDICT
        test_features = np.stack(test_features)
        test_labels = [item for sublist in test_labels for item in sublist]
        preds = objClassifier.predict(test_features)
        
        test_features_set2 = test_features
        test_labels_set2 = test_labels
        preds_set2 = preds
        
        show_results(test_labels,preds,'TEST SET 2. Iter: ' + str(j),show=False)

    #''' UNCOMMENT!!!! TESTING NAIVE - FULLCONTEXT
    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ('[%s] ... Testing' % multiprocessing.current_process().name)
    #TODO: TEST!!!
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)
    #'''
    
    


def main():
    train()

if __name__ == "__main__":
    main()
