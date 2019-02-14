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

# statistics
from util.show_results import show_results

def train(index = 4):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    opt.cuda = False

    # Load Dataset
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
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed)

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
    nameFeatures = 'HoGFeature'
    objFeatures = eval(nameFeatures + '()')

    train_features = []
    train_labels = []
    
    ## EXTRACT FEATURES TRAIN
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        # transform batch of data and label tensors to numpy
        data = data.numpy().transpose(0,2,3,1)
        labels = labels.numpy().tolist()
        for i in range(len(data)):
            features = objFeatures.extract(data[i])
            train_features.append(features)
        train_labels.append(labels)
        if batch_idx > 10:
            break

    ## TRAIN
    train_features = np.stack(train_features)
    train_labels = [item for sublist in train_labels for item in sublist]
    objFeatures.train(train_features,train_labels)

    ## EXTRACT FEATURES TEST
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
        if batch_idx > 10:
            break

    ## PREDICT
    test_features = np.stack(test_features)
    test_labels = [item for sublist in test_labels for item in sublist]
    preds = objFeatures.predict(test_features,test_labels)

    show_results(test_labels,preds,'HoG')

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
