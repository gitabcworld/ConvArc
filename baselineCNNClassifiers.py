import os
import sys

# Add all the python paths needed to execute when using Python 3.6
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models/arc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "skiprnn_pytorch"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models/wrn"))

import time
import numpy as np
from datetime import datetime, timedelta
from logger import Logger
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import shutil

from models import models
from models.models import ArcBinaryClassifier

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

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

# FCN
from models.conv_cnn import ConvCNNFactory

from do_epoch_fns import do_epoch_classification
import arc_train
import arc_val
import arc_test

import multiprocessing

import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

# CUDA_VISIBLE_DEVICES == 1 (710) / CUDA_VISIBLE_DEVICES == 0 (1070)
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(index = None):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    # use cuda?
    opt.cuda = torch.cuda.is_available()

    cudnn.benchmark = True # set True to speedup

    # Load mean/std if exists
    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(opt.save, 'mean.npy')):
        train_mean = np.load(os.path.join(opt.save, 'mean.npy'))
        train_std = np.load(os.path.join(opt.save, 'std.npy'))
    
    # Load FCN
    # Convert the opt params to dict.
    optDict = dict([(key, value) for key, value in opt._get_kwargs()])
    fcn = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
    if opt.wrn_load and os.path.exists(opt.wrn_load):
        if torch.cuda.is_available():
            fcn.load_state_dict(torch.load(opt.wrn_load))
        else:
            fcn.load_state_dict(torch.load(opt.wrn_load, map_location=torch.device('cpu')))
    if opt.cuda:
        fcn.cuda()

    # Load Dataset
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=fcn)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot, opt=opt, fcn=fcn,train_mean=train_mean,
                                        train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknote, opt=opt, fcn=fcn, train_mean=train_mean,
                                        train_std=train_std)
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

    train_mean = dataLoader.train_mean
    train_std = dataLoader.train_std
    if not os.path.exists(os.path.join(opt.save, 'mean.npy')):
        np.save(os.path.join(opt.save, 'mean.npy'), train_mean)
        np.save(os.path.join(opt.save, 'std.npy'), train_std)

    if opt.cuda:
        models.use_cuda = True

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

    loss_fn = torch.nn.CrossEntropyLoss()
    if opt.cuda:
        loss_fn = loss_fn.cuda()

    optimizer = torch.optim.Adam(params=fcn.parameters(), lr=opt.arc_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=opt.arc_lr_patience, verbose=True)

    # load preexisting optimizer values if exists
    if os.path.exists(opt.arc_optimizer_path):
        if torch.cuda.is_available():
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path))
        else:
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path, map_location=torch.device('cpu')))

    # Select the epoch functions
    do_epoch_fn = do_epoch_classification
    discriminator = None
    coAttn = None

    ###################################
    ## TRAINING ARC/CONVARC
    ###################################
    epoch = 0
    if opt.arc_resume == True or opt.arc_load is None:

        try:
            while epoch < opt.train_num_batches:
                epoch += 1

                train_acc_epoch, train_loss_epoch = arc_train.arc_train(epoch, do_epoch_fn, opt, train_loader,
                                                                        discriminator, logger, optimizer=optimizer,
                                                                        loss_fn=loss_fn, fcn=fcn, coAttn=coAttn)
                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)
                if epoch % opt.val_freq == 0:
                    val_acc_epoch, val_loss_epoch, is_model_saved = arc_val.arc_val(epoch, do_epoch_fn, opt, val_loader,
                                                                                    discriminator, logger,
                                                                                    optimizer=optimizer,
                                                                                    loss_fn=loss_fn, fcn=fcn, coAttn=coAttn)
                    if is_model_saved:
                        test_acc_epoch = arc_test.arc_test(epoch, do_epoch_fn, opt, test_loader, discriminator, logger)

                logger.step()

            print ("[%s] ... training done" % multiprocessing.current_process().name)
            print ("[%s], best validation accuracy: %.2f, best validation loss: %.5f" % (
                multiprocessing.current_process().name, arc_val.best_accuracy, arc_val.best_validation_loss))
            print ("[%s] ... exiting training regime " % multiprocessing.current_process().name)

        except KeyboardInterrupt:
            pass
    ###################################

    #''' UNCOMMENT!!!! TESTING NAIVE - FULLCONTEXT
    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ('[%s] ... Testing Set1' % multiprocessing.current_process().name)
    test_acc_epoch = arc_test.arc_test(epoch, do_epoch_fn, opt, test_loader, discriminator, logger)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)
    #'''

    ## Get the set2 and try
    print ('[%s] ... Loading Set2' % multiprocessing.current_process().name)
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
    print ('[%s] ... Testing Set2' % multiprocessing.current_process().name)
    test_acc_epoch = arc_test.arc_test(epoch, do_epoch_fn, opt, test_loader, discriminator, logger)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)

def main():
    train()

if __name__ == "__main__":
    main()
