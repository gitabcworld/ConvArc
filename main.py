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
from dataset.omniglot import Omniglot_Pairs
from dataset.omniglot import OmniglotOneShot

# Mini-imagenet dataset
from miniimagenetDataLoader import miniImagenetDataLoader
from dataset.mini_imagenet import MiniImagenet
from dataset.mini_imagenet import MiniImagenetPairs
from dataset.mini_imagenet import MiniImagenetOneShot

# Banknote dataset
from dataset.banknote_pytorch import FullBanknoteROI

from models.conv_cnn import ConvCNNFactory
from models.fullContext import FullContextARC
from models.naiveARC import NaiveARC

from do_epoch_fns import do_epoch_ARC, do_epoch_ARC_unroll, do_epoch_naive_full
import arc_train
import arc_val
import arc_test
import context_train
import context_val
import context_test

import multiprocessing

import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

# CUDA_VISIBLE_DEVICES == 1 (710) / CUDA_VISIBLE_DEVICES == 0 (1070)
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(index = 2):

    # change parameters
    options = Options().parse()
    #options = Options().parse() if options is None else options
    options = tranform_options(index, options)
    # use cuda?
    options.cuda = torch.cuda.is_available()

    cudnn.benchmark = True # set True to speedup

    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(options.save, 'mean.npy')):
        train_mean = np.load(os.path.join(options.save, 'mean.npy'))
        train_std = np.load(os.path.join(options.save, 'std.npy'))
    
    if options.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenetPairs, opt=options)
    elif options.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot_Pairs, opt=options, train_mean=train_mean,
                                        train_std=train_std)
    else:
        pass

    # Get the params
    opt = dataLoader.opt
    
    # Use the same seed to split the train - val - test
    if os.path.exists(os.path.join(options.save, 'dataloader_rnd_seed_arc.npy')):
        rnd_seed = np.load(os.path.join(options.save, 'dataloader_rnd_seed_arc.npy'))
    else:    
        rnd_seed = np.random.randint(0, 100000)
        np.save(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy'), rnd_seed)

    # Get the DataLoaders from train - val - test
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed)

    train_mean = dataLoader.train_mean
    train_std = dataLoader.train_std
    if not os.path.exists(os.path.join(options.save, 'mean.npy')):
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

    fcn = None
    convCNN = None
    if opt.apply_wrn:
        # Convert the opt params to dict.
        optDict = dict([(key, value) for key, value in opt._get_kwargs()])
        convCNN = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
        if opt.wrn_load:
            # Load the model in fully convolutional mode
            fcn, params, stats = convCNN.load(opt.wrn_load, fully_convolutional = True)
        else:
            fcn = convCNN.create(fully_convolutional = True)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.arc_numGlimpses,
                                        glimpse_h=opt.arc_glimpseSize,
                                        glimpse_w=opt.arc_glimpseSize,
                                        channels=opt.arc_nchannels,
                                        controller_out=opt.arc_numStates,
                                        attn_type = opt.arc_attn_type,
                                        attn_unroll = opt.arc_attn_unroll,
                                        attn_dense=opt.arc_attn_dense)

    # load from a previous checkpoint, if specified.
    if opt.arc_load is not None and os.path.exists(opt.arc_load):
        if torch.cuda.is_available():
            discriminator.load_state_dict(torch.load(opt.arc_load))
        else:
            discriminator.load_state_dict(torch.load(opt.arc_load, map_location=torch.device('cpu')))
        

    if opt.cuda:
        discriminator.cuda()

    loss_fn = torch.nn.BCELoss()
    if opt.cuda:
        loss_fn = loss_fn.cuda()

    if opt.apply_wrn:
        optimizer = torch.optim.Adam(params=list(discriminator.parameters()) + list(fcn.params.values()), lr=opt.arc_lr)
    else:
        optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=opt.arc_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=opt.arc_lr_patience, verbose=True)

    # load preexisting optimizer values if exists
    if os.path.exists(opt.arc_optimizer_path):
        if torch.cuda.is_available():
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path))
        else:
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path, map_location=torch.device('cpu')))

    # Select the epoch functions
    do_epoch_fn = None
    if opt.arc_attn_unroll == True:
        do_epoch_fn = do_epoch_ARC_unroll
    else:
        do_epoch_fn = do_epoch_ARC

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
                                                                        loss_fn=loss_fn, fcn=fcn)
                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)
                if epoch % opt.val_freq == 0:
                    val_acc_epoch, val_loss_epoch, is_model_saved = arc_val.arc_val(epoch, do_epoch_fn, opt, val_loader,
                                                                                    discriminator, logger,
                                                                                    convCNN=convCNN,
                                                                                    optimizer=optimizer,
                                                                                    loss_fn=loss_fn, fcn=fcn)
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
    print ('[%s] ... Testing' % multiprocessing.current_process().name)
    test_acc_epoch = arc_test.arc_test(epoch, do_epoch_fn, opt, test_loader, discriminator, logger)
    #'''

    ###########################################
    ## Now Train the NAIVE of FULL CONTEXT model
    ###########################################
    print ('###########################################')
    print ('... Starting Context Classification')
    print ('###########################################')

    # Set the new batchSize as in the ARC code.
    options.__dict__['batchSize'] = opt.naive_batchSize

    # Add the model_fn Naive / Full Context classification
    context_fn = None
    if opt.naive_full_type == 'Naive':
        context_fn = NaiveARC(numStates = opt.arc_numStates)
    elif opt.naive_full_type == 'FullContext':
        layer_sizes = opt.naive_full_layer_sizes
        vector_dim = opt.arc_numStates
        num_layers = opt.naive_full_num_layers
        context_fn = FullContextARC(hidden_size=layer_sizes, num_layers=num_layers, vector_dim=vector_dim)

    # Load the discriminator
    if opt.arc_load is not None and os.path.exists(opt.arc_load):
        if torch.cuda.is_available():
            discriminator.load_state_dict(torch.load(opt.arc_load))
        else:
            discriminator.load_state_dict(torch.load(opt.arc_load, map_location=torch.device('cpu')))
    if opt.cuda and discriminator is not None:
        discriminator = discriminator.cuda()

    # Load the Naive / Full classifier
    if opt.naive_full_load_path is not None and os.path.exists(opt.naive_full_load_path):
        if torch.cuda.is_available():
            context_fn.load_state_dict(torch.load(opt.naive_full_load_path))
        else:
            context_fn.load_state_dict(torch.load(opt.naive_full_load_path, map_location=torch.device('cpu')))
    if opt.cuda and context_fn is not None:
        context_fn = context_fn.cuda()

    # Set the epoch function
    do_epoch_fn = do_epoch_naive_full

    # Load the dataset
    if options.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenetOneShot, opt=options)
    elif options.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=OmniglotOneShot, opt=options, train_mean=train_mean,
                                      train_std=train_std)
    else:
        pass

    # Get the params
    opt = dataLoader.opt

    # Use the same seed to split the train - val - test
    if os.path.exists(os.path.join(options.save, 'dataloader_rnd_seed_naive_full.npy')):
        rnd_seed = np.load(os.path.join(options.save, 'dataloader_rnd_seed_naive_full.npy'))
    else:    
        rnd_seed = np.random.randint(0, 100000)
        np.save(os.path.join(opt.save, 'dataloader_rnd_seed_naive_full.npy'), rnd_seed)

    # Get the DataLoaders from train - val - test
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed)

    # Loss
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.BCELoss()
    if opt.cuda:
        loss_fn = loss_fn.cuda()

    optimizer = torch.optim.Adam(params=context_fn.parameters(), lr=opt.naive_full_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=opt.arc_lr_patience, verbose=True,
                                  cooldown=opt.arc_lr_patience)

    # load preexisting optimizer values if exists
    if os.path.exists(opt.naive_full_optimizer_path):
        if torch.cuda.is_available():
            optimizer.load_state_dict(torch.load(opt.naive_full_optimizer_path))
        else:
            optimizer.load_state_dict(torch.load(opt.naive_full_optimizer_path, map_location=torch.device('cpu')))

    ###################################
    ## TRAINING NAIVE/FULLCONTEXT
    if opt.naive_full_resume == True or opt.naive_full_load_path is None:

        try:
            epoch = 0
            while epoch < opt.naive_full_epochs:
                epoch += 1
                train_acc_epoch, train_loss_epoch = context_train.context_train(epoch, do_epoch_naive_full, opt,
                                                                                train_loader, discriminator, context_fn,
                                                                                logger, optimizer, loss_fn, fcn)

                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)

                if epoch % opt.naive_full_val_freq == 0:

                    val_acc_epoch, val_loss_epoch, is_model_saved = context_val.context_val(epoch, do_epoch_naive_full,
                                                                                            opt, val_loader,
                                                                                            discriminator, context_fn,
                                                                                            logger, loss_fn, fcn)
                    if is_model_saved:
                        # Save the optimizer
                        torch.save(optimizer.state_dict(), opt.naive_full_optimizer_path)
                        # Test the model
                        test_acc_epoch = context_test.context_test(epoch, do_epoch_naive_full, opt, test_loader,
                                                                    discriminator, context_fn, logger)
                logger.step()

            print ("[%s] ... training done" % multiprocessing.current_process().name)
            print ("[%s] best validation accuracy: %.2f, best validation loss: %.5f" % (
                multiprocessing.current_process().name, context_val.best_accuracy, context_val.best_validation_loss))
            print ("[%s] ... exiting training regime" % multiprocessing.current_process().name)

        except KeyboardInterrupt:
            pass
    ###################################

    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ('[%s] ... Testing' % multiprocessing.current_process().name)
    test_acc_epoch = context_test.context_test(epoch, do_epoch_fn, opt, test_loader, discriminator, context_fn, logger)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)

def main():
    train()

if __name__ == "__main__":
    main()
