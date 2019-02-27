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
from dataset.omniglot import OmniglotPairs
from dataset.omniglot import OmniglotOneShot

# Mini-imagenet dataset
from miniimagenetDataLoader import miniImagenetDataLoader
from dataset.mini_imagenet import MiniImagenet
from dataset.mini_imagenet import MiniImagenetPairs
from dataset.mini_imagenet import MiniImagenetOneShot

# Banknote dataset
from banknoteDataLoader import banknoteDataLoader
from dataset.banknote_pytorch import FullBanknote
from dataset.banknote_pytorch import FullBanknotePairs
from dataset.banknote_pytorch import FullBanknoteOneShot

# FCN
from models.conv_cnn import ConvCNNFactory
# Attention module in ARC
from models.fullContext import FullContextARC
from models.naiveARC import NaiveARC
# Co-Attn module
from models.coAttn import CoAttn

from do_epoch_fns import do_epoch_ARC, do_epoch_ARC_unroll, do_epoch_naive_full
import arc_train
import arc_val
import arc_test
import context_train
import context_val
import context_test

import multiprocessing
import ntpath
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

# CUDA_VISIBLE_DEVICES == 1 (710) / CUDA_VISIBLE_DEVICES == 0 (1070)
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def data_generation(opt):

    # use cuda?
    opt.cuda = torch.cuda.is_available()
    cudnn.benchmark = True # set True to speedup

    # Load mean/std if exists
    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(opt.save, 'mean.npy')):
        train_mean = np.load(os.path.join(opt.save, 'mean.npy'))
        train_std = np.load(os.path.join(opt.save, 'std.npy'))

    # Load Dataset
    opt.setType='set1'
    dataLoader = banknoteDataLoader(type=FullBanknotePairs, opt=opt, fcn=None, train_mean=train_mean,
                                    train_std=train_std)

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
    
    epoch = 0
    if opt.arc_resume == True or opt.arc_load is None:    
        try:
            while epoch < opt.train_num_batches:

                # wait to check if it is neede more data
                lst_epochs = train_loader.dataset.getFolderEpochList()
                if len(lst_epochs) > 50:
                    time.sleep(10)
                
                # In case there is more than one generator.
                # get the last folder epoch executed and update the epoch accordingly
                if len(lst_epochs)>0:
                    epoch = np.array([path_leaf(str).split('_')[-1] for str in lst_epochs if 'train' in str]).astype(np.int).max()
                
                epoch += 1

                ## set information of the epoch in the dataloader
                repetitions = 1
                start_time = datetime.now()
                for repetition in range(repetitions):
                    train_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, label) in enumerate(train_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[train]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

                if epoch % opt.val_freq == 0:

                    repetitions = opt.val_num_batches
                    start_time = datetime.now()
                    for repetition in range(repetitions):
                        val_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                        for batch_idx, (data, label) in enumerate(val_loader):
                            noop = 0
                    time_elapsed = datetime.now() - start_time
                    print ("[val]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

                    repetitions = opt.test_num_batches
                    start_time = datetime.now()
                    for repetition in range(repetitions):
                        test_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                        for batch_idx, (data, label) in enumerate(test_loader):
                            noop = 0
                    time_elapsed = datetime.now() - start_time
                    print ("[test]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
                
                epoch += 1

            print ("[%s] ... generating data done" % multiprocessing.current_process().name)

        except KeyboardInterrupt:
            pass

    print ('###########################################')
    print ('.... Starting Context Data Generation ....')
    print ('###########################################')

    # Set the new batchSize as in the ARC code.
    opt.__dict__['batchSize'] = opt.naive_batchSize
    # Change the path_tmp_data to point to one_shot
    opt.path_tmp_data = opt.path_tmp_data.replace('/data/','/data_one_shot/')

    # Load the dataset
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenetOneShot, opt=opt, fcn=None)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=OmniglotOneShot, opt=opt, fcn=None, train_mean=train_mean,
                                      train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknoteOneShot, opt=opt, fcn=None, 
                                            train_mean=train_mean, train_std=train_std)
    else:
        pass

    # Get the DataLoaders from train - val - test
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed)

    epoch = 0
    try:
        while epoch < opt.naive_full_epochs:

            # wait to check if it is neede more data
            lst_epochs = train_loader.dataset.getFolderEpochList()
            if len(lst_epochs) > 50:
                time.sleep(10)
            
            # In case there is more than one generator.
            # get the last folder epoch executed and update the epoch accordingly
            if len(lst_epochs)>0:
                epoch = np.array([path_leaf(str).split('_')[-1] for str in lst_epochs if 'train' in str]).astype(np.int).max()
            
            epoch += 1

            ## set information of the epoch in the dataloader
            repetitions = 1
            start_time = datetime.now()
            for repetition in range(repetitions):
                train_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                for batch_idx, (data, label) in enumerate(train_loader):
                    noop = 0
            time_elapsed = datetime.now() - start_time
            print ("[train]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

            if epoch % opt.naive_full_val_freq == 0:

                repetitions = opt.val_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    val_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, label) in enumerate(val_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[val]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

                repetitions = opt.test_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    test_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, label) in enumerate(test_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[test]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
            
            epoch += 1

        print ("[%s] ... generating data done" % multiprocessing.current_process().name)

    except KeyboardInterrupt:
        pass

    ###################################



def server_processing(opt):

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
    fcn = None
    if opt.apply_wrn:
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
        dataLoader = miniImagenetDataLoader(type=MiniImagenetPairs, opt=opt, fcn=fcn)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=OmniglotPairs, opt=opt, fcn=fcn,train_mean=train_mean,
                                        train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknotePairs, opt=opt, fcn=fcn, train_mean=train_mean,
                                        train_std=train_std)
    else:
        pass

    # Get the params
    # opt = dataLoader.opt
    
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

    # Load the Co-Attn module
    coAttn = None
    if opt.use_coAttn:
        coAttn = CoAttn(size = opt.coAttn_size, num_filters=opt.arc_nchannels, typeActivation = opt.coAttn_type, p = opt.coAttn_p)
        if opt.coattn_load is not None and os.path.exists(opt.coattn_load):
            if torch.cuda.is_available():
                coAttn.load_state_dict(torch.load(opt.coattn_load))
            else:
                coAttn.load_state_dict(torch.load(opt.coattn_load, map_location=torch.device('cpu')))
        if opt.cuda:
            coAttn.cuda()

    loss_fn = torch.nn.BCELoss()
    if opt.cuda:
        loss_fn = loss_fn.cuda()

    lstOptimizationParameters = []
    lstOptimizationParameters.append(list(discriminator.parameters()))
    if opt.apply_wrn:
        lstOptimizationParameters.append(list(fcn.parameters()))
    if opt.use_coAttn:
        lstOptimizationParameters.append(list(coAttn.parameters()))
    
    flatten_lstOptimizationParameters = [item for sublist in lstOptimizationParameters for item in sublist]
    optimizer = torch.optim.Adam(params=flatten_lstOptimizationParameters, lr=opt.arc_lr)
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

    print ('[%s] ... Testing' % multiprocessing.current_process().name)
    # for the final testing set the data loader to generator
    test_loader.dataset.mode = 'generator_processor'
    test_loader.dataset.remove_path_tmp_epoch(epoch)
    test_acc_epoch = arc_test.arc_test(epoch, do_epoch_fn, opt, test_loader, discriminator, logger)
    #'''

    ###########################################
    ## Now Train the NAIVE of FULL CONTEXT model
    ###########################################
    print ('###########################################')
    print ('... Starting Context Classification')
    print ('###########################################')

    # Set the new batchSize as in the ARC code.
    opt.__dict__['batchSize'] = opt.naive_batchSize

    # Add the model_fn Naive / Full Context classification
    context_fn = None
    if opt.naive_full_type == 'Naive':
        context_fn = NaiveARC(numStates = opt.arc_numStates)
    elif opt.naive_full_type == 'FullContext':
        layer_sizes = opt.naive_full_layer_sizes
        vector_dim = opt.arc_numStates
        num_layers = opt.naive_full_num_layers
        context_fn = FullContextARC(hidden_size=layer_sizes, num_layers=num_layers, vector_dim=vector_dim)

    # Load the Fcn
    fcn = None
    if opt.apply_wrn:
        # Convert the opt params to dict.
        optDict = dict([(key, value) for key, value in opt._get_kwargs()])
        fcn = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
        if torch.cuda.is_available():
            fcn.load_state_dict(torch.load(opt.wrn_load))
        else:
            fcn.load_state_dict(torch.load(opt.wrn_load, map_location=torch.device('cpu')))
        if opt.cuda:
            fcn.cuda()

    # Load the discriminator
    if opt.arc_load is not None and os.path.exists(opt.arc_load):
        if torch.cuda.is_available():
            discriminator.load_state_dict(torch.load(opt.arc_load))
        else:
            discriminator.load_state_dict(torch.load(opt.arc_load, map_location=torch.device('cpu')))
    if opt.cuda and discriminator is not None:
        discriminator = discriminator.cuda()

    # Load the Co-Attn module
    coAttn = None
    if opt.use_coAttn:
        coAttn = CoAttn(size = opt.coAttn_size, num_filters=opt.arc_nchannels, typeActivation = opt.coAttn_type, p = opt.coAttn_p)
        if opt.coattn_load is not None and os.path.exists(opt.coattn_load):
            if torch.cuda.is_available():
                coAttn.load_state_dict(torch.load(opt.coattn_load))
            else:
                coAttn.load_state_dict(torch.load(opt.coattn_load, map_location=torch.device('cpu')))
        if opt.cuda:
            coAttn.cuda()

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
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenetOneShot, opt=opt, fcn=fcn)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=OmniglotOneShot, opt=opt, fcn=fcn, train_mean=train_mean,
                                      train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknoteOneShot, opt=opt, fcn=fcn, 
                                            train_mean=train_mean, train_std=train_std)
    else:
        pass

    # Get the params
    opt = dataLoader.opt

    # Use the same seed to split the train - val - test
    if os.path.exists(os.path.join(opt.save, 'dataloader_rnd_seed_naive_full.npy')):
        rnd_seed = np.load(os.path.join(opt.save, 'dataloader_rnd_seed_naive_full.npy'))
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
                                                                                logger, optimizer, loss_fn, fcn, coAttn)

                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)

                if epoch % opt.naive_full_val_freq == 0:

                    val_acc_epoch, val_loss_epoch, is_model_saved = context_val.context_val(epoch, do_epoch_naive_full,
                                                                                            opt, val_loader,
                                                                                            discriminator, context_fn,
                                                                                            logger, loss_fn, fcn, coAttn)
                    if is_model_saved:
                        # Save the optimizer
                        torch.save(optimizer.state_dict(), opt.naive_full_optimizer_path)
                        # Test the model
                        test_acc_epoch = context_test.context_test(epoch, do_epoch_naive_full, opt, test_loader,
                                                                    discriminator, context_fn, logger, fcn, coAttn)
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
    test_loader.dataset.mode = 'generator_processor'
    test_acc_epoch = context_test.context_test(epoch, do_epoch_fn, opt, test_loader, discriminator, context_fn, logger, fcn=fcn, coAttn=coAttn)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)



def train(index = None):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    if opt.mode == 'generator':
        print('Starting generator...')
        data_generation(opt)
    elif opt.mode == 'generator_processor':
        print('Starting generator - processor no save images...')
        server_processing(opt)  
    else:
        print('Starting processor...')
        server_processing(opt)  

def main():
    train()

if __name__ == "__main__":
    main()
