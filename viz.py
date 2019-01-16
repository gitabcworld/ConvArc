import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil

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

def display(opt, image1, mask1, image2, mask2, name="no_name.png"):
    _, ax = plt.subplots(1, 2)

    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.arc_glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1, cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2, cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.savefig(os.path.join('D:/PhD/logs/images/', name))


def visualize(index = 0):

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
        discriminator.load_state_dict(torch.load(opt.arc_load))

    if opt.cuda:
        discriminator.cuda()

    # Set for the first batch a random seed for AumentationAleju
    train_loader.dataset.agumentation_seed = int(np.random.rand() * 1000)

    for batch_idx, (data, label) in enumerate(train_loader):

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        inputs = Variable(data, requires_grad=False)
        targets = Variable(label)

        batch_size, npair, nchannels, x_size, y_size = inputs.shape
        inputs = inputs.view(batch_size * npair, nchannels, x_size, y_size)
        if fcn:
            inputs = fcn(inputs)
        _ , nfilters, featx_size, featy_size = inputs.shape
        inputs = inputs.view(batch_size, npair, nfilters, featx_size, featy_size)

        #features, updated_states = discriminator(inputs)
        all_hidden = discriminator.arc._forward(inputs)
        glimpse_params = torch.tanh(discriminator.arc.glimpser(all_hidden)) # return [num_glimpses*2,batchsize,(x, y, delta)]
        
        sample = data[0]
        _, channels, height, witdth = sample.shape

        # separate the masks of each image.
        masks1 = []
        masks2 = []
        for i in range(glimpse_params.shape[0]):
            mask = discriminator.arc.glimpse_window.get_attention_mask(glimpse_params[i], mask_h=height, mask_w=witdth)
            if i % 2 == 1:  # the first image outputs the hidden state for the next image
                masks1.append(mask)
            else:
                masks2.append(mask)

        channels = 3
        for glimpse_i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
            for batch_i in range(data.shape[0]):
                sample_0 = ((data[batch_i,0].data.cpu().numpy() * train_std + train_mean)*255.0).transpose(1,2,0).astype(np.uint8)
                sample_1 = ((data[batch_i,1].data.cpu().numpy() * train_std + train_mean)*255.0).transpose(1,2,0).astype(np.uint8)
                if sample_0.shape[2] == 1:
                    sample_0 = np.repeat(sample_0,3,axis=2)
                    sample_1 = np.repeat(sample_1,3,axis=2)
                display(opt, sample_0, mask1[batch_i], sample_1, mask2[batch_i],"img_batch_%d_glimpse_%d.png" % (batch_i,glimpse_i))


if __name__ == "__main__":
    visualize()
