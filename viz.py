import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from logger import Logger
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import shutil
from models import models
from models.models import ArcBinaryClassifier

from omniglotBenchmarks import omniglotBenchMark

from option import Options, tranform_options
from dataset.omniglot_pytorch import OmniglotOSPairs
from dataset.omniglot_pytorch import OmniglotOneShot
from models.conv_cnn import ConvCNNFactory
from do_epoch_fns import do_epoch_ARC, do_epoch_ARC_unroll
import arc_train
import arc_val
import arc_test

opt = parser.parse_args()

def display(image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.numpy()/255, cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.numpy()/255, cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.savefig(os.path.join(images_path, name))


def get_sample(discriminator):

    # size of the set to choose sample from from
    sample_size = 30
    X, Y = batcher.fetch_batch("train", batch_size=sample_size)
    pred = discriminator(X)

    if opt.same:
        same_pred = pred[sample_size // 2:].data.numpy()[:, 0]
        mx = same_pred.argsort()[len(same_pred) // 2]  # choose the sample with median confidence
        index = mx + sample_size // 2
    else:
        diff_pred = pred[:sample_size // 2].data.numpy()[:, 0]
        mx = diff_pred.argsort()[len(diff_pred) // 2]  # choose the sample with median confidence
        index = mx

    return X[index]


def visualize():

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))

    arc = discriminator.arc

    sample = get_sample(discriminator)

    all_hidden = arc._forward(sample[None, :, :])[:, 0, :]  # (2*numGlimpses, controller_out)
    glimpse_params = torch.tanh(arc.glimpser(all_hidden))
    _, channels, height, witdth = sample.shape
    masks = arc.glimpse_window.get_attention_mask(glimpse_params, mask_h=height, mask_w=witdth)

    # separate the masks of each image.
    masks1 = []
    masks2 = []
    for i, mask in enumerate(masks):
        if i % 2 == 1:  # the first image outputs the hidden state for the next image
            masks1.append(mask)
        else:
            masks2.append(mask)

    channels = 3
    for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
        sample_0 = (sample[0].transpose(0, 1).transpose(1, 2) + \
                    Variable(torch.from_numpy(batcher.mean_pixel)).float()) * 255
        sample_1 = (sample[1].transpose(0, 1).transpose(1, 2) + \
                    Variable(torch.from_numpy(batcher.mean_pixel)).float()) * 255

        display(sample_0, mask1, sample_1, mask2,"img_{}".format(i))


if __name__ == "__main__":
    visualize()



def train(index = 0):

    # change parameters
    options = Options().parse()
    options = tranform_options(index, options)
    cudnn.benchmark = True # set True to speedup

    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(options.save, 'mean.npy')):
        train_mean = np.load(os.path.join(options.save, 'mean.npy'))
        train_std = np.load(os.path.join(options.save, 'std.npy'))
    bnktBenchmark = omniglotBenchMark(type=OmniglotOSPairs, opt=options, train_mean=train_mean,
                                      train_std=train_std)
    opt = bnktBenchmark.opt
    train_loader, val_loader, test_loader = bnktBenchmark.get()

    train_mean = bnktBenchmark.train_mean
    train_std = bnktBenchmark.train_std
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

    # make directory for storing images.
    images_path = os.path.join(opt.name, "visualization")
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

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
    if opt.arc_load is not None:
        discriminator.load_state_dict(torch.load(opt.arc_load))

    if opt.cuda:
        discriminator.cuda()

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
    logger = None
    optimizer = None
    loss_fn = None
    val_acc_epoch, val_loss_epoch, is_model_saved = arc_val.arc_val(epoch, do_epoch_fn, opt, val_loader,
                                                                    discriminator, logger,
                                                                    convCNN=convCNN,
                                                                    optimizer=optimizer,
                                                                    loss_fn=loss_fn, fcn=fcn)

