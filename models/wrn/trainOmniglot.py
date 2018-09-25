"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    2017 Albert Berenguel
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchnet.engine import Engine
from tqdm import tqdm

from dataset.omniglot_pytorch import OmniglotOS
from resnet import resnet
from utils import cast, data_parallel
from util import cvtransforms as T
import torchvision.transforms as transforms
from optionOmniglot import Options

from sklearn.metrics import accuracy_score
import time
import random

cudnn.benchmark = True


def main():

    opt = Options().parse()
    epoch_step = json.loads(opt.epoch_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    kwargs = {'num_workers': opt.nthread, 'pin_memory': True} if opt.cuda else {}
    cv2_scale = lambda x: cv2.resize(x, dsize=(opt.imageSize, opt.imageSize),
                                     interpolation=cv2.INTER_AREA).astype(np.uint8)
    np_reshape = lambda x: np.reshape(x, (opt.imageSize, opt.imageSize, opt.nchannels))
    np_repeat = lambda x: np.repeat(x, 3, axis=2)

    #################################
    # NORMALIZATION: Calculate the mean and std of training.
    #################################
    train_transform = tnt.transform.compose([
        cv2_scale,
        np_reshape,
        transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(
        OmniglotOS(root = opt.dataroot,
                     train='train', transform=train_transform, target_transform=None),
        batch_size=opt.batchSize, shuffle=True, **kwargs)

    pbar = tqdm(enumerate(train_loader))
    tmp = []
    for batch_idx, (data, labels) in pbar:
        tmp.append(data)
        pbar.set_description(
            '[{}/{} ({:.0f}%)]\t'.format(
                batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))
    omn_mean = torch.cat(tmp).mean()
    omn_std = torch.cat(tmp).std()
    # Free cuda memory
    tmp = []
    data = []
    labels = []

    #################################
    # TRANSFORMATIONS: transformations for the TRAIN dataset
    #################################
    train_transform = tnt.transform.compose([
        cv2_scale,
        np_reshape,
        np_repeat,
        T.AugmentationAleju(channel_is_first_axis = False,
                            hflip = opt.hflip, vflip = opt.vflip,
                            rotation_deg = opt.rotation_deg, shear_deg = opt.shear_deg,
                            translation_x_px = opt.translation_px,
                            translation_y_px = opt.translation_px),
        T.Normalize([omn_mean, omn_mean, omn_mean], [omn_std, omn_std, omn_std]),
        transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        OmniglotOS(root=opt.dataroot,
                   train='train', transform=train_transform, target_transform=None),
        batch_size=opt.batchSize, shuffle=True, **kwargs)

    #################################
    # TRANSFORMATIONS: transformations for the EVAL and TEST dataset
    #################################
    eval_test_transform = tnt.transform.compose([
        cv2_scale,
        np_reshape,
        np_repeat,
        T.Normalize([omn_mean, omn_mean, omn_mean], [omn_std, omn_std, omn_std]),
        transforms.ToTensor(),
    ])
    val_loader = torch.utils.data.DataLoader(
        OmniglotOS(root=opt.dataroot,
                   train='val', transform=eval_test_transform, target_transform=None),
        batch_size=opt.batchSize, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        OmniglotOS(root=opt.dataroot,
                   train='test', transform=eval_test_transform, target_transform=None),
        batch_size=opt.batchSize, shuffle=False, **kwargs)

    num_classes = train_loader.dataset.getNumClasses()
    f, params, stats = resnet(opt.depth, opt.width, num_classes, False)

    def create_optimizer(opt, lr):
        print 'creating optimizer with lr = ', lr
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(params.values(), lr, 0.9, weight_decay=opt.weightDecay)
        elif opt.optim_method == 'Adam':
            return torch.optim.Adam(params.values(), lr)

    def log(t, optimizer, params, stats, opt):
        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
                        stats=stats,
                        optimizer=optimizer.state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'w'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print z


    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in params.iteritems():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print '\nParameters:'
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data)
    print '\nAdditional buffers:'
    kmax = max(len(key) for key in stats.keys())
    for i, (key, v) in enumerate(stats.items()):
        print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v)

    n_parameters = sum(p.numel() for p in params.values() + stats.values())
    print '\nTotal number of parameters:', n_parameters

    # Save folder
    best_val_acc = 0
    if opt.save == '':
        opt.save = './logs/resnet_' + str(random.getrandbits(128))[:-20]
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    ######################################
    # TRAIN
    ######################################
    for epoch in range(opt.epochs):
        train_all_acc = []
        train_all_losses = []
        tick = time.clock()
        for batch_idx, (data, label) in enumerate(train_loader):
            if opt.cuda:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            model_training = True
            train_preds = data_parallel(f, inputs, params, stats, model_training, np.arange(opt.ngpu))
            training_loss = F.cross_entropy(train_preds, targets)
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            train_probs, train_classes = torch.max(train_preds, 1)
            train_acc = accuracy_score(targets.data.cpu().numpy(),train_classes.data.cpu().numpy())

            train_all_acc.append(train_acc)
            train_all_losses.append(training_loss.data.cpu().numpy()[0])
            # Free memory
            inputs = []
            targets = []

        # Adjust learning rate
        if epoch in epoch_step:
            lr = optimizer.param_groups[0]['lr']
            optimizer = create_optimizer(opt, lr * opt.lr_decay_ratio)

        # Validation
        if epoch % opt.eval_freq == 0:
            all_preds = []
            all_targets = []
            for batch_idx, (data, label) in enumerate(val_loader):
                if opt.cuda:
                    data = data.cuda()
                inputs = Variable(data)
                model_training = False
                y = data_parallel(f, inputs, params, stats, model_training, np.arange(opt.ngpu))
                all_preds.append(y.cpu().data.numpy())
                all_targets.append(label.numpy())

            all_preds = np.vstack(all_preds).argmax(1)
            all_targets = np.hstack(all_targets)
            val_acc = accuracy_score(all_targets,all_preds)
            print ("++++++++++++++++++++++++")
            print ("epoch: %d, val acc: %.2f" % (epoch, val_acc))
            print ("++++++++++++++++++++++++")

            if val_acc >= best_val_acc:
                log({
                    "train_loss": float(np.mean(train_all_losses)),
                    "train_acc": float(np.mean(train_all_acc)),
                    "test_acc": val_acc,
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "n_parameters": n_parameters,
                }, optimizer, params, stats, opt)
                best_val_acc = val_acc

            # Free memory
            data = [],
            label = []
            y = []

        tock = time.clock()

        print ("epoch: %d, train loss: %f, train acc: %.2f, time: %.2f s" %
               (epoch, np.round(np.mean(train_all_losses), 6), np.mean(train_all_acc),
                    np.round((tock - tick))))



if __name__ == '__main__':
    main()