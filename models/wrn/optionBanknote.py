##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):

        # Training settings
        parser = argparse.ArgumentParser(description='Wide Residual Networks')
        # Model options
        parser.add_argument('--imageSize', default=32, type=int)
        parser.add_argument('--nchannels', default=1, type=int)

        parser.add_argument('--model', default='resnet', type=str)
        parser.add_argument('--depth', default=10, type=int)
        parser.add_argument('--width', default=1, type=float)
        parser.add_argument('--dataset', default='omniglot', type=str)
        parser.add_argument('--dataroot', default=
                                os.path.join('/home/aberenguel/Dataset/', 'banknote'), type=str)
        parser.add_argument('--dtype', default='float', type=str)
        parser.add_argument('--groups', default=1, type=int)
        parser.add_argument('--nthread', default=8, type=int)

        # Training options
        parser.add_argument('--batchSize', default=100, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--epochs', default=20000, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--weightDecay', default=0.0005, type=float)
        parser.add_argument('--epoch_step', default='[1000,2000,4000,6000,8000,10000,12000,14000,16000,50000]', type=str,
                            help='json list with epochs to drop lr on')
        parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
        parser.add_argument('--resume', default='', type=str)
        parser.add_argument('--optim_method', default='SGD', type=str)
        parser.add_argument('--eval_freq', default=100, type=int)

        # Augmentation
        parser.add_argument('--hflip', default=True, type=bool)
        parser.add_argument('--vflip', default=True, type=bool)
        parser.add_argument('--scale', default=0.2, type=float)
        parser.add_argument('--rotation_deg', default=20, type=float)
        parser.add_argument('--shear_deg', default=10, type=float)
        parser.add_argument('--translation_px', default=5, type=float)

        # Device options
        parser.add_argument('--cuda', default=True, type=bool)
        parser.add_argument('--save', default='', type=str,
                            help='save parameters and logs in this folder')
        parser.add_argument('--ngpu', default=1, type=int,
                            help='number of GPUs to use for training')
        parser.add_argument('--gpu_id', default='0', type=str,
                            help='id(s) for CUDA_VISIBLE_DEVICES')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()