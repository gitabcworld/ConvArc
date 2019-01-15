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

pathResults = os.path.dirname(os.path.abspath(__file__)) + '/results/'


lst_parameters_change = [
    [
        ('datasetName', 'omniglot'),
        ('dataroot', 'D:/PhD/code/datasets/convarc/omniglot/'),
        ('one_shot_n_way', 20),
        ('one_shot_n_shot', 5),

        ('save', pathResults + 'os/lstm_channel_1_carc_naive/'),
        ('nchannels', 3),
        ('train_num_batches', 10),
        ('partitionType', 'Lake'),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'os/lstm_channel_1_carc_naive/'),
        ('wrn_load', pathResults + 'os/lstm_channel_1_carc_naive/'),
        #('wrn_load', None),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'os/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        ('arc_load', pathResults + 'os/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),
        
        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'os/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_load_path', pathResults + 'os/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive/context_optimizer.pt7'),
    ],
    [
        ('datasetName', 'omniglot'),
        ('dataroot', 'D:/PhD/code/datasets/convarc/omniglot/'),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'os_within/lstm_channel_1_carc_naive/'),
        ('nchannels', 1),
        ('train_num_batches', 10000),
        ('partitionType', 'Lake'),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'os_within/lstm_channel_1_carc_naive/'),
        ('wrn_load', pathResults + 'os_within/lstm_channel_1_carc_naive/'),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'os_within/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        ('arc_load', pathResults + 'os_within/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'os_within/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_load_path', pathResults + 'os_within/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'os_within/lstm_channel_1_carc_naive/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'os_within/lstm_channel_1_carc_naive/context_optimizer.pt7'),
    ],
    [
        ('datasetName', 'miniImagenet'),
        ('dataroot', 'D:/PhD/code/datasets/convarc/mini_imagenet'),
        ('one_shot_n_way', 20),
        ('one_shot_n_shot', 5),

        ('save', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/'),
        ('nchannels', 3),
        ('train_num_batches', 10),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/'),
        ('wrn_load', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/'),
        #('wrn_load', None),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/ARCmodel.pt7'),
        ('arc_load', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),
        
        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/context.pt7'),
        ('naive_full_load_path', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'os/lstm_miniimagenet_carc_naive_nway_20_nshot_5/context_optimizer.pt7'),
    ],

]


def tranform_options(index, options):
    params = lst_parameters_change[index]
    # New parameters
    for name_params, val_params in params:
        options.__setattr__(name_params, val_params)
    return options


class Options():
    def __init__(self):

        # General settings
        parser = argparse.ArgumentParser(description='Wide Residual Networks')
        parser.add_argument('--train_num_batches', type=int, default=10, help='train epochs')
        parser.add_argument('--val_freq', type=int, default=5, help='validation frequency')
        parser.add_argument('--val_num_batches', type=int, default=5, help='validation num batches')
        parser.add_argument('--test_num_batches', type=int, default=10, help='test num batches')
        parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                         ' model checkpoints in a separate folder.')
        parser.add_argument('--nthread', default=7, type=int)
        parser.add_argument('--dropout', default=0.1, type=float, help='Dropout for training. Done in the input'
                                                                       'in the case of ARC and in the residual block'
                                                                       'in the case of CARC.')
        parser.add_argument('--save', default=pathResults + 'os/lstm_0/',
                            help='path of the folder to create general info.')

        # Dataset
        parser.add_argument('--datasetName', default='miniImagenet', type=str, help='omniglot or miniimagenet datasets')
        parser.add_argument('--dataroot', default='D:/PhD/code/datasets/convarc/mini_imagenet', type=str)
        parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
        parser.add_argument('--nchannels', default=3, help='num channels input images.')
        parser.add_argument('--partitionType', default='30_10_10', type=str,
                            help='default: 30_10_10')
        parser.add_argument('--reduced_dataset', default=False, type=bool)
        parser.add_argument('--numTrials', type=int, default=32, help='number of Trials. Equal to batch size.')


        # Augmentation
        parser.add_argument('--hflip', default=True, type=bool)
        parser.add_argument('--vflip', default=True, type=bool)
        parser.add_argument('--scale', default=1.2, type=float)
        parser.add_argument('--rotation_deg', default=20, type=float)
        parser.add_argument('--shear_deg', default=10, type=float)
        parser.add_argument('--translation_px', default=5, type=float)

        # Fully convolutional network parameters
        parser.add_argument('--apply_wrn', default=False, help='apply wide residual network')
        parser.add_argument('--wrn_name_type', default='WideResidualNetwork', type=str,
                            help='type name of the network')
        parser.add_argument('--wrn_load', default=None, type=str,
                            help='path to the trained wide residual network.')
        parser.add_argument('--wrn_save', default=pathResults + 'os/lstm_0/', type=str,
                            help='folder to store the network')
        parser.add_argument('--wrn_train_resume', default=None, help='path to continue training from a checkpoint.')
        parser.add_argument('--wrn_depth', default=4, help='depth of wide Residual Networks.')
        parser.add_argument('--wrn_width', default=2, help='width of wide Residual Networks.')
        parser.add_argument('--wrn_ngpu', default=1, type=int, help='number of GPUs to use for training.')
        parser.add_argument('--wrn_full', default=False, type=bool, help='apply all groups in Wide Residual Networks.')
        parser.add_argument('--wrn_groups', default=1, type=int, help='values=[0,1,2,3] Num of groups in WRN.')
        parser.add_argument('--wrn_optim_method', default='SGD', type=str, help='optimizer SGD/ADAM.')
        parser.add_argument('--wrn_lr', default=1e-2, type=float, help='learning rate, default=0.0001')
        parser.add_argument('--wrn_epochs', default=3000, type=int, help='num epochs training wrn')
        parser.add_argument('--wrn_lr_patience', default=500000, type=int, help='num epochs to check lr stagnation.')
        parser.add_argument('--wrn_val_freq', default=25, type=int, help='n epochs for evaluation dataset '
                                                                         'during training')
        parser.add_argument('--wrn_targetsByCharacters', default=True, type=bool, help='Train by characters or '
                                                                                      'by alphabets')
        parser.add_argument('--wrn_num_classes', default=964, type=int, help='num classes of wide Residual Networks. '
                                                                    'If the previous parameter '
                                                                    'wrn_targetsByCharacters == True,there will be 964 '
                                                                    'classes in training. If set to false there will be'
                                                                    ' 30 classes.')

        # ARC Parameters
        parser.add_argument('--arc_load', default=None,
                            help='the model to load from. Start fresh if not specified.')
        parser.add_argument('--arc_resume', default=False, type=bool, help='continue ARC training?')
        parser.add_argument('--arc_save', default=pathResults + 'os/lstm_0/ARCmodel.pt7',
                            help='the model to load from. Start fresh if not specified.')
        parser.add_argument('--arc_nchannels', type=int, default=3, help='num inputs channels to discriminator')
        parser.add_argument('--arc_glimpseSize', type=int, default=4, help='the height / width of glimpse seen by ARC')
        parser.add_argument('--arc_numStates', type=int, default=512, help='number of hidden states in ARC controller')
        parser.add_argument('--arc_numGlimpses',type=int, default=8,
                            help='the number glimpses of each image in pair seen by ARC')
        parser.add_argument('--arc_lr', type=float, default=1e-4, help='learning rate, default=0.0001')
        parser.add_argument('--arc_lr_patience', type=int, default=500000, help='num epochs to check lr stagnation.')
        parser.add_argument('--arc_cost_per_sample', type=float, default=0.0, help='comput budget with a cost per sample.')
        # attention of arc
        parser.add_argument('--arc_attn_type', type=str, default='LSTM', help='type of attn LSTM. types: SkipLSTM/LSTM'
                                                                              ', default=LSTM')
        parser.add_argument('--arc_attn_unroll', type=bool, default=False, help='Unroll LSTM?.')
        parser.add_argument('--arc_attn_dense', type=bool, default=False, help='Dense conections?.')

        # Naive / FullContext Parameters
        parser.add_argument('--naive_batchSize', type=int, default=32, help='input naive batch size')
        parser.add_argument('--naive_numTrials', type=int, default=32, help='number of naive Trials. Equal to batch size.')
        parser.add_argument('--naive_full_type', default='Naive', type=str,
                            help='Select between -Naive- or -FullContext- models.')
        parser.add_argument('--naive_full_load_path', default=None,
                            help='the model to load from. Start fresh if not specified.')
        parser.add_argument('--naive_full_resume', default=False, type=bool, help='continue ARC training?')
        parser.add_argument('--naive_full_save_path', default=pathResults + 'os/lstm_0/context.pt7',
                            help='the model to load from. Start fresh if not specified.')
        parser.add_argument('--naive_full_layer_sizes', default=128, type=int,
                            help='Layer sizes in the LSTM from the full context model.')
        parser.add_argument('--naive_full_num_layers', default=1, type=int,
                            help='Num of LSTM layers from the full context model.')
        parser.add_argument('--naive_full_lr', default=3e-4, type=float, help='learning rate, default=0.0001')
        parser.add_argument('--naive_full_lr_patience', default=50000, type=int, help='num epochs to check lr stagnation.')
        parser.add_argument('--naive_full_epochs', default=10, type=int, help='num epochs training naive/full context')
        parser.add_argument('--naive_full_val_freq', default=5, type=int, help='n epochs for evaluation dataset '
                                                                         'during training')
        # Optimizers
        parser.add_argument('--arc_optimizer_path', default=None, help='arc optimizer path to load/save')
        parser.add_argument('--naive_full_optimizer_path', default=None, help='naive/full optimizer path to load/save')

        # One-Shot parameters
        parser.add_argument('--one_shot_n_way', type=int, default=5, help='one-shot n-way. Default=5')
        parser.add_argument('--one_shot_n_shot', type=int, default=1, help='one-shot n-shot. Default=1')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()