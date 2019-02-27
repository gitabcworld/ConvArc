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
if os.path.exists('D:/PhD/code/datasets/convarc'):
    dataroot = 'D:/PhD/code/datasets/convarc'
elif os.path.exists('/datatmp/users/aberenguel/convarc'):
    dataroot = '/datatmp/users/aberenguel/convarc'
else:
    dataroot = '/home/icar/datasets/convarc'

lst_parameters_change = [
    [   # 0
        ('datasetName', 'omniglot'),
        ('dataroot', os.path.join(dataroot,'omniglot')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'os/lstm_channel_1_carc_naive/'),
        ('nchannels', 1),
        ('train_num_batches', 10),
        ('partitionType', 'Lake'),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'os/lstm_channel_1_carc_naive/fcn.pt7'),
        ('wrn_load', pathResults + 'os/lstm_channel_1_carc_naive/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'os/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        ('arc_load', pathResults + 'os/lstm_channel_1_carc_naive/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', False),
        
        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'os/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_load_path', pathResults + 'os/lstm_channel_1_carc_naive/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive/context_optimizer.pt7')
    ],
    [   # 1
        ('datasetName', 'omniglot'),
        ('dataroot', os.path.join(dataroot,'omniglot')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'os/lstm_channel_1_carc_naive_standard/'),
        ('nchannels', 1),
        ('train_num_batches', 200000),
        ('partitionType', 'OmniglotStandard'),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'os/lstm_channel_1_carc_naive_standard/fcn.pt7'),
        ('wrn_load', pathResults + 'os/lstm_channel_1_carc_naive_standard/fcn.pt7'),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'os/lstm_channel_1_carc_naive_standard/ARCmodel.pt7'),
        ('arc_load', pathResults + 'os/lstm_channel_1_carc_naive_standard/ARCmodel.pt7'),
        ('arc_resume', False),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'os/lstm_channel_1_carc_naive_standard/context.pt7'),
        ('naive_full_load_path', pathResults + 'os/lstm_channel_1_carc_naive_standard/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive_standard/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'os/lstm_channel_1_carc_naive_standard/context_optimizer.pt7'),
    ],
    [   # 2
        ('datasetName', 'miniImagenet'),
        ('dataroot', os.path.join(dataroot,'mini_imagenet')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'miniimagenet'),
        ('nchannels', 3),
        ('train_num_batches', 10),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'miniimagenet/fcn.pt7'),
        ('wrn_load', pathResults + 'miniimagenet/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 2048),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'miniimagenet/ARCmodel.pt7'),
        ('arc_load', pathResults + 'miniimagenet/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', False),
        
        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'miniimagenet/context.pt7'),
        ('naive_full_load_path', pathResults + 'miniimagenet/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'miniimagenet/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'miniimagenet/context_optimizer.pt7'),

        ('use_coAttn', True),
        ('coAttn_size', (7,7)),
        ('coAttn_type', 'sum_abs_pow'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'miniimagenet/coAttn.pt7'),
        ('coattn_save', pathResults + 'miniimagenet/coAttn.pt7'),
    ],
    [   # 3
        ('datasetName', 'miniImagenet'),
        ('dataroot', os.path.join(dataroot,'mini_imagenet')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'miniimagenet224'),
        ('nchannels', 3),
        ('train_num_batches', 200000),

        ('apply_wrn', True),
        ('wrn_save', pathResults + 'miniimagenet224/fcn.pt7'),
        ('wrn_load', pathResults + 'miniimagenet224/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 64),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'miniimagenet224/ARCmodel.pt7'),
        ('arc_load', pathResults + 'miniimagenet224/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', False),
        
        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'miniimagenet224/context.pt7'),
        ('naive_full_load_path', pathResults + 'miniimagenet224/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'miniimagenet224/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'miniimagenet224/context_optimizer.pt7'),
    ],
    [  # 4
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'noCoAttnNaiveResnet50'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'noCoAttnNaiveResnet50/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'ResNet50'),
        ('wrn_save', pathResults + 'noCoAttnNaiveResnet50/fcn.pt7'),
        ('wrn_load', pathResults + 'noCoAttnNaiveResnet50/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 1024),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'noCoAttnNaiveResnet50/ARCmodel.pt7'),
        ('arc_load', pathResults + 'noCoAttnNaiveResnet50/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'noCoAttnNaiveResnet50/context.pt7'),
        ('naive_full_load_path', pathResults + 'noCoAttnNaiveResnet50/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'noCoAttnNaiveResnet50/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'noCoAttnNaiveResnet50/context_optimizer.pt7'),

        ('use_coAttn', False),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'noCoAttnNaiveResnet50/coAttn.pt7'),
        ('coattn_save', pathResults + 'noCoAttnNaiveResnet50/coAttn.pt7'),
    ],
    [  # 5
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'CoAttnNaiveResnet50'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'CoAttnNaiveResnet50/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'ResNet50'),
        ('wrn_save', pathResults + 'CoAttnNaiveResnet50/fcn.pt7'),
        ('wrn_load', pathResults + 'CoAttnNaiveResnet50/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 1024),
        ('arc_save', pathResults + 'CoAttnNaiveResnet50/ARCmodel.pt7'),
        ('arc_load', pathResults + 'CoAttnNaiveResnet50/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'CoAttnNaiveResnet50/context.pt7'),
        ('naive_full_load_path', pathResults + 'CoAttnNaiveResnet50/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'CoAttnNaiveResnet50/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'CoAttnNaiveResnet50/context_optimizer.pt7'),

        ('use_coAttn', True),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'CoAttnNaiveResnet50/coAttn.pt7'),
        ('coattn_save', pathResults + 'CoAttnNaiveResnet50/coAttn.pt7'),
    ],
    [  # 6
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'noCoAttnNaiveMobilenetV2'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'noCoAttnNaiveMobilenetV2/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'Mobilenetv2'),
        ('wrn_save', pathResults + 'noCoAttnNaiveMobilenetV2/fcn.pt7'),
        ('wrn_load', pathResults + 'noCoAttnNaiveMobilenetV2/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 96),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'noCoAttnNaiveMobilenetV2/ARCmodel.pt7'),
        ('arc_load', pathResults + 'noCoAttnNaiveMobilenetV2/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'noCoAttnNaiveMobilenetV2/context.pt7'),
        ('naive_full_load_path', pathResults + 'noCoAttnNaiveMobilenetV2/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'noCoAttnNaiveMobilenetV2/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'noCoAttnNaiveMobilenetV2/context_optimizer.pt7'),

        ('use_coAttn', False),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'noCoAttnNaiveMobilenetV2/coAttn.pt7'),
        ('coattn_save', pathResults + 'noCoAttnNaiveMobilenetV2/coAttn.pt7'),
    ],
    [  # 7
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'CoAttnNaiveMobilenetV2'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'CoAttnNaiveMobilenetV2/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'Mobilenetv2'),
        ('wrn_save', pathResults + 'CoAttnNaiveMobilenetV2/fcn.pt7'),
        ('wrn_load', pathResults + 'CoAttnNaiveMobilenetV2/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 96),
        ('arc_save', pathResults + 'CoAttnNaiveMobilenetV2/ARCmodel.pt7'),
        ('arc_load', pathResults + 'CoAttnNaiveMobilenetV2/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'CoAttnNaiveMobilenetV2/context.pt7'),
        ('naive_full_load_path', pathResults + 'CoAttnNaiveMobilenetV2/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'CoAttnNaiveMobilenetV2/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'CoAttnNaiveMobilenetV2/context_optimizer.pt7'),

        ('use_coAttn', True),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'CoAttnNaiveMobilenetV2/coAttn.pt7'),
        ('coattn_save', pathResults + 'CoAttnNaiveMobilenetV2/coAttn.pt7'),
    ],
    [  # 8
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'noCoAttnNaiveWRN'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'noCoAttnNaiveWRN/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'WideResidualNetworkImagenet'),
        ('wrn_save', pathResults + 'noCoAttnNaiveWRN/fcn.pt7'),
        ('wrn_load', pathResults + 'noCoAttnNaiveWRN/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 1024),
        ('arc_attn_type', 'LSTM'),
        ('arc_save', pathResults + 'noCoAttnNaiveWRN/ARCmodel.pt7'),
        ('arc_load', pathResults + 'noCoAttnNaiveWRN/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),
        #('arc_resume', False),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'noCoAttnNaiveWRN/context.pt7'),
        ('naive_full_load_path', pathResults + 'noCoAttnNaiveWRN/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'noCoAttnNaiveWRN/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'noCoAttnNaiveWRN/context_optimizer.pt7'),

        ('use_coAttn', False),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'noCoAttnNaiveWRN/coAttn.pt7'),
        ('coattn_save', pathResults + 'noCoAttnNaiveWRN/coAttn.pt7'),
    ],
    [  # 9
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'CoAttnNaiveWRN'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'WideResidualNetworkImagenet/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'WideResidualNetworkImagenet'),
        ('wrn_save', pathResults + 'CoAttnNaiveWRN/fcn.pt7'),
        ('wrn_load', pathResults + 'CoAttnNaiveWRN/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 1024),
        ('arc_save', pathResults + 'CoAttnNaiveWRN/ARCmodel.pt7'),
        ('arc_load', pathResults + 'CoAttnNaiveWRN/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'CoAttnNaiveWRN/context.pt7'),
        ('naive_full_load_path', pathResults + 'CoAttnNaiveWRN/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'CoAttnNaiveWRN/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'CoAttnNaiveWRN/context_optimizer.pt7'),

        ('use_coAttn', True),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'CoAttnNaiveWRN/coAttn.pt7'),
        ('coattn_save', pathResults + 'CoAttnNaiveWRN/coAttn.pt7'),
    ],
    [  # 10
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'noCoAttnNaivePeleeNet'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'noCoAttnNaivePeleeNet/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'PeleeNet'),
        ('wrn_save', pathResults + 'noCoAttnNaivePeleeNet/fcn.pt7'),
        ('wrn_load', pathResults + 'noCoAttnNaivePeleeNet/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 512),
        ('arc_save', pathResults + 'noCoAttnNaivePeleeNet/ARCmodel.pt7'),
        ('arc_load', pathResults + 'noCoAttnNaivePeleeNet/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'noCoAttnNaivePeleeNet/context.pt7'),
        ('naive_full_load_path', pathResults + 'noCoAttnNaivePeleeNet/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'noCoAttnNaivePeleeNet/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'noCoAttnNaivePeleeNet/context_optimizer.pt7'),

        ('use_coAttn', False),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'noCoAttnNaivePeleeNet/coAttn.pt7'),
        ('coattn_save', pathResults + 'noCoAttnNaivePeleeNet/coAttn.pt7'),
    ],
    [  # 11
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'CoAttnNaivePeleeNet'),
        ('batchSize', 20),
        ('naive_batchSize', 20),
        
        ('train_num_batches', 50000),
        ('val_freq', 1000),
        ('naive_full_epochs', 20000),
        ('naive_full_val_freq', 1000),
        ('path_tmp_data', pathResults + 'CoAttnNaivePeleeNet/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'PeleeNet'),
        ('wrn_save', pathResults + 'CoAttnNaivePeleeNet/fcn.pt7'),
        ('wrn_load', pathResults + 'CoAttnNaivePeleeNet/fcn.pt7'),
        #('wrn_load', None),

        ('arc_nchannels', 512),
        ('arc_save', pathResults + 'CoAttnNaivePeleeNet/ARCmodel.pt7'),
        ('arc_load', pathResults + 'CoAttnNaivePeleeNet/ARCmodel.pt7'),
        #('arc_load', None),
        ('arc_resume', True),

        ('naive_full_type', 'Naive'),
        ('naive_full_save_path', pathResults + 'CoAttnNaivePeleeNet/context.pt7'),
        ('naive_full_load_path', pathResults + 'CoAttnNaivePeleeNet/context.pt7'),
        ('naive_full_resume', True),

        ('arc_optimizer_path', pathResults + 'CoAttnNaivePeleeNet/arc_optimizer.pt7'),
        ('naive_full_optimizer_path', pathResults + 'CoAttnNaivePeleeNet/context_optimizer.pt7'),

        ('use_coAttn', True),
        ('coAttn_size', (14,14)),
        ('coAttn_type', 'None'),
        ('coAttn_p', 2),
        ('coattn_load', pathResults + 'CoAttnNaivePeleeNet/coAttn.pt7'),
        ('coattn_save', pathResults + 'CoAttnNaivePeleeNet/coAttn.pt7'),
    ],
    [  # 12
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'Triplet_PeleeNet'),
        ('batchSize', 20),

        #('mode', 'processor'),
        ('path_tmp_data', pathResults + 'Triplet_PeleeNet/data/'),

        ('train_num_batches', 10000),
        ('val_freq', 1000),
        ('val_num_batches', 250),
        ('test_num_batches', 500),
        ('path_tmp_data', pathResults + 'Triplet_PeleeNet/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'PeleeNetClassification'),
        ('wrn_save', pathResults + 'Triplet_PeleeNet/fcn.pt7'),
        ('wrn_load', pathResults + 'Triplet_PeleeNet/fcn.pt7'),
        #('wrn_load', None),
        
        ('arc_optimizer_path', pathResults + 'Triplet_PeleeNet/arc_optimizer.pt7'),

        ('use_coAttn', False),
    ],
    [  # 13
        ('datasetName', 'banknote'),
        ('dataroot', os.path.join(dataroot,'banknote')),
        ('one_shot_n_way', 5),
        ('one_shot_n_shot', 1),

        ('save', pathResults + 'Siamese_PeleeNet'),
        ('batchSize', 20),

        #('mode', 'processor'),
        ('path_tmp_data', pathResults + 'Siamese_PeleeNet/data/'),

        ('train_num_batches', 10000),
        ('val_freq', 1000),
        ('val_num_batches', 250),
        ('test_num_batches', 500),
        ('path_tmp_data', pathResults + 'Siamese_PeleeNet/data/'),

        ('apply_wrn', True),
        ('wrn_name_type', 'PeleeNetClassification'),
        ('wrn_save', pathResults + 'Siamese_PeleeNet/fcn.pt7'),
        ('wrn_load', pathResults + 'Siamese_PeleeNet/fcn.pt7'),
        #('wrn_load', None),
        
        ('arc_optimizer_path', pathResults + 'Siamese_PeleeNet/arc_optimizer.pt7'),

        ('use_coAttn', False),
    ],
]


def tranform_options(index, options):
    if index is None:
        index = options.train_settings
    params = lst_parameters_change[index]
    # New parameters
    for name_params, val_params in params:
        options.__setattr__(name_params, val_params)
    return options


class Options():
    def __init__(self):

        # General settings
        parser = argparse.ArgumentParser(description='Wide Residual Networks')

        parser.add_argument('--train_settings', type=int, default=0, help='num of option setting to test.')

        parser.add_argument('--train_num_batches', type=int, default=1500000, help='train epochs')
        parser.add_argument('--val_freq', type=int, default=1000, help='validation frequency')
        parser.add_argument('--val_num_batches', type=int, default=2, help='validation num batches')
        parser.add_argument('--test_num_batches', type=int, default=5, help='test num batches')
        parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
        parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                         ' model checkpoints in a separate folder.')
        parser.add_argument('--nthread', default=4, type=int)
        parser.add_argument('--dropout', default=0.1, type=float, help='Dropout for training. Done in the input'
                                                                       'in the case of ARC and in the residual block'
                                                                       'in the case of CARC.')
        parser.add_argument('--save', default=pathResults + 'os/lstm_0/',
                            help='path of the folder to create general info.')

        # Dataset
        parser.add_argument('--datasetName', default='miniImagenet', type=str, help='omniglot or miniimagenet datasets')
        parser.add_argument('--dataroot', default='D:/PhD/code/datasets/convarc/mini_imagenet', type=str)
        parser.add_argument('--datasetCompactSize', type=int, default=None, help='the height / width of the input image to save as a compacted file')
        parser.add_argument('--imageSize', type=int, default=None, help='the height / width of the input image to ARC')
        parser.add_argument('--nchannels', default=3, help='num channels input images.')
        parser.add_argument('--partitionType', default='30_10_10', type=str,
                            help='default: 30_10_10')
        parser.add_argument('--reduced_dataset', default=False, type=bool)
        parser.add_argument('--mode', default='generator', type=str)
        parser.add_argument('--path_tmp_data', default=pathResults + '/data', type=str)

        # Dataset specific banknote
        parser.add_argument('--datasetBanknoteOneShotSameClass', default=False, type=bool, help='Only uses one class to do the OneShot')

        # Augmentation
        parser.add_argument('--hflip', default=True, type=bool)
        parser.add_argument('--vflip', default=True, type=bool)
        parser.add_argument('--scale', default=1.2, type=float)
        parser.add_argument('--rotation_deg', default=20, type=float)
        parser.add_argument('--shear_deg', default=10, type=float)
        parser.add_argument('--translation_px', default=5, type=float)

        # Fully convolutional network parameters
        parser.add_argument('--apply_wrn', default=False, help='apply wide residual network')
        #parser.add_argument('--wrn_name_type', default='Mobilenetv2', type=str,
        #parser.add_argument('--wrn_name_type', default='Mobilenetv2Classification', type=str,
        #parser.add_argument('--wrn_name_type', default='WideResidualNetwork', type=str,
        #parser.add_argument('--wrn_name_type', default='WideResidualNetworkImagenet', type=str,
        #parser.add_argument('--wrn_name_type', default='WideResidualNetworkImagenetClassification', type=str,
        parser.add_argument('--wrn_name_type', default='ResNet50', type=str,
        #parser.add_argument('--wrn_name_type', default='ResNet50Classificaton', type=str,
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
        parser.add_argument('--wrn_groups', default=2, type=int, help='values=[0,1,2,3] Num of groups in WRN.')
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
        # Fully convolutional network parameters Update
        parser.add_argument('--fcn_applyOnDataLoader', default=False, type=bool, help='apply FCN in DataLoader')
        

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
        parser.add_argument('--naive_batchSize', type=int, default=10, help='input naive batch size')
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
        parser.add_argument('--naive_full_epochs', default=50000, type=int, help='num epochs training naive/full context')
        parser.add_argument('--naive_full_val_freq', default=1000, type=int, help='n epochs for evaluation dataset '
                                                                         'during training')
        # Optimizers
        parser.add_argument('--arc_optimizer_path', default=None, help='arc optimizer path to load/save')
        parser.add_argument('--naive_full_optimizer_path', default=None, help='naive/full optimizer path to load/save')

        # One-Shot parameters
        parser.add_argument('--one_shot_n_way', type=int, default=5, help='one-shot n-way. Default=5')
        parser.add_argument('--one_shot_n_shot', type=int, default=1, help='one-shot n-shot. Default=1')

        # Co-Attention parameters
        parser.add_argument('--use_coAttn', type=bool, default=True, help='Activate co-Attn. Default=True')
        parser.add_argument('--coAttn_size', type=int, nargs='+', default=(7,7), help='co-Attn size. Default=(7,7)')
        parser.add_argument('--coAttn_type', type=str, default='sum_abs', help='co-Attn type. Types: None/sum_abs/sum_abs_pow/max_abs_pow. Default=sum_abs')
        parser.add_argument('--coAttn_p', type=int, default=2, help='co-Attn size. Default=2')
        parser.add_argument('--coattn_load', type=str, default=pathResults + '/coAttn.pt7', help='co-Attn path load.')
        parser.add_argument('--coattn_save', type=str, default=pathResults + '/coAttn.pt7', help='co-Attn path save.')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()