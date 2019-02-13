import torch.nn as nn
import re
import hickle as hkl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#from visualization.visualize import make_dot
import os.path
from torch.utils import model_zoo


# Alternative Code for train WideResNet: https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

def conv2d(input, params, base, stride=1, pad=0):
    return F.conv2d(input, params[base + '_weight'],
                    params[base + '_bias'], stride, pad)

def group(input, params, base, stride, n):
    o = input
    for i in range(0, n):
        b_base = ('%s_block%d_conv') % (base, i)
        x = o
        o = conv2d(x, params, b_base + '0')
        o = F.relu(o)
        o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
        o = F.relu(o)
        o = conv2d(o, params, b_base + '2')
        if i == 0:
            o += conv2d(x, params, b_base + '_dim', stride=stride)
        else:
            o += x
        o = F.relu(o)
    return o


# Code from: https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb
class WideResNet_50_2(nn.Module):

    def __init__(self, useCuda= True, num_groups = 3, num_classes = None):
        super(WideResNet_50_2, self).__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
        
        # convert numpy arrays to torch Variables
        print('loaded weigths shape...')
        for k,v in sorted(params.items()):
            print(k, tuple(v.shape))
            if useCuda:
                params[k] = Variable(v.cuda(), requires_grad=True)
            else:
                params[k] = Variable(v, requires_grad=True)

        # Change the last fully connected layer
        if num_classes is not None:
            tmp = nn.Linear([256,512,1024,2048][num_groups], num_classes)
            if useCuda:
                tmp = tmp.cuda()
            params['fc_weight'] = tmp.weight
            params['fc_bias'] = tmp.bias

        print('\nTotal parameters:', sum(v.numel() for v in params.values()))

        # replace all '.' in the dictionary by '_'
        params_tmp = {}
        for key in params.keys():
            params_tmp[key.replace('.','_')] = params[key]
        params = params_tmp
        del params_tmp

        #create nn.Parameters to return them with parameters() function.
        self.params = nn.ParameterDict({})
        for key in params.keys():
            self.params.update({key:nn.Parameter(params[key],requires_grad=True)})

    def forward(self, input):

        # determine network size by parameters
        blocks = [sum([re.match('group%d_block\d+_conv0.weight' % j, k) is not None
                       for k in self.params.keys()]) for j in range(4)]

        o = F.conv2d(input, self.params['conv0_weight'], self.params['conv0_bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o = group(o, self.params, 'group0', 1, blocks[0])
        if self.num_groups >= 1:
            o = group(o, self.params, 'group1', 2, blocks[1])
        if self.num_groups >= 2:
            o = group(o, self.params, 'group2', 2, blocks[2])
        if self.num_groups >= 2:
            o = group(o, self.params, 'group3', 2, blocks[3])
        if self.num_classes is not None:
            o = F.avg_pool2d(o, o.shape[2], 1, 0)
            o = o.view(o.size(0), -1)
            o = F.linear(o, self.params['fc_weight'], self.params['fc_bias'])
        return o
