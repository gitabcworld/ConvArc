import torch.nn as nn
import re
import hickle as hkl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from visualization.visualize import make_dot
import os.path
import urllib2


__all__ = ['WideResNet', 'wrn']
model_urls = {
    'wrn': 'https://s3.amazonaws.com/pytorch/h5models/wide-resnet-50-2-export.hkl',
}

# Alternative Code for train WideResNet: https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

def conv2d(input, params, base, stride=1, pad=0):
    return F.conv2d(input, params[base + '.weight'],
                    params[base + '.bias'], stride, pad)

def group(input, params, base, stride, n):
    o = input
    for i in range(0, n):
        b_base = ('%s.block%d.conv') % (base, i)
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
class WideResNetImageNet(nn.Module):

    def __init__(self, useCuda= True, num_groups = 3, num_classes = None):
        super(WideResNetImageNet, self).__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        pathWeigths = os.path.join('./data', 'wide-resnet-50-2-export.hkl')
        if not os.path.isfile(pathWeigths):
            print('Downloading pretrained WideResNet to %s' % pathWeigths)
            urlModel = 'https://s3.amazonaws.com/pytorch/h5models/wide-resnet-50-2-export.hkl'
            wrnWeights = urllib2.urlopen(urlModel)
            with open(pathWeigths, 'wb') as output:
                output.write(wrnWeights.read())

        self.params = hkl.load(pathWeigths)
        # convert numpy arrays to torch Variables
        print('Printing loaded weigths shape...')
        for k, v in sorted(self.params.items()):
            print k, v.shape
            if useCuda:
                self.params[k] = Variable(torch.from_numpy(v).cuda(), requires_grad=True)
            else:
                self.params[k] = Variable(torch.from_numpy(v), requires_grad=True)

        # Change the last fully connected layer
        if num_classes is not None:
            tmp = nn.Linear([256,512,1024,2048][num_groups], num_classes)
            if useCuda:
                tmp = tmp.cuda()
            self.params['fc.weight'] = tmp.weight
            self.params['fc.bias'] = tmp.bias

        print '\nTotal parameters:', sum(v.numel() for v in self.params.values())


    def forward(self, input):

        # determine network size by parameters
        blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                       for k in self.params.keys()]) for j in range(4)]

        o = F.conv2d(input, self.params['conv0.weight'], self.params['conv0.bias'], 2, 3)
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
            o = F.linear(o, self.params['fc.weight'], self.params['fc.bias'])
        return o
