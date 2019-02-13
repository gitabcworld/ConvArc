import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
        flatten_params, flatten_stats
import numpy as np


class WideResNet(nn.Module):

    def __init__(self, depth, width, ninputs = 3,
                 num_groups = 3, num_classes = None, dropout = 0.):

        super(WideResNet, self).__init__()
        self.depth = depth
        self.width = width
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = True # Training

        #widths = torch.Tensor([16, 32, 64]).mul(width).int()
        widths = np.array([16, 32, 64]).astype(np.int)*width

        def gen_block_params(ni, no):
            return {
                'conv0': conv_params(ni, no, 3),
                'conv1': conv_params(no, no, 3),
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'convdim': conv_params(ni, no, 1) if ni != no else None,
            }

        def gen_group_params(ni, no, count):
            return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                    for i in range(count)}

        def gen_group_stats(ni, no, count):
            return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no)}
                    for i in range(count)}

        params = {'conv0': conv_params(ni=ninputs, no=widths[0], k=3)}
        stats = {}

        for i in range(num_groups+1):
            if i == 0:
                params.update({'group'+str(i): gen_group_params(widths[i], widths[i], depth)})
                stats.update({'group'+str(i): gen_group_stats(widths[i], widths[i], depth)})
            else:
                params.update({'group'+str(i): gen_group_params(widths[i-1], widths[i], depth)})
                stats.update({'group'+str(i): gen_group_stats(widths[i-1], widths[i], depth)})

        if num_classes is not None:
            params.update({'fc': linear_params(widths[i], num_classes)})
        params.update({'bn': bnparams(widths[i])})
        stats.update({'bn': bnstats(widths[i])})

        params = flatten_params(params)
        stats = flatten_stats(stats)        

        self.params = nn.ParameterDict({})
        self.stats = nn.ParameterDict({})
        for key in params.keys():
            self.params.update({key:nn.Parameter(params[key],requires_grad=True)})
        for key in stats.keys():
            self.stats.update({key:nn.Parameter(stats[key], requires_grad=False)})  

    ''' TODO:CHECK
    def train(self, mode=True):
        self.mode = mode
        for key in self.params.keys():
            self.params[key].requires_grad = self.mode
        return super(WideResNet, self).train(mode=mode)

    def eval(self):
        self.mode = False
        for key in self.params.keys():
            self.params[key].requires_grad = self.mode
        return super(WideResNet, self).eval()
    '''

    def forward(self, input):

        def activation(x, params, stats, base, mode):
            return F.relu(F.batch_norm(x, weight=params[base + '_weight'],
                                       bias=params[base + '_bias'],
                                       running_mean=stats[base + '_running_mean'],
                                       running_var=stats[base + '_running_var'],
                                       training=mode, momentum=0.1, eps=1e-5), inplace=True)

        def block(x, params, stats, base, mode, stride):
            o1 = activation(x, params, stats, base + '_bn0', mode)
            y = F.conv2d(o1, params[base + '_conv0'], stride=stride, padding=1)
            o2 = activation(y, params, stats, base + '_bn1', mode)
            o2 = torch.nn.Dropout(p=self.dropout)(o2) # Dropout from the code of ARC. dropout = 0.3
            z = F.conv2d(o2, params[base + '_conv1'], stride=1, padding=1)
            if base + '_convdim' in params:
                return z + F.conv2d(o1, params[base + '_convdim'], stride=stride)
            else:
                return z + x

        def group(o, params, stats, base, mode, stride):
            for i in range(self.depth):
                o = block(o, params, stats, '%s_block%d' % (base, i), mode, stride if i == 0 else 1)
            return o

        assert input.is_cuda == self.params['conv0'].is_cuda
        if input.is_cuda:
            assert input.get_device() == self.params['conv0'].get_device()
        x = F.conv2d(input.float(), self.params['conv0'], padding=1)
        o = group(x, self.params, self.stats, 'group0', self.mode, stride=1)
        if self.num_groups >= 1:
            o = group(o, self.params, self.stats, 'group1', self.mode, stride=2)
        if self.num_groups >= 2:
            o = group(o, self.params, self.stats, 'group2', self.mode, stride=2)
        o = activation(o, self.params, self.stats, 'bn', self.mode)

        if self.num_classes is not None:
            o = F.avg_pool2d(o, o.shape[2], 1, 0)
            o = o.view(o.size(0), -1)
            o = F.linear(o, self.params['fc_weight'], self.params['fc_bias'])
        return o






def resnet(depth, width, num_classes, is_full_wrn = True, is_fully_convolutional = False):
    #assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    #n = (depth - 4) // 6
    #wrn = WideResNet(depth, width, ninputs=3,useCuda=True, num_groups=3, num_classes=num_classes)
    n = depth
    widths = torch.Tensor([16, 32, 64]).mul(width).int()

    def gen_block_params(ni, no):
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'bn0': bnparams(ni),
            'bn1': bnparams(no),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no)}
                for i in range(count)}

    params = {
        'conv0': conv_params(3,16,3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': bnparams(widths[2]),
        'fc': linear_params(widths[2], num_classes),
    }

    stats = {
        'group0': gen_group_stats(16, widths[0], n),
        'group1': gen_group_stats(widths[0], widths[1], n),
        'group2': gen_group_stats(widths[1], widths[2], n),
        'bn': bnstats(widths[2]),
    }
    if not is_full_wrn:
        ''' omniglot '''
        params['bn'] = bnparams(widths[1])
        #params['fc'] = linear_params(widths[1]*16*16, num_classes)
        params['fc'] = linear_params(widths[1], num_classes)
        stats['bn'] = bnstats(widths[1])
        '''
        # banknote
        params['bn'] = bnparams(widths[2])
        #params['fc'] = linear_params(widths[2]*16*16, num_classes)
        params['fc'] = linear_params(widths[2], num_classes)
        stats['bn'] = bnstats(widths[2])
        '''



    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight=params[base + '.weight'],
                                   bias=params[base + '.bias'],
                                   running_mean=stats[base + '.running_mean'],
                                   running_var=stats[base + '.running_var'],
                                   training=mode, momentum=0.1, eps=1e-5), inplace=True)

    def block(x, params, stats, base, mode, stride):
        o1 = activation(x, params, stats, base + '.bn0', mode)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = activation(y, params, stats, base + '.bn1', mode)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def full_wrn(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = activation(g2, params, stats, 'bn', mode)
        o = F.avg_pool2d(o, o.shape[2], 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    def not_full_wrn(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        # omniglot
        o = activation(g1, params, stats, 'bn', mode)
        o = F.avg_pool2d(o, o.shape[2], 1, 0)
        # banknote
        '''
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = activation(g2, params, stats, 'bn', mode)
        o = F.avg_pool2d(o, 16, 1, 0)
        '''
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    def fcn_full_wrn(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = activation(g2, params, stats, 'bn', mode)
        return o

    def fcn_not_full_wrn(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        o = activation(g1, params, stats, 'bn', mode)
        return o

    if is_fully_convolutional:
        if is_full_wrn:
            return fcn_full_wrn, flat_params, flat_stats
        else:
            return fcn_not_full_wrn, flat_params, flat_stats
    else:
        if is_full_wrn:
            return full_wrn, flat_params, flat_stats
        else:
            return not_full_wrn, flat_params, flat_stats

