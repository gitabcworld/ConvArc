##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttn(nn.Module):
    def __init__(self, size = (7,7), typeActivation = 'sum_abs', p = 2):
        """
        Initializes the naive ARC 
        """
        super(CoAttn, self).__init__()
        self.size = size
        self.size_pow = (pow(self.size[0],2),pow(self.size[1],2))
        self.typeActivation = typeActivation
        self.p = p
        self.W = nn.Parameter(torch.FloatTensor(self.size_pow[0],self.size_pow[1]).uniform_())

    def forward(self, x):
        
        batch_size, npairs, nfilters, sizeFilterX, sizeFilterY = x.shape
        assert sizeFilterX == self.size[0]
        assert sizeFilterY == self.size[1]

        Q_a = x[:,0,:,:,:] # 10 x 2048 x 7 x 7
        Q_b = x[:,1,:,:,:] # 10 x 2048 x 7 x 7
        
        if self.typeActivation == 'None': # keep the nfilters
            Q_a_1 = Q_a.contiguous().view(batch_size*nfilters,self.size[0],self.size[1]) # 20480 x 7 x 7
            Q_b_1 = Q_b.contiguous().view(batch_size*nfilters,self.size[0],self.size[1]) # 20480 x 7 x 7
            Q_a_1 = Q_a_1.view(batch_size*nfilters,-1).unsqueeze(2).expand(batch_size*nfilters,self.size_pow[0],self.size_pow[1]) # 20480 x 49 x 49
            Q_b_1 = Q_b_1.transpose(1,2).contiguous().view(batch_size*nfilters,-1).unsqueeze(2).expand(batch_size*nfilters,self.size_pow[0],self.size_pow[1]) # 20480 x 49 x 49
            L = Q_b_1*self.W*Q_a_1 # 20480 x 49 x 49
            A_a = torch.nn.Softmax(dim=1)(L.view(batch_size*nfilters,self.size_pow[0]*self.size_pow[1])).view(batch_size*nfilters,self.size_pow[0],self.size_pow[1]) # 20480 x 49 x 49
            A_b = torch.nn.Softmax(dim=1)(L.contiguous().transpose(1,2).contiguous().view(batch_size*nfilters,self.size_pow[0]*self.size_pow[1])).view(batch_size*nfilters,self.size_pow[0],self.size_pow[1]) # 20480 x 49 x 49
            Z_a = Q_a * A_a.mean(dim=1).view(batch_size,nfilters,self.size[0],self.size[1]) # 10 x 2048 x 7 x 7
            Z_b = Q_b * A_b.mean(dim=1).view(batch_size,nfilters,self.size[0],self.size[1]) # 10 x 2048 x 7 x 7
        else:
            ## Activation-Based attention transfer. Paper: Paying More Attention to Attention: Improving the Performance of Convolutional
            ## Neural Networks via Attention Transfer. ICLR 2017. 
            if self.typeActivation == 'sum_abs':
                Q_a_attn = Q_a.abs().sum(dim=1) # 10 x 7 x 7
                Q_b_attn = Q_b.abs().sum(dim=1) # 10 x 7 x 7
            elif self.typeActivation == 'sum_abs_pow': 
                Q_a_attn = Q_a.abs().pow(self.p).sum(dim=1) # 10 x 7 x 7
                Q_b_attn = Q_b.abs().pow(self.p).sum(dim=1) # 10 x 7 x 7
            elif self.typeActivation == 'max_abs_pow':
                Q_a_attn = Q_a.abs().pow(self.p).max(dim=1)[0] # 10 x 7 x 7
                Q_b_attn = Q_b.abs().pow(self.p).max(dim=1)[0] # 10 x 7 x 7

            Q_a_attn = Q_a_attn.view(batch_size,-1).unsqueeze(2).expand(batch_size,self.size_pow[0],self.size_pow[1]) # 10 x 49 x 49
            Q_b_attn = Q_b_attn.view(batch_size,-1).unsqueeze(2).expand(batch_size,self.size_pow[0],self.size_pow[1]) # 10 x 49 x 49
            L = Q_a_attn.transpose(1,2)*self.W*Q_b_attn # 10 x 49 x 49
            A_a = torch.nn.Softmax(dim=1)(L.view(-1,self.size_pow[0]*self.size_pow[1])).view(-1,self.size_pow[0],self.size_pow[1]) # 10 x 49 x 49
            A_b = torch.nn.Softmax(dim=1)(L.transpose(1,2).contiguous().view(-1,self.size_pow[0]*self.size_pow[1])).view(-1,self.size_pow[0],self.size_pow[1]) # 10 x 49 x 49
            Z_a = torch.bmm(Q_a.view(batch_size,nfilters,-1),A_a).view(batch_size,nfilters,self.size[0],self.size[1]) # 10 x 2048 x 7 x 7
            Z_b = torch.bmm(Q_b.view(batch_size,nfilters,-1),A_b).view(batch_size,nfilters,self.size[0],self.size[1]) # 10 x 2048 x 7 x 7
        x = torch.cat((Z_a.unsqueeze(1),Z_b.unsqueeze(1)),1)
        return x