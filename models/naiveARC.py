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

class NaiveARC(nn.Module):
    def __init__(self, numStates):
        """
        Initializes the naive ARC 
        """
        super(NaiveARC, self).__init__()
        self.dense1 = nn.Linear(numStates, 64)
        self.dense2 = nn.Linear(64, 1)
        #self.dense1 = nn.Linear(numStates, 1)

    def forward(self, x):
        d1 = torch.squeeze(F.elu(self.dense1(x)))
        #d1 = torch.squeeze(self.dense1(x))
        d2 = torch.sigmoid(torch.squeeze(self.dense2(d1)))
        return d2
        #decision = nn.Softmax(d2)
        #return decision
        #return d1





