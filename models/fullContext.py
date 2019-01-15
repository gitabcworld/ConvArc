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
from torch.autograd import Variable
import torch.nn.functional as F

class FullContextARC(nn.Module):
    def __init__(self, hidden_size, num_layers, vector_dim):
        """
        Initializes a multi layer bidirectional LSTM
        :param hidden_size: the neurons per layer 
        :param num_layers: number of layers                            
        :param batch_size: The experiments batch size
        """
        super(FullContextARC, self).__init__()
        self.lstm = nn.LSTM(input_size=vector_dim,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first = True)
        #self.dense1 = nn.Linear(hidden_size*2, 64)
        #self.dense2 = nn.Linear(64, 1)
        self.dense1 = nn.Linear(hidden_size * 2, 1)
        #self.relu = nn.ReLU()
        #self.logSoftmax = nn.LogSoftmax()

    def forward(self, x):
        ## bidirectional lstm
        x, (hn, cn) = self.lstm(x)
        #x = F.elu(self.dense1(x)).squeeze()
        x = torch.sigmoid(self.dense1(x).squeeze())
        #x = self.dense2(x).squeeze()
        #x = self.relu(x)
        #x = self.logSoftmax(x)
        return x



