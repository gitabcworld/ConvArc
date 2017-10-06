import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import resnet18

class Siamese(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(Siamese, self).__init__()
        self.net = resnet18.resnet18(pretrained=True)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
