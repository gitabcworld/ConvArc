import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import resnet18

class Siamese(nn.Module):
    """TFeat model definition
    """
    def __init__(self, isOmniglot = False):
        super(Siamese, self).__init__()
        self.net = resnet18.resnet18(pretrained=False)
        self.isOmniglot = isOmniglot
        # Reduce the resnet18
        if self.isOmniglot:
            # remove the 2 and 3th layer
            self.net.layer2 = None
            self.net.layer3 = None
            # create similarity score
            self.classifier = nn.Linear(128, 1)

    def forward(self, input):
        res = []
        for i in range(2):
            if self.isOmniglot:
                x = input[:,i,...]
                #replicate the input to have 3 channels
                x = x.expand((x.shape[0], 3, x.shape[2], x.shape[3]))
                x = self.net.conv1(x)
                x = self.net.bn1(x)
                x = self.net.relu(x)
                x = self.net.maxpool(x)
                x = self.net.layer1(x)
                x = self.net.avgpool(x)
                x = x.view(x.size(0), -1)
                res.append(x)
            else:
                x = self.net(x)
                x = x.view(x.size(0), -1)
                #x = self.classifier(x)
                res.append(x)
        #euclidean_distance = F.pairwise_distance(res[0], res[1])
        x = torch.cat((res[0], res[1]),1)
        x = nn.Sigmoid()(self.classifier(x))
        return x

    def save_to_file(self, file_path):
        torch.save(self.state_dict(), file_path)


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
