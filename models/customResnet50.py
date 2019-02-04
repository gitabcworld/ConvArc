import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def spatial_pyramid_pooling(input, output_size):
    assert input.dim() == 4 and input.size(2) == input.size(3)
    F.max_pool2d(input, kernel_size=input.size(2) // output_size)

class CustomResNet50(nn.Module):

    def __init__(self, out_size=None):
        super(CustomResNet50, self).__init__()
        self.out_size = out_size
        self.resnet18 = models.resnet50(pretrained=True)
        if not(self.out_size is None):
            self.adaptativeAvgPooling = nn.AdaptiveAvgPool2d((out_size[0],out_size[1]))

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x) # 1024 x 14 x 14
        #x = self.resnet18.layer4(x) # 2048 x 7 x 7
        if not(self.out_size is None):
            x = self.adaptativeAvgPooling(x)
        return x


