from __future__ import generators
import abc
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from wrn.utils import data_parallel
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
import torchvision.models as models
from datetime import datetime

#from wrn.wideResNet import resnet
from wrn.wideResNet_50_2 import WideResNet_50_2
from wrn.wideResNet import WideResNet
from customResnet50 import CustomResNet50
from mobilenetv2 import mobilenetv2
from peleeNet import peleeNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

cudnn.benchmark = True

# Factory class
class ConvCNNFactory:
    factories = {}
    @staticmethod
    def addFactory(id, convCNNFactory):
        ConvCNNFactory.factories.put[id] = convCNNFactory
    # A Template Method:
    @staticmethod
    def createCNN(id, opt):
        if id not in ConvCNNFactory.factories:
            ConvCNNFactory.factories[id] = \
                eval(id + '.Factory()')
        return ConvCNNFactory.factories[id].create(opt)

# Base class
class ConvCNN_Base(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, opt):
        super(ConvCNN_Base, self).__init__()
        self.opt = opt


######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.

class WideResidualNetwork(ConvCNN_Base):

    def __init__(self, opt):

        super(WideResidualNetwork, self).__init__(opt)

        # Initialize network 
        fully_convolutional = True
        self.model = WideResNet(self.opt['wrn_depth'], self.opt['wrn_width'], ninputs=self.opt['nchannels'],
                         num_groups=self.opt['wrn_groups'],
                         num_classes=None if fully_convolutional else self.opt['wrn_num_classes'],
                         dropout=self.opt['dropout'])
    
    #Overload parameters to not return the stats parameters which running_mean and running_std does not
    #contain derivative calculation.
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if not('stats' in name):
                yield param

    def forward(self, x):
        return self.model.forward(x)

    class Factory:
        def create(self,opt): return WideResidualNetwork(opt)


######################################################################
######################################################################
######################################################################

# Specialized Class. ResNet

class ResNet50(ConvCNN_Base):

    def __init__(self, opt):

        super(ResNet50, self).__init__(opt)

        # Initialize network
        #self.model = CustomResNet50(out_size=(100,60))
        self.model = CustomResNet50(out_size=None)

    def forward(self, x):
        return self.model(x)

    class Factory:
        def create(self,opt): return ResNet50(opt)

######################################################################
######################################################################
######################################################################

# Specialized Class. ResNet

class ResNet50Classificaton(ConvCNN_Base):

    def __init__(self, opt, num_classes = 2):

        super(ResNet50Classificaton, self).__init__(opt)
        # two class problem (genuine-counterfeit)
        num_classes = 2
        # Initialize network
        self.model = models.resnet50(pretrained=True)
        #block_expansion = 1 # Resnet 18,34
        block_expansion = 4 # Resnet 50,101,152
        self.model.fc = nn.Linear(512 * block_expansion, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def forward_features(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward_classifier(self,x):
        return self.model.fc(x)

    class Factory:
        def create(self,opt): return ResNet50Classificaton(opt)

######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.

class Mobilenetv2(ConvCNN_Base):

    def __init__(self, opt):

        super(Mobilenetv2, self).__init__(opt)

        # Initialize network
        self.model = mobilenetv2(pretrained=True)

    def forward(self, x):
        for i in range(14):
            x = self.model.features[i](x) # torch.Size([40, 96, 14, 14])
        return x
        # return self.model.features(x)  # returns B x 320 x 7 x 7
    
        # 0 torch.Size([40, 32, 112, 112])
        # 1 torch.Size([40, 16, 112, 112])
        # 2 torch.Size([40, 24, 56, 56])
        # 3 torch.Size([40, 24, 56, 56])
        # 4 torch.Size([40, 32, 28, 28])
        # 5 torch.Size([40, 32, 28, 28])
        # 6 torch.Size([40, 32, 28, 28])
        # 7 torch.Size([40, 64, 14, 14])
        # 8 torch.Size([40, 64, 14, 14])
        # 9 torch.Size([40, 64, 14, 14])
        # 10 torch.Size([40, 64, 14, 14])
        # 11 torch.Size([40, 96, 14, 14])
        # 12 torch.Size([40, 96, 14, 14])
        # 13 torch.Size([40, 96, 14, 14])
        # 14 torch.Size([40, 160, 7, 7])
        # 15 torch.Size([40, 160, 7, 7])
        # 16 torch.Size([40, 160, 7, 7])
        # 17 torch.Size([40, 320, 7, 7])

    class Factory:
        def create(self,opt): return Mobilenetv2(opt)

######################################################################
######################################################################
######################################################################

class Mobilenetv2Classification(ConvCNN_Base):

    def __init__(self, opt):

        super(Mobilenetv2Classification, self).__init__(opt)

        # Initialize network
        self.model = mobilenetv2(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=2)

    def forward(self, x):
        return self.model(x)

    def forward_features(self,x):
        x = self.model.features(x)
        x = self.model.conv(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward_classifier(self,x):
        return self.model.classifier(x)


    class Factory:
        def create(self,opt): return Mobilenetv2Classification(opt)



######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.
class InceptionNetwork(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader,
                    model,
                    optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            train_preds = model(inputs)
            train_preds = train_preds[0] #get only the last fc
            training_loss = F.cross_entropy(train_preds, targets)
            if isTraining:
                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            train_probs, train_classes = torch.max(train_preds, 1)
            train_acc = accuracy_score(targets.data.cpu().numpy(),
                                       train_classes.data.cpu().numpy())

            all_acc.append(train_acc)
            all_losses.append(training_loss.data.cpu().numpy()[0])
            # Free memory
            inputs = []
            targets = []
        return model, all_acc, all_losses

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        inception = models.inception_v3(pretrained=True)
        # change the last fully convolutional layer
        inception.fc = nn.Linear(2048, self.opt['wrn_num_classes'])
        # Anulation of AuxLogits
        #modules_filtered = [not i == inception.AuxLogits for i in inception.children()]
        #inception = nn.Sequential(*list(np.array(list(inception.children()))[modules_filtered]))
        inception.AuxLogits.fc = nn.Linear(768, self.opt['wrn_num_classes'])

        # create optimizer
        optimizer = self.create_optimizer(params=inception.parameters(),
                                          method=self.opt['wrn_optim_method'],
                                          lr = self.opt['wrn_lr'])

        if resume is not None:
            state_dict = torch.load(resume)
            inception.load_state_dict(state_dict['state_dict'])
            optimizer.load_state_dict(state_dict['optimizer'])

        #inception = torch.nn.DataParallel(inception)
        if self.opt['cuda']:
            inception = inception.cuda()
        inception.train()

        try:
            epoch = 0
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                print ('epoch: %d' % epoch)
                inception, train_all_acc, train_all_losses = \
                                    self.__doepoch__(train_loader,
                                    inception, optimizer, isTraining=True)

                # update optimizer
                if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer = self.create_optimizer(params=inception.parameters(),
                                                      method=self.opt['wrn_optim_method'],
                                                      lr=lr * self.opt['wrn_lr_decay_ratio'])

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    inception.eval()
                    inception, val_all_acc, val_all_losses = \
                                    self.__doepoch__(val_loader,
                                     inception, optimizer = None, isTraining=False)
                    val_acc_epoch = np.mean(val_all_acc)
                    val_losses_epoch = np.mean(val_all_losses)
                    if val_acc_epoch >= best_val_acc:
                        n_parameters = sum(p.numel() for p in inception.parameters())
                        self.log({
                            "train_loss": float(np.mean(train_all_losses)),
                            "train_acc": float(np.mean(train_all_acc)),
                            "test_acc": val_acc_epoch,
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, inception.parameters())
                        best_val_acc = val_acc_epoch
                    inception.train()

        except KeyboardInterrupt:
            pass

        return inception, best_val_acc


    def load(self, modelPath, fully_convolutional = False):

        inception = models.inception_v3(pretrained=False)
        state_dict = torch.load(modelPath)
        inception.load_state_dict(state_dict['state_dict'])
        n_parameters = sum(p.numel() for p in inception.parameters())
        print ('\nTotal number of parameters:'+ n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.items()},
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['save'], 'model.pt7'), 'w'))
        z = vars(self.opt).copy(); z.update(dictParams)
        logname = os.path.join(self.opt['save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print (z)

    def forward(self, data_loader, inception):
        model, data_all_acc, data_all_losses = \
            self.__doepoch__(self, data_loader,
                             inception, optimizer = None, isTraining=False)
        return model, data_all_acc, data_all_losses

    def forward_fcnn(self, data_loader, f, params, stats):
        feature_maps = []
        targets = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            target = Variable(label)
            isTraining = False
            featureMap = data_parallel(f, inputs, params, stats, isTraining,
                                        np.arange(self.opt['wrn_ngpu']))
            feature_maps.append(featureMap)
            targets.append(target)
        return feature_maps, targets

    class Factory:
        def create(self,opt): return InceptionNetwork(opt)

######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.
class VGGNetwork(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader,
                    model,
                    optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            train_preds = model(inputs)
            training_loss = F.cross_entropy(train_preds, targets)
            if isTraining:
                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            train_probs, train_classes = torch.max(train_preds, 1)
            train_acc = accuracy_score(targets.data.cpu().numpy(),
                                       train_classes.data.cpu().numpy())

            all_acc.append(train_acc)
            all_losses.append(training_loss.data.cpu().numpy()[0])
            # Free memory
            inputs = []
            targets = []
        return model, all_acc, all_losses

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        vgg16 = models.vgg16(pretrained=True)
        # change the last fully convolutional layer
        vgg16.classifier = nn.Sequential(*list(vgg16.classifier)[:-1] + [nn.Linear(4096, self.opt['wrn_num_classes'])])

        # create optimizer
        optimizer = self.create_optimizer(params=vgg16.parameters(),
                                          method=self.opt['wrn_optim_method'],
                                          lr = self.opt['wrn_lr'])

        if resume is not None:
            state_dict = torch.load(resume)
            vgg16.load_state_dict(state_dict['state_dict'])
            optimizer.load_state_dict(state_dict['optimizer'])

        if self.opt['cuda']:
            vgg16 = vgg16.cuda()
        vgg16.train()

        try:
            epoch = 0
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                print ('epoch: %d' % epoch)
                vgg16, train_all_acc, train_all_losses = \
                                    self.__doepoch__(train_loader,
                                                     vgg16, optimizer, isTraining=True)

                # update optimizer
                if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer = self.create_optimizer(params=vgg16.parameters(),
                                                      method=self.opt['wrn_optim_method'],
                                                      lr=lr * self.opt['wrn_lr_decay_ratio'])

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    vgg16.eval()
                    vgg16, val_all_acc, val_all_losses = \
                                    self.__doepoch__(val_loader,
                                                     vgg16, optimizer = None, isTraining=False)
                    val_acc_epoch = np.mean(val_all_acc)
                    val_losses_epoch = np.mean(val_all_losses)
                    if val_acc_epoch >= best_val_acc:
                        n_parameters = sum(p.numel() for p in vgg16.parameters())
                        self.log({
                            "train_loss": float(np.mean(train_all_losses)),
                            "train_acc": float(np.mean(train_all_acc)),
                            "test_acc": val_acc_epoch,
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, vgg16.parameters())
                        best_val_acc = val_acc_epoch
                        vgg16.train()

        except KeyboardInterrupt:
            pass

        return vgg16, best_val_acc


    def load(self, modelPath, fully_convolutional = False):

        inception = models.vgg16(pretrained=False)
        state_dict = torch.load(modelPath)
        inception.load_state_dict(state_dict['state_dict'])
        n_parameters = sum(p.numel() for p in inception.parameters())
        print ('\nTotal number of parameters:'+ n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.items()},
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['save'], 'model.pt7'), 'w'))
        z = vars(self.opt).copy(); z.update(dictParams)
        logname = os.path.join(self.opt['save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print (z)

    def forward(self, data_loader, inception):
        model, data_all_acc, data_all_losses = \
            self.__doepoch__(self, data_loader,
                             inception, optimizer = None, isTraining=False)
        return model, data_all_acc, data_all_losses

    def forward_fcnn(self, data_loader, f, params, stats):
        feature_maps = []
        targets = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            target = Variable(label)
            isTraining = False
            featureMap = data_parallel(f, inputs, params, stats, isTraining,
                                        np.arange(self.opt['wrn_ngpu']))
            feature_maps.append(featureMap)
            targets.append(target)
        return feature_maps, targets

    class Factory:
        def create(self,opt): return VGGNetwork(opt)

######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.
class SqueezeNetNetwork(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader,
                    model,
                    optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            #train_preds = model(inputs) ==> Not working, why???
            train_preds = model.classifier(model.features(inputs))
            train_preds = train_preds.squeeze()
            training_loss = F.cross_entropy(train_preds.squeeze(), targets)
            if isTraining:
                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

            train_probs, train_classes = torch.max(train_preds, 1)
            train_acc = accuracy_score(targets.data.cpu().numpy(),
                                       train_classes.data.cpu().numpy())

            all_acc.append(train_acc)
            all_losses.append(training_loss.data.cpu().numpy()[0])
            # Free memory
            inputs = []
            targets = []
        return model, all_acc, all_losses

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        squeezenet = models.squeezenet1_0(pretrained=True)
        # change the last fully convolutional layer
        temp = list(squeezenet.classifier)
        temp[1] = torch.nn.Conv2d(512, self.opt['wrn_num_classes'], kernel_size=(1, 1),
                                                         stride=(1, 1))
        squeezenet.classifier = nn.Sequential(*temp)

        # create optimizer
        optimizer = self.create_optimizer(params=squeezenet.parameters(),
                                          method=self.opt['wrn_optim_method'],
                                          lr = self.opt['wrn_lr'])

        if resume is not None:
            state_dict = torch.load(resume)
            squeezenet.load_state_dict(state_dict['state_dict'])
            optimizer.load_state_dict(state_dict['optimizer'])

        if self.opt['cuda']:
            squeezenet = squeezenet.cuda()
        squeezenet.train()

        try:
            epoch = 0
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                print ('epoch: %d' % epoch)
                squeezenet, train_all_acc, train_all_losses = \
                                    self.__doepoch__(train_loader,
                                                     squeezenet, optimizer, isTraining=True)

                # update optimizer
                if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer = self.create_optimizer(params=squeezenet.parameters(),
                                                      method=self.opt['wrn_optim_method'],
                                                      lr=lr * self.opt['wrn_lr_decay_ratio'])

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    squeezenet.eval()
                    squeezenet, val_all_acc, val_all_losses = \
                                    self.__doepoch__(val_loader,
                                                     squeezenet, optimizer = None, isTraining=False)
                    val_acc_epoch = np.mean(val_all_acc)
                    val_losses_epoch = np.mean(val_all_losses)
                    if val_acc_epoch >= best_val_acc:
                        n_parameters = sum(p.numel() for p in squeezenet.parameters())
                        self.log({
                            "train_loss": float(np.mean(train_all_losses)),
                            "train_acc": float(np.mean(train_all_acc)),
                            "test_acc": val_acc_epoch,
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, squeezenet.parameters())
                        best_val_acc = val_acc_epoch
                        squeezenet.train()

        except KeyboardInterrupt:
            pass

        return squeezenet, best_val_acc


    def load(self, modelPath, fully_convolutional = False ):

        inception = models.squeezenet(pretrained=False)
        state_dict = torch.load(modelPath)
        inception.load_state_dict(state_dict['state_dict'])
        n_parameters = sum(p.numel() for p in inception.parameters())
        print ('\nTotal number of parameters:'+ n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.items()},
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['save'], 'model.pt7'), 'w'))
        z = vars(self.opt).copy(); z.update(dictParams)
        logname = os.path.join(self.opt['save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print (z)

    def forward(self, data_loader, inception):
        model, data_all_acc, data_all_losses = \
            self.__doepoch__(self, data_loader,
                             inception, optimizer = None, isTraining=False)
        return model, data_all_acc, data_all_losses

    def forward_fcnn(self, data_loader, f, params, stats):
        feature_maps = []
        targets = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            target = Variable(label)
            isTraining = False
            featureMap = data_parallel(f, inputs, params, stats, isTraining,
                                        np.arange(self.opt['wrn_ngpu']))
            feature_maps.append(featureMap)
            targets.append(target)
        return feature_maps, targets

    class Factory:
        def create(self,opt): return SqueezeNetNetwork(opt)

######################################################################
######################################################################
######################################################################

class WideResidualNetworkImagenet(ConvCNN_Base):

    def __init__(self, opt):

        super(WideResidualNetworkImagenet, self).__init__(opt)
        self.model = WideResNet_50_2(useCuda= torch.cuda.is_available(), num_groups = self.opt['wrn_groups'], num_classes = None)
    
    #Overload parameters to not return the stats parameters which running_mean and running_std does not
    #contain derivative calculation.
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if not('stats' in name):
                yield param

    def forward(self, x):
        return self.model.forward(x)

    class Factory:
        def create(self,opt): return WideResidualNetworkImagenet(opt)

######################################################################
######################################################################
######################################################################

class WideResidualNetworkImagenetClassification(ConvCNN_Base):

    def __init__(self, opt):

        super(WideResidualNetworkImagenetClassification, self).__init__(opt)
        self.model = WideResNet_50_2(useCuda= torch.cuda.is_available(), num_groups = self.opt['wrn_groups'], num_classes = 2)
    
    #Overload parameters to not return the stats parameters which running_mean and running_std does not
    #contain derivative calculation.
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if not('stats' in name):
                yield param

    def forward(self, x):
        return self.model.forward(x)

    def forward_features(self, x):
        old_state_num_classes = self.model.num_classes
        # Set the num classes to None
        self.model.num_classes = None
        x = self.model.forward(x)
        self.model.num_classes = old_state_num_classes
        return x

    def forward_classifier(self, x):
        # Execute only the classifier
        x = F.avg_pool2d(x, x.shape[2], 1, 0)
        x = x.view(x.size(0), -1)
        x = F.linear(x, self.model.params['fc_weight'], self.model.params['fc_bias'])
        return x


    class Factory:
        def create(self,opt): return WideResidualNetworkImagenetClassification(opt)

######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.

class PeleeNet(ConvCNN_Base):

    def __init__(self, opt):

        super(PeleeNet, self).__init__(opt)

        # Initialize network
        self.model = peleeNet(pretrained=True)

    def forward(self, x):
        for i in range(9):
            x = self.model.features[i](x) 
        return x # returns B x 512 x 14 x 14
    
        #def forward(self, x):
        #    return self.model.features(x) # returns # B x 704 x 7 x 7

    class Factory:
        def create(self,opt): return PeleeNet(opt)

######################################################################
######################################################################
######################################################################

class PeleeNetClassification(ConvCNN_Base):

    def __init__(self, opt):

        super(PeleeNetClassification, self).__init__(opt)

        # Initialize network
        self.model = peleeNet(pretrained=True)
        self.model.classifier = nn.Linear(704,2, bias=True)

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        x = self.model.features(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3))).view(x.size(0), -1)
        if self.model.drop_rate > 0:
            x = F.dropout(x, p=self.model.drop_rate, training=self.model.training)
        return x

    def forward_classifier(self, x):
        # Execute only the classifier
        x = self.model.classifier(x)
        return x


    class Factory:
        def create(self,opt): return PeleeNetClassification(opt)



######################################################################
######################################################################
######################################################################


# Generate all name of the factorization classes
def ConvCNN_NameGen():
    types = ConvCNN_Base.__subclasses__()
    for type in types:
        yield type.__name__










