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
from models.wrn.utils import data_parallel
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
import torchvision.models as models
from datetime import datetime

from models.wrn.resnet import resnet
from models.wrn.wrn import WideResNetImageNet
from models.wrn.resnet import WideResNet
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
        if not (id in ConvCNNFactory.factories):
            ConvCNNFactory.factories[id] = \
                eval(id + '.Factory()')
        return ConvCNNFactory.factories[id].create(opt)

# Base class
class ConvCNN_Base(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, opt):
        self.image_size = opt['imageSize']

    def create_optimizer(self, params, method = 'SGD', lr = 1e04,
                         momentum = 0.9, weight_decay = 0.0005 ):
        print ('creating optimizer with lr = %f' % lr)
        if method == 'SGD':
            return torch.optim.SGD(params, lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)
        elif method == 'Adam':
            return torch.optim.Adam(params, lr)

    @abc.abstractmethod
    def train(self, train_loader, val_loader, resume = None):
        pass

    @abc.abstractmethod
    def load(self, modelPath, fully_convolutional = False):
        pass


######################################################################
######################################################################
######################################################################

# Specialized Class. Wide Residual Networks.

class WideResidualNetwork(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader, f, optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            weigths = f(inputs)
            loss = F.cross_entropy(weigths, targets)
            if isTraining:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs, classes = torch.max(nn.Softmax(dim=1)(weigths), 1)
            acc = accuracy_score(targets.data.cpu().numpy(),
                                       classes.data.cpu().numpy())

            all_acc.append(acc)
            all_losses.append(loss.data.cpu().numpy()[0])
            # Free memory
            inputs = []
            targets = []
        return f, np.mean(all_acc), np.mean(all_losses)

    def create(self, fully_convolutional = False):
        # create network
        wrn = WideResNet(self.opt['wrn_depth'], self.opt['wrn_width'], ninputs=self.opt['nchannels'],
                         num_groups=self.opt['wrn_groups'],
                         num_classes=None if fully_convolutional else self.opt['wrn_num_classes'],
                         dropout=self.opt['dropout'])
        wrn.cuda()
        return wrn

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        # create network
        wrn = WideResNet(self.opt['wrn_depth'], self.opt['wrn_width'], ninputs = self.opt['nchannels'],
                        num_groups = self.opt['wrn_groups'], num_classes = self.opt['wrn_num_classes'],
                        dropout=self.opt['dropout'])
        wrn.cuda()

        # create optimizer
        optimizer = self.create_optimizer(params=wrn.params.values(),
                                          method=self.opt['wrn_optim_method'],
                                          lr=self.opt['wrn_lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.opt['wrn_lr_patience'], verbose=True)

        epoch = 0
        if resume is not None:
            state_dict = torch.load(os.path.join(resume, 'model.pt7'))
            epoch = state_dict['epoch']
            params_tensors, stats = state_dict['params'], state_dict['stats']
            for k, v in wrn.params.iteritems():
                v.data.copy_(params_tensors[k])
            optimizer.load_state_dict(state_dict['optimizer'])

        try:
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                start_time = datetime.now()
                wrn.train(mode=True)
                wrn, train_mean_acc, train_mean_losses = \
                    self.__doepoch__(train_loader,
                                     wrn, optimizer, isTraining=True)
                time_elapsed = datetime.now() - start_time
                print ("train epoch: %d, train loss: %f, train acc: %f. time: %02ds:%02dms" %
                       (epoch, train_mean_losses, train_mean_acc,
                       time_elapsed.seconds, time_elapsed.microseconds / 1000))

                # update optimizer
                #if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                #    lr = optimizer.param_groups[0]['lr']
                #    optimizer = self.create_optimizer(params=wrn.params.values(),
                #                                      method=self.opt['wrn_optim_method'],
                #                                      lr=lr * self.opt['wrn_lr_decay_ratio'])
                scheduler.step(train_mean_losses)

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    wrn.eval()
                    wrn, val_mean_acc, val_mean_losses = \
                                    self.__doepoch__(val_loader,
                                     wrn, optimizer = None, isTraining=False)
                    if val_mean_acc >= best_val_acc:
                        n_parameters = sum(p.numel() for p in wrn.params.values() + wrn.stats.values())
                        self.log({
                            "train_loss": float(train_mean_losses),
                            "train_acc": float(train_mean_acc),
                            "val_acc": float(val_mean_acc),
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, wrn.params, wrn.stats)
                        best_val_acc = val_mean_acc

        except KeyboardInterrupt:
            pass

        return wrn, best_val_acc


    def load(self, modelPath, fully_convolutional = False):

        wrn = WideResNet(self.opt['wrn_depth'], self.opt['wrn_width'], ninputs=self.opt['nchannels'],
                         num_groups=self.opt['wrn_groups'],
                         num_classes= None if fully_convolutional else self.opt['wrn_num_classes'],
                         dropout=self.opt['dropout'])

        print ('Loading Wide Residual Network...')
        state_dict = torch.load(os.path.join(modelPath, 'model.pt7'))
        params_tensors, stats = state_dict['params'], state_dict['stats']
        # copy the params tensors
        for k, v in wrn.params.items():
            v.data.copy_(params_tensors[k])
        # copy the stats tensors
        wrn.stats = stats
        #for k, v in wrn.stats.items():
        #    v = stats[k]

        print ('Wide Residual Network parameters...')
        print ('\nParameters:')
        kmax = max(len(key) for key in wrn.params.keys())
        for i, (key, v) in enumerate(wrn.params.items()):
            print (str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data))
        print ('\nAdditional buffers:')
        kmax = max(len(key) for key in wrn.stats.keys())
        for i, (key, v) in enumerate(wrn.stats.items()):
            print (str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v))

        n_parameters = sum(p.numel() for p in list(wrn.params.values()) + list(wrn.stats.values()))
        print ('\nTotal number of parameters: %d' % n_parameters)

        return wrn, None, None

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
                        stats=stats,
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['wrn_save'], 'model.pt7'), 'w'))
        z = self.opt.copy()
        z.update(dictParams)
        logname = os.path.join(self.opt['wrn_save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print (z)

    def forward(self, data_loader, f, params, stats):
        params, stats, mean_acc, mean_losses = \
            self.__doepoch__(self, data_loader,
                             f, params, stats,
                             optimizer = None, isTraining=False)
        return params, stats, mean_acc, mean_losses

    class Factory:
        def create(self,opt): return WideResidualNetwork(opt)



'''
class WideResidualNetwork(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader,
                    f, params, stats,
                    optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            train_preds = data_parallel(f, inputs, params, stats, isTraining,
                                        np.arange(self.opt['wrn_ngpu']))
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
        return params, stats, all_acc, all_losses

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        # create network
        f, params, stats = resnet(self.opt['wrn_depth'],
                                  self.opt['wrn_width'],
                                  self.opt['wrn_num_classes'],
                                  is_full_wrn=self.opt['wrn_full'])

        # create optimizer
        optimizer = self.create_optimizer(params=params.values(),
                                          method=self.opt['wrn_optim_method'],
                                          lr = self.opt['wrn_lr'])

        epoch = 0
        if resume is not None:
            state_dict = torch.load(resume)
            epoch = state_dict['epoch']
            params_tensors, stats = state_dict['params'], state_dict['stats']
            for k, v in params.iteritems():
                v.data.copy_(params_tensors[k])
            optimizer.load_state_dict(state_dict['optimizer'])

        try:
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                time_ini_epoch = datetime.datetime.now()
                params, stats, train_all_acc, train_all_losses = \
                                    self.__doepoch__(train_loader,
                                    f, params, stats,
                                    optimizer, isTraining=True)
                time_end_epoch = datetime.datetime.now()
                print ("train epoch: %d, train loss: %f, train acc: %f. time: %f s." %
                       (epoch, np.mean(train_all_losses), np.mean(train_all_acc),
                        int((time_end_epoch-time_ini_epoch).total_seconds())))

                # update optimizer
                if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer = self.create_optimizer(params=params.values(),
                                                      method=self.opt['wrn_optim_method'],
                                                      lr=lr * self.opt['wrn_lr_decay_ratio'])

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    params, stats, val_all_acc, val_all_losses = \
                                    self.__doepoch__(val_loader,
                                     f, params, stats,
                                     optimizer = None, isTraining=False)
                    val_acc_epoch = np.mean(val_all_acc)
                    val_losses_epoch = np.mean(val_all_losses)
                    if val_acc_epoch >= best_val_acc:
                        n_parameters = sum(p.numel() for p in params.values() + stats.values())
                        self.log({
                            "train_loss": float(np.mean(train_all_losses)),
                            "train_acc": float(np.mean(train_all_acc)),
                            "test_acc": val_acc_epoch,
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, params, stats)
                        best_val_acc = val_acc_epoch

        except KeyboardInterrupt:
            pass

        return f, params, stats, best_val_acc


    def load(self, modelPath, fully_convolutional = False):

        f, params, stats = resnet(self.opt['wrn_depth'],
                                  self.opt['wrn_width'],
                                  self.opt['wrn_num_classes'],
                                  is_full_wrn=self.opt['wrn_full'],
                                  is_fully_convolutional=fully_convolutional)

        print ('Loading Wide Residual Network...')
        state_dict = torch.load(os.path.join(modelPath, 'model.pt7'))
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in params.iteritems():
            v.data.copy_(params_tensors[k])

        print ('Wide Residual Network parameters...')
        print '\nParameters:'
        kmax = max(len(key) for key in params.keys())
        for i, (key, v) in enumerate(params.items()):
            print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data)
        print '\nAdditional buffers:'
        kmax = max(len(key) for key in stats.keys())
        for i, (key, v) in enumerate(stats.items()):
            print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v)

        n_parameters = sum(p.numel() for p in params.values() + stats.values())
        print '\nTotal number of parameters:', n_parameters

        return f, params, stats

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
                        stats=stats,
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['wrn_save'], 'model.pt7'), 'w'))
        z = self.opt.copy()
        z.update(dictParams)
        logname = os.path.join(self.opt['wrn_save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print z

    def forward(self, data_loader, f, params, stats):
        params, stats, data_all_acc, data_all_losses = \
            self.__doepoch__(self, data_loader,
                             f, params, stats,
                             optimizer = None, isTraining=False)
        return params, stats, data_all_acc, data_all_losses

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
        def create(self,opt): return WideResidualNetwork(opt)
'''


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
        print ('\nTotal number of parameters: %d' % n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.iteritems()},
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
        print ('\nTotal number of parameters: %d' % n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.iteritems()},
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
        print ('\nTotal number of parameters: %d' % n_parameters)
        return inception

    def log(self,dictParams, optimizer, params, stats):
        torch.save(dict(state_dict={k: v.data for k, v in params.iteritems()},
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

# Specialized Class. Wide Residual Networks. Preloaded imagenet.
class WideResidualNetworkv2(ConvCNN_Base):

    def __init__(self, opt):
        ConvCNN_Base.__init__(self, opt)
        self.opt = opt

    def __doepoch__(self, data_loader,
                    f, optimizer, isTraining = False):

        all_acc = []
        all_losses = []
        for batch_idx, (data, label) in enumerate(data_loader):
            if self.opt['cuda']:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data)
            targets = Variable(label)

            train_preds = f(inputs)
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
        return f, all_acc, all_losses

    def train(self, train_loader, val_loader, resume = None):

        best_validation_loss = sys.float_info.max
        best_val_acc = 0

        # create network
        wrn = WideResNetImageNet(useCuda=True, num_groups = 3, num_classes=self.opt['wrn_num_classes'])

        # create optimizer
        optimizer = self.create_optimizer(params=wrn.params.values(),
                                          method=self.opt['wrn_optim_method'],
                                          lr = self.opt['wrn_lr'])

        epoch = 0
        if resume is not None:
            state_dict = torch.load(resume)
            wrn.params = state_dict['state_dict']
            epoch = state_dict['epoch']
            optimizer.load_state_dict(state_dict['optimizer'])
        try:
            while epoch < self.opt['wrn_epochs']:
                epoch += 1
                time_ini_epoch = datetime.datetime.now()
                wrn, train_all_acc, train_all_losses = \
                                    self.__doepoch__(train_loader,
                                    wrn, optimizer, isTraining=True)
                time_end_epoch = datetime.datetime.now()
                print ("train epoch: %d, train loss: %f, train acc: %f. time: %f s." %
                       (epoch, np.mean(train_all_losses), np.mean(train_all_acc),
                        int((time_end_epoch-time_ini_epoch).total_seconds())))

                # update optimizer
                if epoch % self.opt['wrn_epochs_update_optimizer'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    optimizer = self.create_optimizer(params=wrn.params.values(),
                                                      method=self.opt['wrn_optim_method'],
                                                      lr=lr * self.opt['wrn_lr_decay_ratio'])

                # do validation
                if epoch % self.opt['wrn_val_freq'] == 0:
                    wrn, val_all_acc, val_all_losses = \
                                    self.__doepoch__(val_loader,
                                     wrn, optimizer = None, isTraining=False)
                    val_acc_epoch = np.mean(val_all_acc)
                    val_losses_epoch = np.mean(val_all_losses)
                    if val_acc_epoch >= best_val_acc:
                        n_parameters = sum(p.numel() for p in wrn.params.values())
                        self.log({
                            "train_loss": float(np.mean(train_all_losses)),
                            "train_acc": float(np.mean(train_all_acc)),
                            "test_acc": val_acc_epoch,
                            "epoch": epoch,
                            "num_classes": self.opt['wrn_num_classes'],
                            "n_parameters": n_parameters,
                        }, optimizer, wrn.params)
                        best_val_acc = val_acc_epoch

        except KeyboardInterrupt:
            pass

        return wrn, best_val_acc

    def load(self, modelPath, fully_convolutional = False):
        wrn = WideResNetImageNet(useCuda=True, num_groups = 0)
        state_dict = torch.load(os.path.join(modelPath, 'model.pt7'))
        wrn.params = state_dict['state_dict']
        n_parameters = sum(p.numel() for p in wrn.parameters())
        print ('\nTotal number of parameters: %d' % n_parameters)
        return wrn, None, None

    def log(self,dictParams, optimizer, params):
        torch.save(dict(state_dict=params,
                        optimizer=optimizer.state_dict(),
                        epoch=dictParams['epoch']),
                   open(os.path.join(self.opt['wrn_save'], 'model.pt7'), 'w'))
        z = self.opt.copy()
        z.update(dictParams)
        logname = os.path.join(self.opt['wrn_save'], 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print (z)

    def forward(self, data_loader, f, params, stats):
        params, stats, data_all_acc, data_all_losses = \
            self.__doepoch__(self, data_loader,
                             f, params, stats,
                             optimizer = None, isTraining=False)
        return params, stats, data_all_acc, data_all_losses

    class Factory:
        def create(self,opt): return WideResidualNetworkv2(opt)


######################################################################
######################################################################
######################################################################





# Generate all name of the factorization classes
def ConvCNN_NameGen():
    types = ConvCNN_Base.__subclasses__()
    for type in types:
        yield type.__name__










