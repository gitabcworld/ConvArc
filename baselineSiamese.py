import os
import sys

# Add all the python paths needed to execute when using Python 3.6
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models/wrn"))

import time
import numpy as np
from datetime import datetime, timedelta
from logger import Logger
import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve
import shutil
from tqdm import tqdm

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from option import Options, tranform_options

# Omniglot dataset
from omniglotDataLoader import omniglotDataLoader
from dataset.omniglot import Omniglot, OmniglotPairs

# Mini-imagenet dataset
from miniimagenetDataLoader import miniImagenetDataLoader
from dataset.mini_imagenet import MiniImagenet, MiniImagenetPairs

# Banknote dataset
from banknoteDataLoader import banknoteDataLoader
from dataset.banknote_pytorch import FullBanknote, FullBanknotePairs

# FCN
from models.conv_cnn import ConvCNNFactory

import multiprocessing

import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

def compute_accuracy_roc(labels, predictions):
    '''Compute ROC accuracy with a range of thresholds on distances.'''
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    max_acc = max(0.5*(tpr+tnr))
    return max_acc, thresholds

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Label: 0 if genuine, 1 if imposter
    """
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def do_epoch(epoch, repetitions, opt, data_loader, fcn, logger, optimizer=None):
    
    acc_epoch = []
    loss_epoch = []
    n_repetitions = 0
    while n_repetitions < repetitions:
        acc_batch = []
        loss_batch = []

        # generate new pairs
        data_loader.dataset.generate_pairs(opt.batchSize)

        for batch_idx, (data, label) in enumerate(data_loader):
            if opt.cuda:
                data = data.cuda()
                label = label.cuda()
            if optimizer:
                inputs = Variable(data, requires_grad=True)
            else:
                inputs = Variable(data, requires_grad=False)
            targets = Variable(label)

            feats1 = fcn.forward_features(inputs[:, 0, :, :, :])
            feats2 = fcn.forward_features(inputs[:, 1, :, :, :])

            #feats1 = fcn.features(inputs[:, 0, :, :, :])
            #feats2 = fcn.features(inputs[:, 1, :, :, :])
            feats1 = feats1 / (feats1.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats1)
            feats2 = feats2 / (feats2.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats2)
            logsoft_feats2 = torch.nn.LogSoftmax(dim=1)(fcn.forward_classifier(feats2))

            # LOSS 1 - Constrastive loss
            loss_fn_1 = ContrastiveLoss()
            if opt.cuda:
                loss_fn_1 = loss_fn_1.cuda()
            #ContrastiveLoss: Label: 0 if genuine, 1 if imposter -> ((targets-1).abs())
            loss1 = loss_fn_1(feats1, feats2, ((targets-1).abs()).float())
            
            loss_fn_2 = torch.nn.NLLLoss()
            if opt.cuda:
                loss_fn_2 = loss_fn_2.cuda()
            # LOSS 2 - Involve the classifier
            loss2 = loss_fn_2(logsoft_feats2, targets.long())
            
            # Total loss
            loss = loss1 + loss2

            loss_batch.append(loss.item())
            
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calculate distances. Similar samples should be close, dissimilar should be large.
            #dists = torch.sqrt(torch.sum((feats1 - feats2) ** 2, 1)) # euclidean distance
            #acc, thresholds = compute_accuracy_roc(targets, predictions=dists)
            probs = torch.exp(logsoft_feats2)
            max_index = probs.max(dim = 1)[1]
            acc = (max_index == targets.long()).sum().float()/len(targets)
            acc_batch.append(acc.item())

        acc_epoch.append(np.mean(acc_batch))
        loss_epoch.append(np.mean(loss_batch))
        n_repetitions += 1

    
    acc_epoch = np.mean(acc_epoch)
    loss_epoch = np.mean(loss_epoch)

    return acc_epoch, loss_epoch

def do_epoch_classification(epoch, repetitions, opt, data_loader, fcn, logger):

    acc_epoch = []
    n_repetitions = 0
    while n_repetitions < repetitions:
        acc_batch = []

        for batch_idx, (data, label) in enumerate(tqdm(data_loader)):
            if opt.cuda:
                data = data.cuda()
                label = label.cuda()
            inputs = Variable(data, requires_grad=False)
            targets = Variable(label, requires_grad=False)

            feats = fcn.forward_features(inputs)
            feats = feats / (feats.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats)
            logsoft_feats = torch.nn.LogSoftmax(dim=1)(fcn.forward_classifier(feats))
            probs = torch.exp(logsoft_feats)
            max_index = probs.max(dim = 1)[1]
            acc = (max_index == targets.long()).sum().float()/len(targets)
            acc_batch.append(acc.item())

        acc_epoch.append(np.mean(acc_batch))
        n_repetitions += 1

    acc_epoch = np.mean(acc_epoch)
    return acc_epoch

def train(index = 12):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    # use cuda?
    opt.cuda = torch.cuda.is_available()

    cudnn.benchmark = True # set True to speedup

    # Load mean/std if exists
    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(opt.save, 'mean.npy')):
        train_mean = np.load(os.path.join(opt.save, 'mean.npy'))
        train_std = np.load(os.path.join(opt.save, 'std.npy'))
    
    # Load FCN
    # Convert the opt params to dict.
    optDict = dict([(key, value) for key, value in opt._get_kwargs()])
    fcn = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
    if opt.wrn_load and os.path.exists(opt.wrn_load):
        if torch.cuda.is_available():
            fcn.load_state_dict(torch.load(opt.wrn_load))
        else:
            fcn.load_state_dict(torch.load(opt.wrn_load, map_location=torch.device('cpu')))
    if opt.cuda:
        fcn.cuda()

    # Load Dataset
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenetPairs, opt=opt, fcn=fcn)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=OmniglotPairs, opt=opt, fcn=fcn,train_mean=train_mean,
                                        train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknotePairs, opt=opt, fcn=fcn, train_mean=train_mean,
                                        train_std=train_std)
    else:
        pass

    # Use the same seed to split the train - val - test
    if os.path.exists(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy')):
        rnd_seed = np.load(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy'))
    else:    
        rnd_seed = np.random.randint(0, 100000)
        np.save(os.path.join(opt.save, 'dataloader_rnd_seed_arc.npy'), rnd_seed)

    # Get the DataLoaders from train - val - test
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed)

    train_mean = dataLoader.train_mean
    train_std = dataLoader.train_std
    if not os.path.exists(os.path.join(opt.save, 'mean.npy')):
        np.save(os.path.join(opt.save, 'mean.npy'), train_mean)
        np.save(os.path.join(opt.save, 'std.npy'), train_std)

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        #opt.name = "train_{}_{}_{}_{}_{}_wrn".format(str_model_fn, opt.numGlimpses, opt.glimpseSize, opt.numStates,
        opt.name = "{}_{}_{}_{}_{}_{}_wrn".format(opt.naive_full_type,
                                        "fcn" if opt.apply_wrn else "no_fcn",
                                        opt.arc_numGlimpses,
                                        opt.arc_glimpseSize, opt.arc_numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("[{}]. Will start training {} with parameters:\n{}\n\n".format(multiprocessing.current_process().name,
                                                                         opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join(opt.save, opt.name)
    if not os.path.isdir(models_path):
	    os.makedirs(models_path)
    else:
        shutil.rmtree(models_path)

    # create logger
    logger = Logger(models_path)

    # optimizer
    optimizer = torch.optim.Adam(params=fcn.parameters(), lr=opt.arc_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=opt.arc_lr_patience, verbose=True)

    # load preexisting optimizer values if exists
    if os.path.exists(opt.arc_optimizer_path):
        if torch.cuda.is_available():
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path))
        else:
            optimizer.load_state_dict(torch.load(opt.arc_optimizer_path, map_location=torch.device('cpu')))

    ###################################
    ## TRAINING ARC/CONVARC
    ###################################
    best_validation_loss = sys.float_info.max
    best_accuracy = 0.0
    saving_threshold = 1.02
    epoch = 0
    if opt.arc_resume == True or opt.arc_load is None:

        try:
            while epoch < opt.train_num_batches:
                epoch += 1
                fcn.train() # Set to train the network
                start_time = datetime.now()
                train_acc_epoch, train_loss_epoch = do_epoch(epoch=epoch, repetitions=1, opt=opt, data_loader = train_loader, fcn=fcn, 
                                                                logger=logger, optimizer=optimizer)
                time_elapsed = datetime.now() - start_time
                print ("[train]", "epoch: ", epoch, ", loss: ", train_loss_epoch, ", accuracy: ", train_acc_epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
                logger.log_value('train_acc', train_acc_epoch)
                logger.log_value('train_loss', train_loss_epoch)

                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)
                if epoch % opt.val_freq == 0:
                    fcn.eval() # set to test the network
                    start_time = datetime.now()
                    val_acc_epoch, val_loss_epoch = do_epoch(epoch=epoch, repetitions=opt.val_num_batches, opt=opt, data_loader = val_loader, fcn=fcn, 
                                                            logger=logger, optimizer=None)
                    time_elapsed = datetime.now() - start_time
                    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[VAL]", "epoch: ", epoch, ", loss: ", val_loss_epoch \
                            , ", accuracy: ", val_acc_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                    logger.log_value('val_acc', val_acc_epoch)
                    logger.log_value('val_loss', val_loss_epoch)

                    is_model_saved = False
                    #if best_validation_loss > (saving_threshold * val_loss_epoch):
                    if best_accuracy < (saving_threshold * val_acc_epoch):
                        print("[{}] Significantly improved validation loss from {} --> {}. accuracy from {} --> {}. Saving...".format(
                            multiprocessing.current_process().name, best_validation_loss, val_loss_epoch, best_accuracy, val_acc_epoch))
                        # save classifier
                        torch.save(fcn.state_dict(),opt.wrn_save)
                        # Save optimizer
                        torch.save(optimizer.state_dict(), opt.arc_optimizer_path)
                        # Acc-loss values
                        best_validation_loss = val_loss_epoch
                        best_accuracy = val_acc_epoch
                        is_model_saved = True

                    if is_model_saved:
                        fcn.eval() # set to test the network
                        start_time = datetime.now()
                        test_acc_epoch, _ = do_epoch(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, 
                                                        logger=logger, optimizer=None)
                        time_elapsed = datetime.now() - start_time
                        print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[TEST]", "epoch: ", epoch, ", accuracy: ", test_acc_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                        logger.log_value('test_acc', test_acc_epoch)

                logger.step()

            print ("[%s] ... training done" % multiprocessing.current_process().name)
            print ("[%s], best validation accuracy: %.2f, best validation loss: %.5f" % (
                multiprocessing.current_process().name, best_accuracy, best_validation_loss))
            print ("[%s] ... exiting training regime " % multiprocessing.current_process().name)

        except KeyboardInterrupt:
            pass
    ###################################

    # TODO: LOAD THE BEST MODEL
    fcn.load_state_dict(torch.load(opt.wrn_load))
    fcn.eval() # set to test the network

    # Set the num val/test repetitions to 1
    opt.val_num_batches = 1
    opt.test_num_batches = 1

    # Load Dataset
    opt.setType='set1'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=fcn)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot, opt=opt, fcn=fcn,train_mean=train_mean,
                                        train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknote, opt=opt, fcn=fcn, train_mean=train_mean,
                                        train_std=train_std)
    else:
        pass
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed, dataPartition = [None,None,'train+val+test'])
    print ('[%s] ... Testing Set1' % multiprocessing.current_process().name)
    start_time = datetime.now()
    test_acc_epoch = do_epoch_classification(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, logger=logger)
    time_elapsed = datetime.now() - start_time
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
        "[TEST]", "epoch: ", epoch, ", accuracy: ", test_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)
    #'''

    ## Get the set2 and try
    print ('[%s] ... Loading Set2' % multiprocessing.current_process().name)
    opt.setType='set2'
    if opt.datasetName == 'miniImagenet':
        dataLoader = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=None)
    elif opt.datasetName == 'omniglot':
        dataLoader = omniglotDataLoader(type=Omniglot, opt=opt, fcn=None,train_mean=None,
                                        train_std=None)
    elif opt.datasetName == 'banknote':
        dataLoader = banknoteDataLoader(type=FullBanknote, opt=opt, fcn=None, train_mean=None,
                                        train_std=None)
    else:
        pass
    train_loader, val_loader, test_loader = dataLoader.get(rnd_seed=rnd_seed, dataPartition = [None,None,'train+val+test'])
    print ('[%s] ... Testing Set2' % multiprocessing.current_process().name)
    start_time = datetime.now()
    test_acc_epoch = do_epoch_classification(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, logger=logger)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
        "[TEST]", "epoch: ", epoch, ", accuracy: ", test_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)

def main():
    train()

if __name__ == "__main__":
    main()