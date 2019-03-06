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

import ntpath
import multiprocessing

import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import ranking

import pdb

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

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

    all_probs = []
    all_labels = []
    #acc_epoch = []
    auc_epoch = []
    auc_std_epoch = []
    loss_epoch = []
    n_repetitions = 0
    while n_repetitions < repetitions:
        #acc_batch = []
        auc_batch = []
        loss_batch = []

        # generate new pairs
        data_loader.dataset.generate_pairs(opt.batchSize)
        # set new paths to tmp data
        data_loader.dataset.set_path_tmp_epoch_iteration(epoch,n_repetitions)

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
            #acc_batch.append(acc.item())
            #auc = ranking.roc_auc_score(targets.long().data.cpu().numpy(), probs.cpu().data.numpy()[:,1], average=None, sample_weight=None)            
            #auc_batch.append(auc)
            all_probs.append(probs.cpu().data.numpy()[:,1])
            all_labels.append(targets.long().data.cpu().numpy())

        auc = ranking.roc_auc_score([item for sublist in all_labels for item in sublist],
                                      [item for sublist in all_probs for item in sublist], average=None, sample_weight=None)
        auc_epoch.append(auc)
        #acc_epoch.append(np.mean(acc_batch))
        #auc_epoch.append(np.mean(auc_batch))
        #auc_std_epoch.append(np.std(auc_batch))
        loss_epoch.append(np.mean(loss_batch))
        # remove data repetition
        data_loader.dataset.remove_path_tmp_epoch(epoch,n_repetitions)
        # next repetition
        n_repetitions += 1

    # remove data epoch
    data_loader.dataset.remove_path_tmp_epoch(epoch)

    #acc_epoch = np.mean(acc_epoch)
    #auc_epoch = np.mean(auc_epoch)
    #auc_std_epoch = np.mean(auc_std_epoch)
    auc_std_epoch = np.std(auc_epoch)
    auc_epoch = np.mean(auc_epoch)
    loss_epoch = np.mean(loss_epoch)

    #return acc_epoch, loss_epoch
    return auc_epoch, auc_std_epoch, loss_epoch


def do_epoch_classification(epoch, repetitions, opt, data_loader, fcn, logger):

    auc_epoch = []
    auc_std_epoch = []
    n_repetitions = 0
    all_probs = []
    all_labels = []
    while n_repetitions < repetitions:
        #acc_batch = []
        auc_batch = []

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
            #acc = (max_index == targets.long()).sum().float()/len(targets)
            #acc_batch.append(acc.item())
            #auc = ranking.roc_auc_score(targets.long().data.cpu().numpy(), probs.cpu().data.numpy()[:,1], average=None, sample_weight=None)
            #auc_batch.append(auc)
            all_probs.append(probs.cpu().data.numpy()[:,1])
            all_labels.append(targets.long().data.cpu().numpy())

        #acc_epoch.append(np.mean(acc_batch))
        #pdb.set_trace()
        auc = ranking.roc_auc_score([item for sublist in all_labels for item in sublist], 
                                      [item for sublist in all_probs for item in sublist], average=None, sample_weight=None)
        auc_epoch.append(auc)
        #auc_std_epoch.append(np.mean(auc_batch))
        n_repetitions += 1

    #acc_epoch = np.mean(acc_epoch)
    auc_std_epoch = np.std(auc_epoch)
    auc_epoch = np.mean(auc_epoch)
    #return acc_epoch
    return auc_epoch, auc_std_epoch, None


def data_generation(opt):

    # use cuda?
    opt.cuda = torch.cuda.is_available()
    cudnn.benchmark = True # set True to speedup

    # Load mean/std if exists
    train_mean = None
    train_std = None
    if os.path.exists(os.path.join(opt.save, 'mean.npy')):
        train_mean = np.load(os.path.join(opt.save, 'mean.npy'))
        train_std = np.load(os.path.join(opt.save, 'std.npy'))

    # Load Dataset
    opt.setType='set1'
    dataLoader = banknoteDataLoader(type=FullBanknotePairs, opt=opt, fcn=None, train_mean=train_mean,
                                    train_std=train_std)

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
    
    epoch = 0
    try:
        while epoch < opt.train_num_batches:

            # wait to check if it is neede more data
            lst_epochs = train_loader.dataset.getFolderEpochList()
            if len(lst_epochs) > 50:
                time.sleep(10)
            
            # In case there is more than one generator.
            # get the last folder epoch executed and update the epoch accordingly
            if len(lst_epochs)>0:
                epoch = np.array([path_leaf(str).split('_')[-1] for str in lst_epochs if 'train' in str]).astype(np.int).max()

            epoch += 1
            ## set information of the epoch in the dataloader
            repetitions = 1
            start_time = datetime.now()
            for repetition in range(repetitions):
                train_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                for batch_idx, (data, label) in enumerate(train_loader):
                    noop = 0
            time_elapsed = datetime.now() - start_time
            print ("[train]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

            if epoch % opt.val_freq == 0:

                repetitions = opt.val_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    val_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, label) in enumerate(val_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[val]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

                '''
                repetitions = opt.test_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    test_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, label) in enumerate(test_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[test]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
                '''

        print ("[%s] ... generating data done" % multiprocessing.current_process().name)

    except KeyboardInterrupt:
        pass
    ###################################



def server_processing(opt):

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

    # Remove memory
    del dataLoader

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

    print('Loading set 2 ...')
    opt.setType='set2'
    if opt.datasetName == 'miniImagenet':
        dataLoader2 = miniImagenetDataLoader(type=MiniImagenet, opt=opt, fcn=None)
    elif opt.datasetName == 'omniglot':
        dataLoader2 = omniglotDataLoader(type=Omniglot, opt=opt, fcn=None,train_mean=train_mean,
                                        train_std=train_std)
    elif opt.datasetName == 'banknote':
        dataLoader2 = banknoteDataLoader(type=FullBanknotePairs, opt=opt, fcn=None, train_mean=train_mean,
                                        train_std=train_std)
    else:
        pass
    _, _, test_loader2 = dataLoader2.get(rnd_seed=rnd_seed, dataPartition = [None,None,'train+val+test'])
    # Remove memory
    del dataLoader2


    ###################################
    ## TRAINING ARC/CONVARC
    ###################################
    best_validation_loss = sys.float_info.max
    best_auc = 0.0
    saving_threshold = 1.02
    epoch = 0
    if opt.arc_resume == True or opt.arc_load is None:

        try:
            while epoch < opt.train_num_batches:
                epoch += 1
                fcn.train() # Set to train the network
                start_time = datetime.now()
                train_auc_epoch, train_std_auc_epoch, train_loss_epoch = do_epoch(epoch=epoch, repetitions=1, opt=opt, data_loader = train_loader, fcn=fcn, 
                                                                logger=logger, optimizer=optimizer)
                time_elapsed = datetime.now() - start_time
                print ("[train]", "epoch: ", epoch, ", loss: ", train_loss_epoch, ", auc: ", train_auc_epoch, ",std_auc: ", train_std_auc_epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
                logger.log_value('train_auc', train_auc_epoch)
                logger.log_value('train_auc_std', train_std_auc_epoch)
                logger.log_value('train_loss', train_loss_epoch)

                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)
                if epoch % opt.val_freq == 0:
                    fcn.eval() # set to test the network
                    start_time = datetime.now()
                    val_auc_epoch, val_std_auc_epoch, val_loss_epoch = do_epoch(epoch=epoch, repetitions=opt.val_num_batches, opt=opt, data_loader = val_loader, fcn=fcn, 
                                                            logger=logger, optimizer=None)
                    time_elapsed = datetime.now() - start_time
                    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[VAL]", "epoch: ", epoch, ", loss: ", val_loss_epoch \
                            , ", auc: ", val_auc_epoch, ", auc std: ", val_std_auc_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                    logger.log_value('val_auc', val_auc_epoch)
                    logger.log_value('val_auc_std', val_std_auc_epoch)
                    logger.log_value('val_loss', val_loss_epoch)

                    is_model_saved = False
                    #if best_validation_loss > (saving_threshold * val_loss_epoch):
                    if best_auc < (saving_threshold * val_auc_epoch):
                        print("[{}] Significantly improved validation loss from {} --> {}. auc from {} --> {}. Saving...".format(
                            multiprocessing.current_process().name, best_validation_loss, val_loss_epoch, best_auc, val_auc_epoch))
                        # save classifier
                        torch.save(fcn.state_dict(),opt.wrn_save)
                        # Save optimizer
                        torch.save(optimizer.state_dict(), opt.arc_optimizer_path)
                        # Acc-loss values
                        best_validation_loss = val_loss_epoch
                        best_auc = val_auc_epoch
                        is_model_saved = True

                    if is_model_saved:
                        fcn.eval() # set to test the network
                        start_time = datetime.now()
                        test_loader.dataset.mode = 'generator_processor'
                        test_auc_epoch, test_std_auc_epoch, _ = do_epoch(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, 
                                                        logger=logger, optimizer=None)
                        time_elapsed = datetime.now() - start_time
                        print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[TEST] SET1. ", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                        logger.log_value('test_set1_auc', test_auc_epoch)
                        logger.log_value('test_set1_auc_std', test_std_auc_epoch)

                        start_time = datetime.now()
                        test_loader2.dataset.mode = 'generator_processor'
                        test_auc_epoch, test_std_auc_epoch, _ = do_epoch(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader2, fcn=fcn, 
                                                        logger=logger)
                        time_elapsed = datetime.now() - start_time
                        print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[TEST] SET2. ", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                        logger.log_value('test_set2_auc', test_auc_epoch)
                        logger.log_value('test_set2_auc_std', test_std_auc_epoch)

                # just in case there is a folder not removed, remove it
                train_loader.dataset.remove_path_tmp_epoch(epoch)
                val_loader.dataset.remove_path_tmp_epoch(epoch)
                test_loader.dataset.remove_path_tmp_epoch(epoch)

                logger.step()

            print ("[%s] ... training done" % multiprocessing.current_process().name)
            print ("[%s], best validation auc: %.2f, best validation loss: %.5f" % (
                multiprocessing.current_process().name, best_auc, best_validation_loss))
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

    # set the mode of the dataset to generator_processor
    # which generates and processes the images without saving them.
    opt.mode = 'generator_processor'

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
    test_auc_epoch, test_auc_std_epoch, _ = do_epoch_classification(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, logger=logger)
    time_elapsed = datetime.now() - start_time
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
        "[TEST]", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", auc_std: ", test_auc_std_epoch, ", time: ", \
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
    test_auc_epoch, test_auc_std_epoch, _ = do_epoch_classification(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, logger=logger)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
        "[TEST]", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", auc_std: ", test_auc_std_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    print ('[%s] ... FINISHED! ...' % multiprocessing.current_process().name)





def train(index = None):

    # change parameters
    opt = Options().parse()
    #opt = Options().parse() if opt is None else opt
    opt = tranform_options(index, opt)
    if opt.mode == 'generator':
        print('Starting generator...')
        data_generation(opt)
    elif opt.mode == 'generator_processor':
        print('Starting generator - processor no save images...')
        server_processing(opt)  
    else:
        print('Starting processor...')
        server_processing(opt)   

def main():
    train()

if __name__ == "__main__":
    main()
