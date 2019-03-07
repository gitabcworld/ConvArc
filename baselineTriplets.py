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

# Banknote dataset
from banknoteDataLoader import banknoteDataLoader
from dataset.banknote_pytorch import FullBanknote, FullBanknoteTriplets

# FCN
from models.conv_cnn import ConvCNNFactory
import ntpath
import multiprocessing
import pdb
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import ranking

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def do_epoch(epoch, repetitions, opt, data_loader, fcn, logger, optimizer=None):
    
    acc_epoch = []
    loss_epoch = []
    all_probs = []
    all_labels = []
    auc_epoch = []
    n_repetitions = 0
    while n_repetitions < repetitions:
        acc_batch = []
        loss_batch = []

        data_loader.dataset.set_path_tmp_epoch_iteration(epoch,n_repetitions)

        for batch_idx, (data, info) in enumerate(data_loader):
            if opt.cuda:
                data = data.cuda()
            if optimizer:
                inputs = Variable(data, requires_grad=True)
            else:
                inputs = Variable(data, requires_grad=False)

            feats_p = fcn.forward_features(inputs[:, 0, :, :, :]) # positive
            feats_a = fcn.forward_features(inputs[:, 1, :, :, :]) # anchor
            feats_n = fcn.forward_features(inputs[:, 2, :, :, :]) # negative

            # E1, E2, E3 = model(anchor_img, pos_img, neg_img)
            # dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            # dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            # target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            # if args.cuda:
            #     target = target.cuda()
            # target = Variable(target)
            
            # #Calculate loss
            # loss_triplet = criterion(dist_E1_E2, dist_E1_E3, target)
            # loss_embedd = E1.norm(2) + E2.norm(2) + E3.norm(2)
            # loss = loss_triplet + 0.001*loss_embedd
            # total_loss += loss

            feats_p = feats_p / (feats_p.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats_p)
            feats_a = feats_a / (feats_a.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats_a)
            feats_n = feats_n / (feats_n.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(feats_n)

            # Do the classification for the positive and the negative
            logsoft_feats_p = torch.nn.LogSoftmax(dim=1)(fcn.forward_classifier(feats_p))
            logsoft_feats_n = torch.nn.LogSoftmax(dim=1)(fcn.forward_classifier(feats_n))

            #dists_p = torch.sqrt(torch.sum((feats_p - feats_a) ** 2, 1))  # euclidean distance
            #dists_n = torch.sqrt(torch.sum((feats_n - feats_a) ** 2, 1))  # euclidean distance
            dists_p = F.pairwise_distance(feats_a, feats_p, 2) # PairwiseDistance
            dists_n = F.pairwise_distance(feats_a, feats_n, 2) # PairwiseDistance

            # 1 means, dist_n should be larger than dist_p
            targets = torch.FloatTensor(len(dists_p)).fill_(1)
            targets = Variable(targets)
            if opt.cuda:
                targets = targets.cuda()

            # LOSS 1 - Constrastive loss
            margin = 0.2
            loss_fn_1 = torch.nn.MarginRankingLoss(margin=margin)
            if opt.cuda:
                loss_fn_1 = loss_fn_1.cuda()
            loss1 = loss_fn_1(dists_p, dists_n, targets)

            loss_fn_2 = torch.nn.NLLLoss()
            if opt.cuda:
                loss_fn_2 = loss_fn_2.cuda()
            # LOSS 2 - Involve the classifier
            targets_p = torch.FloatTensor(len(logsoft_feats_p)).fill_(1)
            targets_n = torch.FloatTensor(len(logsoft_feats_n)).fill_(0)
            targets = torch.stack((targets_p,targets_n)).view(-1)
            targets = Variable(targets)
            if opt.cuda:
                targets = targets.cuda()
            logsoft_feats = torch.stack((logsoft_feats_p,logsoft_feats_n)).view(-1,2)
            loss2 = loss_fn_2(logsoft_feats, targets.long())
            
            # Total loss
            loss = loss1 + loss2

            loss_batch.append(loss.item())
            
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #prediction = (dists_n - dists_p - margin).cpu().data
		    #prediction = prediction.view(prediction.numel())
		    #prediction = (prediction > 0).float()
		    #batch_acc = prediction.sum()*1.0/prediction.numel()

            # calculate distances. Similar samples should be close, dissimilar should be large.
            probs = torch.exp(logsoft_feats)
            max_index = probs.max(dim = 1)[1]
            acc = (max_index == targets.long()).sum().float()/len(targets)
            #acc_batch.append(acc.item())

            all_probs.append(probs.cpu().data.numpy()[:,1])
            all_labels.append(targets.long().data.cpu().numpy())

        auc = ranking.roc_auc_score([item for sublist in all_labels for item in sublist],
                                      [item for sublist in all_probs for item in sublist], average=None, sample_weight=None)
        auc_epoch.append(auc)
        #acc_epoch.append(np.mean(acc_batch))
        loss_epoch.append(np.mean(loss_batch))
        # remove data repetition
        data_loader.dataset.remove_path_tmp_epoch(epoch,n_repetitions)
        # next repetition
        n_repetitions += 1
    
    # remove data epoch
    data_loader.dataset.remove_path_tmp_epoch(epoch)

    auc_std_epoch = np.std(auc_epoch)
    auc_epoch = np.mean(auc_epoch)
    #acc_epoch = np.mean(acc_epoch)
    loss_epoch = np.mean(loss_epoch)

    #return acc_epoch, loss_epoch
    return auc_epoch, auc_std_epoch, loss_epoch


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
    dataLoader = banknoteDataLoader(type=FullBanknoteTriplets, opt=opt, fcn=None, train_mean=train_mean,
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
                for batch_idx, (data, info) in enumerate(train_loader):
                    noop = 0
            time_elapsed = datetime.now() - start_time
            print ("[train]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

            if epoch % opt.val_freq == 0:

                repetitions = opt.val_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    val_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, info) in enumerate(val_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[val]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

                repetitions = opt.test_num_batches
                start_time = datetime.now()
                for repetition in range(repetitions):
                    test_loader.dataset.set_path_tmp_epoch_iteration(epoch,repetition)
                    for batch_idx, (data, info) in enumerate(test_loader):
                        noop = 0
                time_elapsed = datetime.now() - start_time
                print ("[test]", "epoch: ", epoch, ", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)

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
    dataLoader = banknoteDataLoader(type=FullBanknoteTriplets, opt=opt, fcn=fcn, train_mean=train_mean,
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
    dataLoader2 = banknoteDataLoader(type=FullBanknoteTriplets, opt=opt, fcn=None, train_mean=train_mean,
                                        train_std=train_std)
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
                ## set information of the epoch in the dataloader
                fcn.train() # Set to train the network
                start_time = datetime.now()
                train_auc_epoch, train_auc_std_epoch, train_loss_epoch = do_epoch(epoch=epoch, repetitions=1, opt=opt, data_loader = train_loader, fcn=fcn, 
                                                                logger=logger, optimizer=optimizer)
                time_elapsed = datetime.now() - start_time
                print ("[train]", "epoch: ", epoch, ", loss: ", train_loss_epoch, ", auc: ", train_auc_epoch, ", auc_std: ", train_auc_std_epoch,", time: ", time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000)
                logger.log_value('train_auc', train_auc_epoch)
                logger.log_value('train_auc_std', train_auc_std_epoch)
                logger.log_value('train_loss', train_loss_epoch)

                # Reduce learning rate when a metric has stopped improving
                scheduler.step(train_loss_epoch)
                if epoch % opt.val_freq == 0:
                    fcn.eval() # set to test the network
                    start_time = datetime.now()
                    val_auc_epoch, val_auc_std_epoch, val_loss_epoch = do_epoch(epoch=epoch, repetitions=opt.val_num_batches, opt=opt, data_loader = val_loader, fcn=fcn, 
                                                            logger=logger, optimizer=None)
                    time_elapsed = datetime.now() - start_time
                    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[VAL]", "epoch: ", epoch, ", loss: ", val_loss_epoch \
                            , ", auc: ", val_auc_epoch, ", auc_std: ", val_auc_std_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                    logger.log_value('val_auc', val_auc_epoch)
                    logger.log_value('val_auc_std', val_auc_std_epoch)
                    logger.log_value('val_loss', val_loss_epoch)

                    is_model_saved = False
                    #if best_validation_loss > (saving_threshold * val_loss_epoch):
                    if best_auc < (saving_threshold * val_auc_epoch):
                        print("[{}] Significantly improved validation loss from {} --> {}. accuracy from {} --> {}. Saving...".format(
                            multiprocessing.current_process().name, best_validation_loss, val_loss_epoch, best_auc, best_auc))
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
                        test_auc_epoch, test_auc_std_epoch, _ = do_epoch(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader, fcn=fcn, 
                                                        logger=logger, optimizer=None)
                        time_elapsed = datetime.now() - start_time
                        print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[TEST] SET1", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", auc_std: ", test_auc_std_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                        logger.log_value('test_set1_auc', test_auc_epoch)
                        logger.log_value('test_set1_auc_std', test_auc_epoch)

                        start_time = datetime.now()
                        test_loader2.dataset.mode = 'generator_processor'
                        test_auc_epoch, test_auc_std_epoch, _ = do_epoch(epoch=epoch, repetitions=opt.test_num_batches, opt=opt, data_loader = test_loader2, fcn=fcn, 
                                                        logger=logger)
                        time_elapsed = datetime.now() - start_time
                        print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "[TEST] SET2. ", "epoch: ", epoch, ", auc: ", test_auc_epoch, ", auc_std: ", test_auc_std_epoch, ", time: ", \
                            time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
                        logger.log_value('test_set2_auc', test_auc_epoch)
                        logger.log_value('test_set2_auc_std', test_auc_std_epoch)

                # just in case there is a folder not removed, remove it
                train_loader.dataset.remove_path_tmp_epoch(epoch)
                val_loader.dataset.remove_path_tmp_epoch(epoch)
                test_loader.dataset.remove_path_tmp_epoch(epoch)

                logger.step()

            print ("[%s] ... training done" % multiprocessing.current_process().name)
            print ("[%s], best validation accuracy: %.2f, best validation loss: %.5f" % (
                multiprocessing.current_process().name, best_auc, best_validation_loss))
            print ("[%s] ... exiting training regime " % multiprocessing.current_process().name)

        except KeyboardInterrupt:
            pass
    ###################################

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
