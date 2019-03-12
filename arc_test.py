import torch
import numpy as np
from datetime import datetime
from models.conv_cnn import ConvCNNFactory
from models.coAttn import CoAttn
import multiprocessing
from sklearn.metrics import ranking
import pdb

def arc_test(epoch, epoch_fn, opt, test_loader, discriminator, logger):

    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ("[%s] ... loading last validation model" % multiprocessing.current_process().name)
    if not(discriminator is None):
        discriminator.load_state_dict(torch.load(opt.arc_save))
    
    if not(discriminator is None):
        # freeze the weights from the ARC.
        for param in discriminator.parameters():
            param.requires_grad = False
        discriminator.eval()
        if opt.cuda:
            discriminator.cuda()

    if opt.apply_wrn:
        # Convert the opt params to dict.
        optDict = dict([(key, value) for key, value in opt._get_kwargs()])
        fcn = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
        if torch.cuda.is_available():
            fcn.load_state_dict(torch.load(opt.wrn_load))
        else:
            fcn.load_state_dict(torch.load(opt.wrn_load, map_location=torch.device('cpu')))
        for param in fcn.parameters():
            param.requires_grad = False
        if opt.cuda:
            fcn.cuda()
        fcn.eval()

    # Load the Co-Attn module
    coAttn = None
    if opt.use_coAttn:
        coAttn = CoAttn(size = opt.coAttn_size, num_filters=opt.arc_nchannels, typeActivation = opt.coAttn_type, p = opt.coAttn_p)
        if torch.cuda.is_available():
            coAttn.load_state_dict(torch.load(opt.coattn_load))
        else:
            coAttn.load_state_dict(torch.load(opt.coattn_load, map_location=torch.device('cpu')))
        if opt.cuda:
            coAttn.cuda()

    # TEST of FCN and ARC models
    start_time = datetime.now()
    print ('[%s] ... testing' % multiprocessing.current_process().name)
    test_epoch = 0
    test_auc_epoch = []
    while test_epoch < opt.test_num_batches:

        test_loader.dataset.set_path_tmp_epoch_iteration(epoch=epoch,iteration=test_epoch)

        if opt.apply_wrn:
            test_auc, test_loss = epoch_fn(opt=opt, loss_fn=None,
                                               discriminator=discriminator,
                                               data_loader=test_loader,
                                               fcn=fcn, coAttn=coAttn )
        else:
            test_auc, test_loss = epoch_fn(opt=opt, loss_fn=None,
                                               discriminator=discriminator,
                                               data_loader=test_loader, coAttn=coAttn)
        
        if isinstance(test_auc, tuple):
             features = [item for sublist in test_auc[0] for item in sublist]
             labels = [item for sublist in test_auc[1] for item in sublist]
             test_auc = ranking.roc_auc_score(labels, features, average=None, sample_weight=None)
        
        test_auc_epoch.append(np.mean(test_auc))

        test_loader.dataset.remove_path_tmp_epoch(epoch=epoch,iteration=test_epoch)

        test_epoch += 1

    test_loader.dataset.remove_path_tmp_epoch(epoch=epoch)

    time_elapsed = datetime.now() - start_time
    #pdb.set_trace()
    test_auc_std_epoch = np.std(test_auc_epoch)
    test_auc_epoch = np.mean(test_auc_epoch)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" +\
                             "epoch: ", epoch, ", test ARC auc: ", test_auc_epoch, ", test ARC auc_std: ", test_auc_std_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    logger.log_value('arc_test_auc', test_auc_epoch)
    logger.log_value('arc_test_auc_std', test_auc_std_epoch)
    return test_auc_epoch, test_auc_std_epoch