import torch
import numpy as np
from datetime import datetime
from models.conv_cnn import ConvCNNFactory
import multiprocessing

def context_test(epoch, epoch_fn, opt, test_loader, discriminator, logger):

    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ("[%s] ... loading last validation model" % multiprocessing.current_process().name)
    discriminator.load_state_dict(torch.load(opt.arc_save))
    # freeze the weights from the ARC.
    for param in discriminator.parameters():
        param.requires_grad = False
    discriminator.eval()

    # Load FCN?
    if opt.apply_wrn:
        # Convert the opt params to dict.
        optDict = dict([(key, value) for key, value in opt._get_kwargs()])
        convCNN = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
        # Load the last saved fully convolutional model
        fcn, _, _ = convCNN.load(opt.wrn_save, fully_convolutional=True)
        for param in fcn.parameters():
            param.requires_grad = False
        fcn.eval()

    # Load the context model
    context_fn = torch.load(opt.naive_full_save_path)
    for param in context_fn.parameters():
        param.requires_grad = False
    context_fn.eval()

    # TEST of FCN and ARC models
    start_time = datetime.now()
    print ('[%s] ... testing' % multiprocessing.current_process().name)
    test_epoch = 0
    test_acc_epoch = []
    while test_epoch < opt.test_num_batches:
        test_epoch += 1
        if opt.apply_wrn:
            test_acc, _ = epoch_fn(opt=opt, discriminator=discriminator,
                                            data_loader=test_loader,
                                            model_fn=context_fn,
                                            fcn=fcn)
        else:
            test_acc, _ = epoch_fn(opt=opt, discriminator=discriminator,
                                            data_loader=test_loader,
                                            model_fn=context_fn)
        test_acc_epoch.append(np.mean(test_acc))

    time_elapsed = datetime.now() - start_time
    test_acc_epoch = np.mean(test_acc_epoch)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "epoch: ", epoch, ", test ARC accuracy: ", test_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    logger.log_value('context_test_acc', test_acc_epoch)
    return test_acc_epoch