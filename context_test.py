import torch
import numpy as np
from datetime import datetime
from models.conv_cnn import ConvCNNFactory
import multiprocessing

def context_test(epoch, epoch_fn, opt, test_loader, discriminator, context_fn, logger, fcn, coAttn):

    # LOAD AGAIN THE FCN AND ARC models. Freezing the weights.
    print ("[%s] ... loading last validation model" % multiprocessing.current_process().name)
    discriminator.load_state_dict(torch.load(opt.arc_save))
    # freeze the weights from the ARC.
    for param in discriminator.parameters():
        param.requires_grad = False
    discriminator.eval()
    if opt.cuda:
        discriminator.cuda()

    # freeze the weights from the fcn and set it to eval.
    if opt.apply_wrn:
        for param in fcn.parameters():
            param.requires_grad = False
        fcn.eval()
        if opt.cuda:
            fcn.cuda()

    # set all gradient to True
    if opt.use_coAttn:
        for param in coAttn.parameters():
            param.requires_grad = False
        coAttn.eval()
        if opt.cuda:
            coAttn.cuda()

    # Load the context model
    context_fn.load_state_dict(torch.load(opt.naive_full_save_path))
    for param in context_fn.parameters():
        param.requires_grad = False
    context_fn.eval()
    if opt.cuda:
        context_fn.cuda()

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
                                            fcn=fcn, coAttn=coAttn)
        else:
            test_acc, _ = epoch_fn(opt=opt, discriminator=discriminator,
                                            data_loader=test_loader,
                                            model_fn=context_fn, coAttn=coAttn)
        test_acc_epoch.append(np.mean(test_acc))

    time_elapsed = datetime.now() - start_time
    test_acc_epoch = np.mean(test_acc_epoch)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                            "epoch: ", epoch, ", test ARC accuracy: ", test_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    logger.log_value('context_test_acc', test_acc_epoch)
    return test_acc_epoch