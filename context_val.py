import sys
import numpy as np
from datetime import datetime
import torch
import multiprocessing
import torch

# Set global values so are being modified.
best_validation_loss = sys.float_info.max
best_accuracy = 0.0
saving_threshold = 1.02

def context_val(epoch, epoch_fn, opt, val_loader, discriminator, context_fn, logger, loss_fn=None, fcn=None, coAttn=None):

    global best_validation_loss, best_accuracy, saving_threshold

    # freeze the weights from the ARC and set it to eval.
    for param in discriminator.parameters():
        param.requires_grad = False
    discriminator.eval()
    # freeze the weights from the fcn and set it to eval.
    if opt.apply_wrn:
        for param in fcn.parameters():
            param.requires_grad = False
        fcn.eval()
    # set all gradient to True
    if opt.use_coAttn:
        for param in coAttn.parameters():
            param.requires_grad = False
        coAttn.eval()
    # set all gradients to true in the context model.
    for param in context_fn.parameters():
        param.requires_grad = False
    context_fn.eval()  # Set to train the naive/full-context model

    val_epoch = 0
    val_acc_epoch = []
    val_loss_epoch = []
    start_time = datetime.now()
    while val_epoch < opt.val_num_batches:
        val_epoch += 1
        if opt.apply_wrn:
            val_acc, val_loss = epoch_fn(opt=opt, loss_fn=loss_fn,
                                                    discriminator=discriminator,
                                                    data_loader=val_loader,
                                                    model_fn=context_fn,
                                                    fcn=fcn, coAttn=coAttn)
        else:
            val_acc, val_loss = epoch_fn(opt=opt, loss_fn=loss_fn,
                                                    discriminator=discriminator,
                                                    data_loader=val_loader,
                                                    model_fn=context_fn, coAttn=coAttn)

        val_acc_epoch.append(np.mean(val_acc))
        val_loss_epoch.append(np.mean(val_loss))
    time_elapsed = datetime.now() - start_time
    val_acc_epoch = np.mean(val_acc_epoch)
    val_loss_epoch = np.mean(val_loss_epoch)
    print ("====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                             "epoch: ", epoch, ", validation loss: ", val_loss_epoch \
        , ", validation accuracy: ", val_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20)
    logger.log_value('context_val_loss', val_loss_epoch)
    logger.log_value('context_val_acc', val_acc_epoch)

    is_model_saved = False
    if best_accuracy < (saving_threshold * val_acc_epoch):
        print("[{}] Significantly improved validation loss from {} --> {}. accuracy from {} --> {}. Saving...".format(
            multiprocessing.current_process().name, best_validation_loss, val_loss_epoch, best_accuracy, val_acc_epoch))
        # Save the context model
        torch.save(context_fn.state_dict(), opt.naive_full_save_path)
        # Acc - loss values
        best_validation_loss = val_loss_epoch
        best_accuracy = val_acc_epoch
        is_model_saved = True

    return val_acc_epoch, val_loss_epoch, is_model_saved
