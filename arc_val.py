import sys
import numpy as np
from datetime import datetime
import multiprocessing

# Set global values so are being modified.
best_validation_loss = sys.float_info.max
best_accuracy = 0.0
saving_threshold = 1.02

def arc_val(epoch, epoch_fn, opt, val_loader, discriminator, logger,
                convCNN = None, optimizer=None, loss_fn=None, fcn=None):

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
                                            fcn=fcn)
        else:
            val_acc, val_loss = epoch_fn(opt=opt, loss_fn=loss_fn,
                                            discriminator=discriminator,
                                            data_loader=val_loader)
        val_acc_epoch.append(np.mean(val_acc))
        val_loss_epoch.append(np.mean(val_loss))
    time_elapsed = datetime.now() - start_time
    val_acc_epoch = np.mean(val_acc_epoch)
    val_loss_epoch = np.mean(val_loss_epoch)
    print "====" * 20, "\n", "[" + multiprocessing.current_process().name + "]" + \
                             "epoch: ", epoch, ", validation loss: ", val_loss_epoch \
        , ", validation accuracy: ", val_acc_epoch, ", time: ", \
        time_elapsed.seconds, "s:", time_elapsed.microseconds / 1000, "ms\n", "====" * 20
    logger.log_value('arc_val_loss', val_loss_epoch)
    logger.log_value('arc_val_acc', val_acc_epoch)

    is_model_saved = False
    #if best_validation_loss > (saving_threshold * val_loss_epoch):
    if best_accuracy < (saving_threshold * val_acc_epoch):
        print("[{}] Significantly improved validation loss from {} --> {}. accuracy from {} --> {}. Saving...".format(
            multiprocessing.current_process().name, best_validation_loss, val_loss_epoch, best_accuracy, val_acc_epoch))
        if opt.apply_wrn:
            # Save the fully convolutional network
            n_parameters = sum(p.numel() for p in fcn.params.values() + fcn.stats.values())
            convCNN.log({
                "val_acc": float(val_acc_epoch),
                "epoch": epoch,
                "n_parameters": n_parameters,
            }, optimizer, fcn.params, fcn.stats)
        # Save the ARC discriminator
        discriminator.save_to_file(opt.arc_save)
        best_validation_loss = val_loss_epoch
        best_accuracy = val_acc_epoch
        is_model_saved = True

    return val_acc_epoch, val_loss_epoch, is_model_saved


