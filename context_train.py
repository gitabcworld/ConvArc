import numpy as np
from datetime import datetime
import multiprocessing

def context_train(epoch, epoch_fn, opt, train_loader, discriminator, context_fn, logger,
              optimizer=None, loss_fn=None, fcn=None):

    start_time = datetime.now()

    # set all gradients to True and the model in evaluation format.
    for param in discriminator.parameters():
        param.requires_grad = False
    discriminator.eval()
    # set all gradients to True and the fcn in evaluation format.
    if opt.apply_wrn:
        for param in fcn.parameters():
            param.requires_grad = False
        fcn.eval()
    # set all gradients to true in the context model.
    for param in context_fn.parameters():
        param.requires_grad = True
    context_fn.train(mode=True)  # Set to train the naive/full-context model

    if opt.apply_wrn:
        train_acc_epoch, train_loss_epoch = epoch_fn(opt=opt, loss_fn=loss_fn,
                                                                discriminator=discriminator,
                                                                data_loader=train_loader,
                                                                model_fn=context_fn,
                                                                optimizer=optimizer, fcn=fcn)
    else:
        train_acc_epoch, train_loss_epoch = epoch_fn(opt=opt, loss_fn=loss_fn,
                                                                discriminator=discriminator,
                                                                data_loader=train_loader,
                                                                model_fn=context_fn,
                                                                optimizer=optimizer)

    time_elapsed = datetime.now() - start_time
    train_acc_epoch = np.mean(train_acc_epoch)
    train_loss_epoch = np.mean(train_loss_epoch)
    print ("[%s] epoch: %d, train loss: %f, train acc: %.2f, time: %02ds:%02dms" %
           (multiprocessing.current_process().name, epoch, np.round(train_loss_epoch, 6), np.round(train_acc_epoch, 6),
            time_elapsed.seconds, time_elapsed.microseconds / 1000))
    logger.log_value('context_train_loss', train_loss_epoch)
    logger.log_value('context_train_acc', train_acc_epoch)

    assert np.isnan(train_loss_epoch) == False, 'ERROR. Found NAN in context_train.'

    # Reduce learning rate when a metric has stopped improving
    logger.log_value('context_train_lr', [param_group['lr'] for param_group in optimizer.param_groups][0])
    return train_acc_epoch, train_loss_epoch