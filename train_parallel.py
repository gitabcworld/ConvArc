import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from logger import Logger
import torch
import shutil

from models import models
from models.wrn.utils import data_parallel
from models.models import ArcBinaryClassifier

from torch.autograd import Variable
from omniglotBenchmarks import omniglotBenchMark
import torch.backends.cudnn as cudnn

from option import Options
from dataset.omniglot_pytorch import OmniglotOS
from dataset.omniglot_pytorch import OmniglotOSPairs

from models.conv_cnn import ConvCNNFactory

cudnn.benchmark = True

def get_pct_accuracy(pred, target):
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().data[0]
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def do_epoch(opt, loss_fn, discriminator, data_loader,
             optimizer = None, f = None, params = None, stats = None):
    acc_epoch = []
    loss_epoch = []

    for batch_idx, (data, label) in enumerate(data_loader):

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        inputs = Variable(data)
        targets = Variable(label)

        if f is not None and params is not None:
            # Convert to ARC Format.
            conv_features = []
            for i in range(inputs.shape[1]):
                if len(inputs.shape) == 4:
                    tmp = inputs[:, i, ...].unsqueeze(1) # add one_channel
                    wrn_features = data_parallel(f, tmp, params, stats, mode=False,
                                                 device_ids=np.arange(opt.ngpu))
                    conv_features.append(wrn_features.unsqueeze(1))
                else:
                    wrn_features = data_parallel(f, inputs[:, i, ...], params, stats,
                                                 mode=False, device_ids=np.arange(opt.wrn_ngpu))
                    conv_features.append(wrn_features.unsqueeze(1))
            inputs = torch.cat(conv_features, 1)

        elif f is not None and params is None:
            # Case in which we have the model but the params are contained inside the model
            # Convert to ARC Format.
            conv_features = []
            for i in range(inputs.shape[1]):
                if len(inputs.shape) == 4:
                    tmp = inputs[:, i, ...].unsqueeze(1) # add one_channel
                    wrn_features = f(tmp)
                    conv_features.append(wrn_features.unsqueeze(1))
                else:
                    wrn_features = f(inputs[:, i, ...])
                    conv_features.append(wrn_features.unsqueeze(1))
            inputs = torch.cat(conv_features, 1)

        pred_train = discriminator(inputs)
        loss = loss_fn(torch.squeeze(pred_train), targets.float())
        # Training...
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_epoch.append(get_pct_accuracy(torch.squeeze(pred_train), targets.int()))
        loss_epoch.append(loss.data[0])

    return acc_epoch, loss_epoch

def train():

    bnktBenchmark = omniglotBenchMark(type=OmniglotOSPairs, opt = Options())
    opt = bnktBenchmark.opt
    train_loader, val_loader, test_loader = bnktBenchmark.get()

    if opt.cuda:
        models.use_cuda = True

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "train_{}_{}_{}_{}_wrn".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join(opt.save, opt.name)
    if not os.path.isdir(models_path):
	    os.makedirs(models_path)
    else:
        shutil.rmtree(models_path)

    # create logger
    logger = Logger(models_path)

    if opt.apply_wrn:

        # Learn a convolutional cnn to generate feature maps which will be used
        # as inputs to the ArcBinaryClassifier.

        # Convert the opt params to dict.
        optDict = dict([(key, value) for key, value in opt._get_kwargs()])
        convCNN = ConvCNNFactory.createCNN(opt.wrn_name_type, optDict)
        if opt.wrn_load_path:
            # Load the model in fully convolutional mode
            f, params, stats = convCNN.load(opt.wrn_load_path, fully_convolutional = True)
        else:
            print('Training Conv CNN....')
            convCNN_bnktBenchmark = omniglotBenchMark(type=OmniglotOS, opt=Options())
            convCNN_train_loader, convCNN_val_loader, _ = convCNN_bnktBenchmark.get()
            convCNN.train(convCNN_train_loader, convCNN_val_loader, resume=opt.wrn_train_resume)
            # Load the model in fully convolutional mode
            f, params, stats = convCNN.load(opt.wrn_save, fully_convolutional = True)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        #channels=opt.nchannels,
                                        channels=64,
                                        controller_out=opt.numStates)
    if opt.cuda:
        discriminator.cuda()

    # load from a previous checkpoint, if specified.
    if opt.load is not None:
        discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr)

    # ready to train ...
    best_validation_loss = sys.float_info.max
    best_accuracy = None
    saving_threshold = 1.02
    last_saved = datetime.utcnow()
    save_every = timedelta(minutes=10)

    meta_data = {}
    meta_data["num_output"] = 2
    meta_data["train_loss"] = []
    meta_data["train_acc"] = []
    meta_data["val_loss"] = []
    meta_data["val_acc"] = []
    meta_data["test_loss"] = []
    meta_data["test_acc"] = []

    try:
        epoch = 0
        while epoch < opt.train_num_batches:
            epoch += 1
            tick = time.clock()

            if opt.apply_wrn:
                train_acc_epoch, train_loss_epoch = do_epoch(opt=opt, loss_fn=bce,
                                                             discriminator=discriminator,
                                                             data_loader=train_loader,
                                                             optimizer=optimizer, f=f,
                                                             params=params, stats=stats)
            else:
                train_acc_epoch, train_loss_epoch = do_epoch(opt=opt, loss_fn=bce,
                                                             discriminator=discriminator,
                                                             data_loader=train_loader,
                                                             optimizer=optimizer)
            tock = time.clock()
            train_acc_epoch = np.mean(train_acc_epoch)
            train_loss_epoch = np.mean(train_loss_epoch)
            meta_data["train_loss"].append((epoch, train_loss_epoch))
            meta_data["train_acc"].append((epoch, train_acc_epoch))
            print ("epoch: %d, train loss: %f, train acc: %.2f, time: %.2f s" %
                   (epoch,np.round(train_loss_epoch,6),np.round(train_acc_epoch,6),
                    np.round(tock - tick,4)))
            logger.add_scalar('train_loss', train_loss_epoch)
            logger.add_scalar('train_acc', train_acc_epoch)

            if np.isnan(train_loss_epoch):
                print ("... NaN Detected, terminating")
                break

            if epoch % opt.val_freq == 0:

                print ('... validation')

                val_epoch = 0
                val_acc_epoch = []
                val_loss_epoch = []
                tick = time.clock()
                while val_epoch < opt.val_num_batches:
                    val_epoch += 1
                    if opt.apply_wrn:
                        val_acc, val_loss = do_epoch(opt=opt, loss_fn=bce,
                                                                 discriminator=discriminator,
                                                                 data_loader=val_loader,
                                                                 f=f,params=params, stats=stats)
                    else:
                        val_acc, val_loss = do_epoch(opt=opt, loss_fn=bce,
                                                                 discriminator=discriminator,
                                                                 data_loader=val_loader)
                    val_acc_epoch.append(np.mean(val_acc))
                    val_loss_epoch.append(np.mean(val_loss))
                tock = time.clock()
                val_acc_epoch = np.mean(val_acc_epoch)
                val_loss_epoch = np.mean(val_loss_epoch)
                meta_data["val_loss"].append((epoch, val_loss_epoch))
                meta_data["val_acc"].append((epoch, val_acc_epoch))
                print ("====" * 20, "\n", "validation loss: ", val_loss_epoch\
                    , ", validation accuracy: ", val_acc_epoch, ", time: ",\
                    np.round((tock - tick)/opt.val_num_batches,4), "\n", "====" * 20)
                logger.add_scalar('val_loss', val_loss_epoch)
                logger.add_scalar('val_acc', val_acc_epoch)

                if best_validation_loss > (saving_threshold * val_loss_epoch):
                    print("Significantly improved validation loss from {} --> {}. Saving...".format(
                        best_validation_loss, val_loss_epoch
                    ))
                    discriminator.save_to_file(os.path.join(models_path, str(val_loss_epoch)))
                    best_validation_loss = val_loss_epoch
                    best_accuracy = val_acc_epoch
                    last_saved = datetime.utcnow()

                if last_saved + save_every < datetime.utcnow():
                    print("It's been too long since we last saved the model. Saving...")
                    discriminator.save_to_file(os.path.join(models_path, str(val_loss_epoch)))
                    last_saved = datetime.utcnow()

                meta_data["val_loss"] = []
                meta_data["val_acc"] = []

            logger.step()

    except KeyboardInterrupt:
        pass

    print ("... training done")
    print ("best validation accuracy: %.2f, best validation loss: %f" % (best_accuracy, best_validation_loss))
    print ("... exiting training regime")

    meta_data["train_loss"] = []
    meta_data["train_acc"] = []

    print ("... loading best validation model")
    opt.load = str(best_validation_loss)
    discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))

    tick = time.clock()
    print ('... testing')
    test_epoch = 0
    test_acc_epoch = []
    test_loss_epoch = []
    while test_epoch < opt.test_num_batches:
        test_epoch += 1
        if opt.apply_wrn:
            test_acc, test_loss = do_epoch(opt=opt, loss_fn=bce,
                                                     discriminator=discriminator,
                                                     data_loader=test_loader,
                                                     f=f,params=params, stats=stats)
        else:
            test_acc, test_loss = do_epoch(opt=opt, loss_fn=bce,
                                                       discriminator=discriminator,
                                                       data_loader=test_loader)
        test_acc_epoch.append(np.mean(test_acc))
        test_loss_epoch.append(np.mean(test_loss))

    tock = time.clock()
    test_acc_epoch = np.mean(test_acc_epoch)
    test_loss_epoch = np.mean(test_loss_epoch)
    meta_data["test_loss"].append((epoch, test_loss_epoch))
    meta_data["test_acc"].append((epoch, test_acc_epoch))
    print ("====" * 20, "\n", "test loss: ", test_loss_epoch \
        , ", test accuracy: ", test_acc_epoch, ", time: ", np.round((tock - tick)/opt.test_num_batches,4), \
        "\n", "====" * 20)
    logger.add_scalar('test_loss', test_loss_epoch)
    logger.add_scalar('test_acc', test_acc_epoch)

def main():
    train()

if __name__ == "__main__":
    main()
