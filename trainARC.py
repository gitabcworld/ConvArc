import argparse
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from logger import Logger
import torch

import models
from dataset import omniglot
from dataset import banknote
from dataset.banknote import BanknoteVerif
from dataset.omniglot import OmniglotOS
from dataset.omniglot import OmniglotVerif
from models.models import ArcBinaryClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', type=int, default=1500000, help='train iterations')
parser.add_argument('--val_freq', type=int, default=500, help='validation frequency')
parser.add_argument('--val_num_batches', type=int, default=100, help='validation num batches')
parser.add_argument('--test_num_batches', type=int, default=100, help='validation num batches')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=4, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=512, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=8, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')


def get_pct_accuracy(pred, target):
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().data[0]
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy


def train():
    opt = parser.parse_args()

    if opt.cuda:
        banknote.use_cuda = True
        omniglot.use_cuda = True
        models.use_cuda = True

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)
    if not os.path.isdir(models_path):
	os.makedirs(models_path)

    # create logger
    logger = Logger(models_path)

    # load the dataset in memory.
    loader = OmniglotOS(batch_size=opt.batchSize, image_size=opt.imageSize)
    #loader = OmniglotVerif(batch_size=opt.batchSize, image_size=opt.imageSize)
    #loader = BanknoteVerif(batch_size=opt.batchSize, image_size=opt.imageSize)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        channels=loader.channels,
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
    meta_data["training_loss"] = []
    meta_data["validation_loss"] = []
    meta_data["validation_accuracy"] = []

    print "... training"
    try:
        iter_n = 0
        smooth_loss = np.log(meta_data["num_output"])
        while iter_n < opt.n_iter:
            iter_n += 1
            tick = time.clock()
            X_train, Y_train = loader.fetch_batch("train")
            pred_train = discriminator(X_train)
            training_loss = bce(pred_train, Y_train.float())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            training_loss = training_loss.data[0]
            train_acc = get_pct_accuracy(pred_train, Y_train)
            tock = time.clock()
            meta_data["training_loss"].append((iter_n, training_loss))
            print ("iteration: %d, train loss: %f, train acc: %.2f, time: %.2f s" %
                   (iter_n,np.round(training_loss, 6),train_acc, np.round((tock - tick))))
            logger.log_value('train_loss', training_loss)
            logger.log_value('train_acc', train_acc)
            #smooth_loss = 0.99 * smooth_loss + 0.01 * training_loss
            #print ("iteration: %d, train loss: %f, train acc: %f, time: %f ms" %
            #       (iter_n,np.round(smooth_loss, 4),train_acc, np.round((tock - tick))))

            if np.isnan(training_loss):
                print "... NaN Detected, terminating"
                break

            if iter_n % opt.val_freq == 0:

                net_val_loss, net_val_acc = 0.0, 0.0
                for i in xrange(opt.val_num_batches):
                    X_val, Y_val = loader.fetch_batch("val")
                    pred_val = discriminator(X_val)
                    loss_val = bce(pred_val, Y_val.float())
                    val_loss = loss_val.data[0]
                    val_acc = get_pct_accuracy(pred_val, Y_val)
                    net_val_loss += val_loss
                    net_val_acc += val_acc
                val_loss = net_val_loss / opt.val_num_batches
                val_acc = net_val_acc / opt.val_num_batches

                meta_data["validation_loss"].append((iter_n, val_loss))
                meta_data["validation_accuracy"].append((iter_n, val_acc))

                print "====" * 20, "\n", "validation loss: ", val_loss\
                    , ", validation accuracy: ", val_acc, "\n", "====" * 20
                logger.log_value('val_loss', val_loss)
                logger.log_value('val_acc', val_acc)

                if best_validation_loss > (saving_threshold * val_loss):
                    print("Significantly improved validation loss from {} --> {}. Saving...".format(
                        best_validation_loss, val_loss
                    ))
                    discriminator.save_to_file(os.path.join(models_path, str(val_loss)))
                    best_validation_loss = val_loss
                    best_accuracy = get_pct_accuracy(pred_val, Y_val)
                    last_saved = datetime.utcnow()

                if last_saved + save_every < datetime.utcnow():
                    print("It's been too long since we last saved the model. Saving...")
                    discriminator.save_to_file(os.path.join(models_path, str(val_loss)))
                    last_saved = datetime.utcnow()

            logger.step()

    except KeyboardInterrupt:
        pass

    print ("... training done")
    print ("best validation accuracy: %.2f, best validation loss: %f" % (best_accuracy, best_validation_loss))
    print "... exiting training regime"

    print ("... loading best validation model")
    opt.load = str(best_validation_loss)
    discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))

    print ('... testing')
    net_test_loss, net_test_acc = 0.0, 0.0
    for i in range(opt.test_num_batches):
        X_test, Y_test = loader.fetch_batch("test")
        pred_test = discriminator(X_test)
        loss_test = bce(pred_test, Y_test.float())
        acc_test = get_pct_accuracy(pred_test, Y_test)
        loss_test = loss_test.data[0]
        print ('i = %d, loss = %f, acc = %f' % (i,loss_test,acc_test))
        net_test_loss += loss_test
        net_test_acc += acc_test
    test_loss = net_test_loss / opt.test_num_batches
    test_acc = net_test_acc / opt.test_num_batches
    print "====" * 20, "\n", "test loss: ", test_loss, ", test accuracy: ", test_acc, "\n", "====" * 20


def main():
    train()


if __name__ == "__main__":
    main()
