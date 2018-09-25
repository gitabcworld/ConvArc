"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    2017 Albert Berenguel
"""

import json
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from resnet import resnet
from utils import cast, data_parallel

from sklearn.metrics import accuracy_score
import time
import random
import operator
#from memory_profiler import profile
from trainBanknoteBenchmarks import banknoteBenchMark

import torch.nn as nn
cudnn.benchmark = True

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y.float() * dist_sq + (1 - y.float()) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores.sort(key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count



def main():

    #bnktBenchmark = banknoteBenchMark(type = 'fullBanknoteROI')
    #bnktBenchmark = banknoteBenchMark(type='fullBanknotePairsROI')
    bnktBenchmark = banknoteBenchMark(type='fullBanknoteTripletsROI')
    opt = bnktBenchmark.opt

    epoch_step = json.loads(opt.epoch_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    train_loader, val_loader, test_loader = bnktBenchmark.get()
    num_classes = train_loader.dataset.numClasses
    f, params, stats = resnet(opt.depth, opt.width, num_classes, is_full_wrn=False)

    def create_optimizer(opt, lr):
        print 'creating optimizer with lr = ', lr
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(params.values(), lr, 0.9, weight_decay=opt.weightDecay)
        elif opt.optim_method == 'Adam':
            return torch.optim.Adam(params.values(), lr)

    def log(t, optimizer, params, stats, opt):
        torch.save(dict(params={k: v.data for k, v in params.iteritems()},
                        stats=stats,
                        optimizer=optimizer.state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'w'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print z


    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in params.iteritems():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print '\nParameters:'
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data)
    print '\nAdditional buffers:'
    kmax = max(len(key) for key in stats.keys())
    for i, (key, v) in enumerate(stats.items()):
        print str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v)

    n_parameters = sum(p.numel() for p in params.values() + stats.values())
    print '\nTotal number of parameters:', n_parameters

    # Save folder
    best_val_acc = 0
    if opt.save == '':
        opt.save = '/home/aberenguel/pytorch/examples/ConvArc/models/wrn/logs/resnet_' + str(random.getrandbits(128))[:-20]
        #opt.save = './logs/resnet_' + str(random.getrandbits(128))[:-20]
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    ######################################
    # TRAIN
    ######################################
    epoch = 0
    try:
        while True:
        # for epoch in range(opt.epochs):

            train_all_acc = []
            train_all_losses = []

            # Every 10 epoch change the Pairs of the dataset
            if epoch % 200 == 0 and not epoch == 0:
                print('Train: generate new data pairs at epoch %d' % epoch)
                train_loader.dataset.generate(1000)

            tick = time.time() #clock()
            for batch_idx, (data, label) in enumerate(train_loader):

                # Select the first ROI of the second branch
                #data = data[:, 0, 1, :]
                #label = label[3]

                if bnktBenchmark.type == 'fullBanknoteROI':
                    # Select only the first ROI
                    data = data[:, 0, :]
                    label = label[1]

                if bnktBenchmark.type == 'fullBanknotePairsROI':
                    model1, target1, model2, target2 = label
                    label = target1 == target2

                if bnktBenchmark.type == 'fullBanknoteTripletsROI':
                    model1, target1, model2, target2, model3, target3 = label
                    label = torch.cat(((target1==target2),(target2==target3)),0)

                if opt.cuda:
                    data = data.cuda()
                    label = label.cuda()
                inputs = Variable(data)
                targets = Variable(label)

                model_training = True

                if bnktBenchmark.type == 'fullBanknoteROI':
                    train_preds = data_parallel(f, inputs, params, stats, model_training, np.arange(opt.ngpu))
                    training_loss = F.cross_entropy(train_preds, targets)

                if bnktBenchmark.type == 'fullBanknotePairsROI':
                    # Get only the first ROI
                    train_preds1 = data_parallel(f, inputs[:, 0, 0, :], params, stats, model_training,
                                                 np.arange(opt.ngpu))
                    train_preds2 = data_parallel(f, inputs[:, 0, 1, :], params, stats, model_training,
                                                 np.arange(opt.ngpu))
                    training_loss = ContrastiveLoss()(train_preds1,train_preds2,targets)
                    distances = torch.sqrt(torch.sum((train_preds1 - train_preds2) ** 2, 1))  # euclidean distance

                if bnktBenchmark.type == 'fullBanknoteTripletsROI':
                    # Get only the first ROI
                    train_preds1 = data_parallel(f, inputs[:, 0, 0, :], params, stats, model_training,
                                                 np.arange(opt.ngpu))
                    train_preds2 = data_parallel(f, inputs[:, 0, 1, :], params, stats, model_training,
                                                 np.arange(opt.ngpu))
                    train_preds3 = data_parallel(f, inputs[:, 0, 2, :], params, stats, model_training,
                                                 np.arange(opt.ngpu))
                    training_loss = F.triplet_margin_loss(train_preds1, train_preds2, train_preds3)

                    distancesPositive = torch.sqrt(
                        torch.sum((train_preds1 - train_preds2) ** 2, 1))
                    distancesNegative = torch.sqrt(
                        torch.sum((train_preds2 - train_preds3) ** 2, 1))  # euclidean distance



                #m = torch.nn.Sigmoid()
                #loss = torch.nn.BCELoss()
                #training_loss = loss(m(train_preds), targets.float())

                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()

                if bnktBenchmark.type == 'fullBanknoteROI':
                    train_probs, train_classes = torch.max(train_preds, 1)
                    train_acc = accuracy_score(targets.data.cpu().numpy(),train_classes.data.cpu().numpy())
                if bnktBenchmark.type == 'fullBanknotePairsROI':
                    fpr95 = ErrorRateAt95Recall(targets.data.cpu().numpy(), distances.data.cpu().numpy())
                    train_acc = fpr95
                if bnktBenchmark.type == 'fullBanknoteTripletsROI':
                    fpr95 = ErrorRateAt95Recall(targets.data.cpu().numpy(),
                                                torch.cat((distancesPositive, distancesNegative), 0).data.cpu().numpy())
                    train_acc = fpr95

                train_all_acc.append(train_acc)
                train_all_losses.append(training_loss.data.cpu().numpy()[0])

                # Free memory
                inputs = []
                targets = []
                data = []
                label = []

            tock = time.time()

            if bnktBenchmark.type == 'fullBanknoteROI':
                print ("epoch: %d, train loss: %f, train acc: %.2f, time: %.2f s" %
                       (epoch, np.round(np.mean(train_all_losses), 6), np.mean(train_all_acc),
                            np.round((tock - tick))))
            if bnktBenchmark.type == 'fullBanknotePairsROI' or bnktBenchmark.type == 'fullBanknoteTripletsROI':
                print ("epoch: %d, train loss: %f, train acc: %.2f, time: %.2f s" %
                   (epoch, np.round(np.mean(train_all_losses), 6), np.mean(train_all_acc),
                    np.round((tock - tick))))

            # Adjust learning rate
            if epoch in epoch_step:
                lr = optimizer.param_groups[0]['lr']
                optimizer = create_optimizer(opt, lr * opt.lr_decay_ratio)

            # Validation
            if epoch % opt.eval_freq == 0:

                tick = time.time()
                all_preds = []
                all_targets = []
                for batch_idx, (data, label) in enumerate(val_loader):

                    if bnktBenchmark.type == 'fullBanknoteROI':
                        # Select only the first ROI
                        data = data[:, 0, :]
                        label = label[1]

                    if bnktBenchmark.type == 'fullBanknotePairsROI' or bnktBenchmark.type == 'fullBanknoteTripletsROI':
                        model1, target1, model2, target2 = label
                        label = target1 == target2

                    if opt.cuda:
                        data = data.cuda()
                    inputs = Variable(data)

                    model_training = False
                    if bnktBenchmark.type == 'fullBanknoteROI':
                        y = data_parallel(f, inputs, params, stats, model_training, np.arange(opt.ngpu))

                    if bnktBenchmark.type == 'fullBanknotePairsROI' or bnktBenchmark.type == 'fullBanknoteTripletsROI':
                        # Get only the first ROI
                        val_preds1 = data_parallel(f, inputs[:, 0, 0, :], params, stats, model_training,
                                                     np.arange(opt.ngpu))
                        val_preds2 = data_parallel(f, inputs[:, 0, 1, :], params, stats, model_training,
                                                     np.arange(opt.ngpu))
                        y = torch.sqrt(torch.sum((val_preds1 - val_preds2) ** 2, 1))

                    inputs = []
                    data = []
                    y = y.data.cpu().numpy()
                    label = label.numpy()
                    all_preds.append(y)
                    all_targets.append(label)

                all_targets = np.hstack(all_targets)
                if bnktBenchmark.type == 'fullBanknoteROI':
                    all_preds = np.vstack(all_preds)
                    all_preds = all_preds.argmax(1)
                    val_acc = accuracy_score(all_targets, all_preds)
                if bnktBenchmark.type == 'fullBanknotePairsROI' or bnktBenchmark.type == 'fullBanknoteTripletsROI':
                    all_preds = np.hstack(all_preds)
                    fpr95 = ErrorRateAt95Recall(all_targets, all_preds)
                    val_acc = 1 - fpr95

                if val_acc >= best_val_acc:
                    log({
                        "train_loss": float(np.mean(train_all_losses)),
                        "train_acc": float(np.mean(train_all_acc)),
                        "val_acc": val_acc,
                        "epoch": epoch,
                        "num_classes": num_classes,
                        "n_parameters": n_parameters,
                    }, optimizer, params, stats, opt)

                # Free memory
                data = [],
                label = []
                y = []

                tock = time.time()  # .clock()
                print ("epoch: %d, val acc: %f, time: %.2f s" %
                   (epoch, val_acc, np.round((tock - tick))))

            epoch = epoch + 1

    except KeyboardInterrupt:
        pass

    train_loader = []
    val_loader = []

    # load best accuracy
    state_dict = torch.load(os.path.join(opt.save, 'model.pt7'))
    params_tensors, stats = state_dict['params'], state_dict['stats']
    for k, v in params.iteritems():
        v.data.copy_(params_tensors[k])
    print('Loaded WRN from epoch %d' % state_dict['epoch'])

    all_preds = []
    all_targets = []
    for batch_idx, (data, label) in enumerate(test_loader):
        # Select the first ROI of the second branch
        # data = data[:, 0, 1, :]
        # label = label[1]
        data = data[:, 0, :]
        label = label[1]

        if opt.cuda:
            data = data.cuda()
        inputs = Variable(data)
        model_training = False
        y = data_parallel(f, inputs, params, stats, model_training, np.arange(opt.ngpu))
        inputs = []
        data = []
        y = y.data.cpu().numpy()
        label = label.numpy()
        all_preds.append(y)
        all_targets.append(label)

    all_preds = np.vstack(all_preds).argmax(1)
    all_targets = np.hstack(all_targets)
    test_acc = accuracy_score(all_targets, all_preds)

    # Free memory
    data = [],
    label = []
    y = []

    tock = time.time()  # .clock()
    print ("++++++ test acc: %f, time: %.2f s" % (test_acc, np.round((tock - tick))))


if __name__ == '__main__':
    main()
