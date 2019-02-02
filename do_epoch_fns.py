import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
import cv2

def compute_budget_loss(loss, updated_states, cost_per_sample = 0.001):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    if cost_per_sample > 0. and updated_states is not None:
        return torch.mean(torch.sum(cost_per_sample * updated_states,1),0)
    else:
        if loss.is_cuda:
            return Variable(torch.zeros(loss.shape).cuda(),requires_grad=True)
        else:
            return Variable(torch.zeros(loss.shape), requires_grad=True)


def compute_time_restricted_budget_loss(loss, updated_states, num_glimpses = 1):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    #x = Variable(torch.from_numpy(np.arange(updated_states.shape[0])).float(), requires_grad=True)
    x = Variable(torch.from_numpy(np.arange(0 - (num_glimpses-1),updated_states.shape[0]-(num_glimpses-1))).float(), requires_grad=True)
    if loss.is_cuda:
        x = x.cuda()

    softplus = nn.Softplus()(x)
    #elu = nn.ELU(1.0)(x - 3 - num_glimpses)
    #elu = nn.ELU(1.0)(x + 1 - num_glimpses)
    updated_states = torch.mean(updated_states.view(updated_states.shape[0], -1), 1)
    #glimpse_loss = torch.nn.ReLU()(updated_states * elu)
    #glimpse_loss = updated_states * elu
    glimpse_loss = updated_states * softplus
    return torch.sum(glimpse_loss) * 0.005


def binary_accuracy(pred, target):
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def do_epoch_ARC(opt, loss_fn, discriminator, data_loader,
             optimizer=None, fcn=None):
    acc_epoch = []
    loss_epoch = []

    activations_layers = []

    # Set for the first batch a random seed for AumentationAleju
    data_loader.dataset.agumentation_seed = int(np.random.rand() * 1000)

    for batch_idx, (data, label) in enumerate(data_loader):

        # If the input has been already forwarded in the DataLoader do not it again
        if not opt.fcn_applyOnDataLoader:

            if opt.cuda:
                data = data.cuda()
                label = label.cuda()
            if optimizer:
                inputs = Variable(data, requires_grad=True)
            else:
                inputs = Variable(data, requires_grad=False)
            targets = Variable(label)

            '''
            for index in range(inputs.shape[0]):
                cv2.imwrite('D:/PhD/images/batch_' + str(batch_idx) +'_index_' + str(index) + '_img1_target_' + str(
                    int(targets[index].data.cpu().numpy())) + '.png',
                            inputs[index, 0, :, :, :].transpose(0, 1).transpose(1, 2).data.cpu().numpy() * 255)
                cv2.imwrite('D:/PhD/images/batch_' + str(batch_idx) +'_index_' + str(index) + '_img2_target_' + str(
                    int(targets[index].data.cpu().numpy())) + '.png',
                            inputs[index, 1, :, :, :].transpose(0, 1).transpose(1, 2).data.cpu().numpy() * 255)
            '''

            # The dropout is done in the input if not CARC. If CARC the dropout is done in the
            # residual blocks in the Wide Residual Network.
            if optimizer and not fcn:
                inputs = torch.nn.Dropout(p=opt.dropout)(inputs)

            batch_size, npair, nchannels, x_size, y_size = inputs.shape
            inputs = inputs.view(batch_size * npair, nchannels, x_size, y_size)
            if fcn:
                inputs = fcn(inputs)
            _ , nfilters, featx_size, featy_size = inputs.shape
            inputs = inputs.view(batch_size, npair, nfilters, featx_size, featy_size)
        else:
            inputs = data
            if opt.cuda:
                label = label.cuda()
            targets = Variable(label)


        features, updated_states = discriminator(inputs)
        if loss_fn:
            loss = loss_fn(features.squeeze(), targets.float())
            # Add the budget computation cost
            #budget_loss = compute_time_restricted_budget_loss(loss, updated_states, num_glimpses=1)
            #alpha = 0.5
            #loss_total = alpha * loss + (1-alpha) * budget_loss
            loss_total = loss
            loss_epoch.append(loss_total.item())

        # Training...
        if optimizer and loss_fn:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        acc_epoch.append(binary_accuracy(features.squeeze(),targets.int()))

        # set a random seed for the next batch
        #data_loader.dataset.agumentation_seed = int(np.random.rand()*1000)
        #activations_layers.append(torch.mean(updated_states.view(updated_states.shape[0],-1),1).data.cpu().numpy())
    #print('Activations: %s' % str(list(np.mean(np.vstack(activations_layers),0))))

    return acc_epoch, loss_epoch

def do_epoch_ARC_unroll(opt, loss_fn, discriminator, data_loader,
             optimizer=None, fcn=None):
    acc_epoch = []
    loss_epoch = []

    data_loader.dataset.agumentation_seed = int(np.random.rand() * 1000)

    activations_layers = []
    lst_losses_epoch = []
    lst_acc_epoch = []
    for batch_idx, (data, label) in enumerate(data_loader):

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        if optimizer:
            inputs = Variable(data, requires_grad=True)
        else:
            inputs = Variable(data, requires_grad=False)
        targets = Variable(label)

        # The dropout is done in the input if not CARC. If CARC the dropout is done in the
        # residual blocks in the Wide Residual Network.
        if optimizer and not fcn:
            inputs = torch.nn.Dropout(p=opt.dropout)(inputs)

        batch_size, npair, nchannels, x_size, y_size = inputs.shape
        inputs = inputs.view(batch_size * npair, nchannels, x_size, y_size)
        if fcn:
            inputs = fcn(inputs)
        _ , nfilters, featx_size, featy_size = inputs.shape
        inputs = inputs.view(batch_size, npair, nfilters, featx_size, featy_size)

        features, decision, loss, lst_losses_turn, lst_acc_turn = discriminator(inputs,targets)
        # Training...
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc_epoch.append(binary_accuracy(decision.squeeze(),targets.int()))
        lst_losses_epoch.append(lst_losses_turn)
        lst_acc_epoch.append(lst_acc_turn)
        loss_epoch.append(loss.data[0])

    print('___'.join([str(i) + '_' + str(data) for i, data in enumerate(np.mean(np.hstack(lst_losses_epoch), 1))]))
    print('___'.join([str(i) + '_' + str(data) for i, data in enumerate(np.mean(np.vstack(lst_acc_epoch),0))]))
    return acc_epoch, loss_epoch

def do_epoch_naive_full(opt, discriminator, data_loader, model_fn,
                        loss_fn=None, optimizer=None, fcn=None):
    acc_epoch = []
    loss_epoch = []

    data_loader.dataset.agumentation_seed = int(np.random.rand() * 1000)

    for batch_idx, (data, label) in enumerate(data_loader):

        # If the input has been already forwarded in the DataLoader do not it again
        if not opt.fcn_applyOnDataLoader:
            if opt.cuda:
                data = data.cuda()
                label = label.cuda()

            '''
            for index_batch in range(data.shape[0]):
                for index_oneshot in range(data.shape[1]):
                    cv2.imwrite('D:/PhD/images/batch_' + str(index_batch) +'_index_' + str(index_oneshot) + '_img_target_' + str(
                        int(label[index_batch][index_oneshot].data.cpu().numpy())) + '.png', data[index_batch, index_oneshot].data.cpu().numpy().transpose(1,2,0) * 255)
            '''

            # not needed gradient graph for the FCN and ARC
            inputs = Variable(data, requires_grad = False)
            #inputs = Variable(data, requires_grad=True)
            targets = Variable(label)
            targets_binary = torch.stack([targets[i,:-data_loader.dataset.n_shot] == targets[i,-data_loader.dataset.n_shot] for i in range(len(targets))])

            batch_size, npair, nchannels, x_size, y_size = inputs.shape
            inputs = inputs.view(batch_size * npair, nchannels, x_size, y_size)
            if fcn:
                inputs = fcn(inputs)
            _ , nfilters, featx_size, featy_size = inputs.shape
            inputs = inputs.view(batch_size, npair, nfilters, featx_size, featy_size)
        else:
            inputs = data
            if opt.cuda:
                label = label.cuda()
            targets = Variable(label)
            targets_binary = torch.stack([targets[i,:-data_loader.dataset.n_shot] == targets[i,-data_loader.dataset.n_shot] for i in range(len(targets))])

        support_train = inputs[:,:data_loader.dataset.n_shot*data_loader.dataset.n_way,:]
        # repmat support test if all the discriminator could be processed in a single batch
        #support_test = inputs[:, npair-1:, :].expand(batch_size, npair-1, nfilters, featx_size, featy_size)
        support_test = inputs[:,data_loader.dataset.n_shot*data_loader.dataset.n_way:, :]

        hidden_features = []
        for i in range(support_train.shape[1]):
            #inputs = torch.cat((support_train[:, i, :].unsqueeze(1),support_test[:, i, :].unsqueeze(1)), dim=1)
            inputs = torch.cat((support_train[:, i, :].unsqueeze(1), support_test), dim=1)
            features = discriminator(inputs, return_arc_out = True)[0]
            hidden_features.append(features.unsqueeze(1))
        # Add the gradient graph control
        hidden_features = torch.cat(hidden_features,dim=1)
        features = Variable(hidden_features.data, requires_grad=True)
        #features = torch.cat(hidden_features, dim=1)
        features = model_fn(features)
        if loss_fn:
            loss = loss_fn(features, targets_binary.float())
            loss_epoch.append(loss.item())

        # Training...
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Find the top n-shots values. 
        values, index = torch.topk(features, k = data_loader.dataset.n_shot, dim=1, largest=True, sorted=True)
        features_binary = torch.zeros(targets_binary.size())
        # Set the indices to 1
        for i in range(len(index)):
            features_binary[i,index[i]]=1 
        tn, fp, fn, tp = confusion_matrix(targets_binary.view(-1).cpu().data.numpy(), features_binary.view(-1).cpu().data.numpy()).ravel()
        tnr = float(tn) / float(tn+fp)
        fnr = float(fn) / float(fn+tp)
        acc = float(tp+tn)/float(tp+tn+fp+fn)
        acc_epoch.append(acc)
        #values, index = torch.nn.Softmax(dim=1)(features).max(1)
        #acc_epoch.append(accuracy_score(y_true=targets.cpu().data.numpy(), y_pred=index.cpu().data.numpy()))


    return acc_epoch, loss_epoch
