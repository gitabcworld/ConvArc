import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from skiprnn_pytorch.rnn_cells.custom_cells import CMultiSkipLSTMCell,CSkipLSTMCell, \
                                                    CMultiSkipGRUCell, CSkipGRUCell, \
                                                    CBasicLSTMCell
import datetime

use_cuda = False


class GlimpseWindow:
    """
    Generates glimpses from images using Cauchy kernels.

    Args:
        glimpse_h (int): The height of the glimpses to be generated.
        glimpse_w (int): The width of the glimpses to be generated.

    """

    def __init__(self, glimpse_h, glimpse_w):
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w

    @staticmethod
    def _get_filterbanks(delta_caps, center_caps, image_size, glimpse_size):
        """
        Generates Cauchy Filter Banks along a dimension.

        Args:
            delta_caps (B,):  A batch of deltas [-1, 1]
            center_caps (B,): A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            image_size (int): size of images along that dimension
            glimpse_size (int): size of glimpses to be generated along that dimension

        Returns:
            (B, image_size, glimpse_size): A batch of filter banks

        """

        # convert dimension sizes to float. lots of math ahead.
        image_size = float(image_size)
        glimpse_size = float(glimpse_size)

        # scale the centers and the deltas to map to the actual size of given image.
        centers = (image_size - 1) * (center_caps + 1) / 2.0  # (B)
        deltas = (float(image_size) / glimpse_size) * (1.0 - torch.abs(delta_caps))

        # calculate gamma for cauchy kernel
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))  # (B)

        # coordinate of pixels on the glimpse
        glimpse_pixels = Variable(torch.arange(0, glimpse_size) - (glimpse_size - 1.0) / 2.0)  # (glimpse_size)
        if use_cuda:
            glimpse_pixels = glimpse_pixels.cuda()

        # space out with delta
        glimpse_pixels = deltas[:, None] * glimpse_pixels[None, :]  # (B, glimpse_size)
        # center around the centers
        glimpse_pixels = centers[:, None] + glimpse_pixels  # (B, glimpse_size)

        # coordinates of pixels on the image
        image_pixels = Variable(torch.arange(0, image_size))  # (image_size)
        if use_cuda:
            image_pixels = image_pixels.cuda()

        fx = image_pixels - glimpse_pixels[:, :, None]  # (B, glimpse_size, image_size)
        fx = fx / gammas[:, None, None]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, None, None] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 1e-4)[:, :, None]  # we add a small constant in the denominator division by 0.

        return fx.transpose(1, 2)

    def get_attention_mask(self, glimpse_params, mask_h, mask_w):
        """
        For visualization, generate a heat map (or mask) of which pixels got the most "attention".

        Args:
            glimpse_params (B, hx):  A batch of glimpse parameters.
            mask_h (int): The height of the image for which the mask is being generated.
            mask_w (int): The width of the image for which the mask is being generated.

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended pixels weighted more.

        """

        batch_size, _ = glimpse_params.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=mask_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=mask_w, glimpse_size=self.glimpse_w)

        # (B, glimpse_h, glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))

        # find the attention mask that lead to the glimpse.
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))

        # scale to between 0 and 1.0
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()

        return mask

    def get_glimpse(self, images, glimpse_params):
        """
        Generate glimpses given images and glimpse parameters. This is the main method of this class.

        The glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
        represents the relative position of the center of the glimpse on the image. delta determines
        the zoom factor of the glimpse.

        Args:
            images (B, c, h, w):  A batch of images
            glimpse_params (B, 3):  A batch of glimpse parameters (h_center, w_center, delta)

        Returns:
            (B, glimpse_h, glimpse_w): A batch of glimpses.

        """
        if len(images.size())==4:
            #channels, batch_size, image_h, image_w = images.size()
            batch_size, channels, image_h, image_w = images.size()
        else:
            batch_size, image_h, image_w = images.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=image_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=image_w, glimpse_size=self.glimpse_w)

        # F_h.T * images * F_w
        glimpses = images
        # support for 1-3 channel images.
        if len(glimpses.shape)==4:
            num_channels = glimpses.shape[1]
            all_glimpses = []
            for c in range(num_channels):
                glimpses_c = torch.bmm(F_h.transpose(1, 2), glimpses[:,c,:,:])
                glimpses_c = torch.bmm(glimpses_c, F_w)
                all_glimpses.append(glimpses_c)
            glimpses = torch.stack(all_glimpses).transpose(0,1)  # (B, c, glimpse_h, glimpse_w)
            glimpses.contiguous()
        else:
            glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
            #glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
            glimpses = torch.bmm(glimpses, F_w) # (B, glimpse_h, glimpse_w)

        return glimpses


class ARC(nn.Module):
    """
    This class implements the Attentive Recurrent Comparators. This module has two main parts.

    1.) controller: The RNN module that takes as input glimpses from a pair of images and emits a hidden state.

    2.) glimpser: A Linear layer that takes the hidden state emitted by the controller and generates the glimpse
                    parameters. These glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
                    represents the relative position of the center of the glimpse on the image. delta determines
                    the zoom factor of the glimpse.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, channels = 3,
                 controller_out=128):
        super(ARC,self).__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out

        # main modules of ARC
        self.controller = nn.LSTMCell(input_size=(glimpse_h * glimpse_w * channels),
                                      hidden_size=self.controller_out)
        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)

        # this will actually generate glimpses from images using the glimpse parameters.
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w)

    def forward(self, image_pairs):
        """
        The method calls the internal _forward() method which returns hidden states for all time steps. This i

        Args:
            image_pairs (B, 2, h, w):  A batch of pairs of images

        Returns:
            (B, controller_out): A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """

        # return only the last hidden state
        all_hidden = self._forward(image_pairs)  # (2*num_glimpses, B, controller_out)
        last_hidden = all_hidden[-1, :, :]  # (B, controller_out)

        return last_hidden, None

    def _forward(self, image_pairs):
        """
        The main forward method of ARC. But it returns hidden state from all time steps (all glimpses) as opposed to
        just the last one. See the exposed forward() method.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (2*num_glimpses, B, controller_out) Hidden states from ALL time steps.

        """

        # convert to images to float.
        image_pairs = image_pairs.float()

        # calculate the batch size
        batch_size = image_pairs.size()[0]

        # an array for collecting hidden states from each time step.
        all_hidden = []

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)
        Cx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)

        if use_cuda:
            Hx, Cx = Hx.cuda(), Cx.cuda()

        # take `num_glimpses` glimpses for both images, alternatively.
        for turn in range(2*self.num_glimpses):
            # select image to show, alternate between the first and second image in the pair
            images_to_observe = image_pairs[:,  turn % 2]  # (B, c, h, w)

            # choose a portion from image to glimpse using attention
            glimpse_params = torch.tanh(self.glimpser(Hx))  # (B, 3)  a batch of glimpse params (x, y, delta)
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.contiguous().view(batch_size, -1)  # (B, glimpse_h * glimpse_w), one time-step

            # feed the glimpses and the previous hidden state to the LSTM.
            Hx, Cx = self.controller(flattened_glimpses, (Hx, Cx))  # (B, controller_out), (B, controller_out)

            # append this hidden state to all states
            all_hidden.append(Hx)

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)

        # return a batch of all hidden states.
        return all_hidden

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class SkipARC(nn.Module):
    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, channels = 3,
                 controller_out=128):
        super(SkipARC,self).__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out

        # modules skip: CMultiSkipLSTMCell, CMultiSkipGRUCell
        #self.controller = CMultiSkipLSTMCell(input_size=(glimpse_h * glimpse_w * channels),
        #                              hidden_size=self.controller_out, batch_first=True, num_layers=3)
        self.controller = CSkipLSTMCell(input_size=(glimpse_h * glimpse_w * channels),
                                             hidden_size=self.controller_out, batch_first=True)

        #self.controller = CSkipLSTMCell(input_size=(glimpse_h * glimpse_w * channels),
        #                                     hidden_size=self.controller_out, batch_first=True)
        #self.controller = CSkipGRUCell(input_size=(glimpse_h * glimpse_w * channels),
        #                                hidden_size=self.controller_out, batch_first=True)

        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w)

    def forward(self, image_pairs):
        all_hidden, updated_states = self._forward(image_pairs)  # (2*num_glimpses, B, controller_out)
        last_hidden = all_hidden[-1, :, :]  # (B, controller_out)
        return last_hidden, updated_states

    def _forward(self, image_pairs):
        image_pairs = image_pairs.float()
        batch_size = image_pairs.size()[0]
        all_hidden = []
        all_updated_states = []

        # initial hidden state of the LSTM.
        #Hx = self.controller.init_hidden(batch_size)
        Hx = self.controller.init_hidden(batch_size)

        if use_cuda:
            if self.controller.num_layers == 1:
                Hx = tuple([x.cuda() for x in Hx])
            else:
                Hx = [tuple([j.cuda() if j is not None else None for j in i]) for i in Hx]

        if 'lstm' in str(type(self.controller)).lower():
            if self.controller.num_layers == 1:
                c_prev, h_prev, _, _ = Hx
            else:
                c_prev, h_prev, _, _ = Hx[-1]
        if 'gru' in str(type(self.controller)).lower():
            if self.controller.num_layers == 1:
                h_prev, _, _ = Hx
            else:
                h_prev, _, _ = Hx[-1]

        # take `num_glimpses` glimpses for both images, alternatively.
        time_elapsed_glimpse = datetime.timedelta(0)
        time_elapsed_lstm = datetime.timedelta(0)
        history_flattened_glimpses = []
        for turn in range(2*self.num_glimpses):
            # select image to show, alternate between the first and second image in the pair
            time_start = datetime.datetime.now()
            images_to_observe = image_pairs[:,  turn % 2]  # (B, c, h, w)
            # choose a portion from image to glimpse using attention
            glimpse_params = torch.tanh(self.glimpser(h_prev))  # (B, 3)  a batch of glimpse params (x, y, delta)
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.contiguous().view(batch_size, -1).unsqueeze(1) # (B, sequence=1, glimpse_h * glimpse_w), one time-step
            time_end = datetime.datetime.now()
            time_elapsed_glimpse += time_end - time_start
            time_start = datetime.datetime.now()
            history_flattened_glimpses.append(flattened_glimpses)
            output, Hx, update_gate = self.controller(torch.cat(history_flattened_glimpses, 1), Hx)
            if output.shape[1] > 1:
                output = output[:,-1,:].unsqueeze(1) # select the last output in the sequence.
                update_gate = update_gate[:,-1,:].unsqueeze(1)
            #output, Hx, update_gate = self.controller(flattened_glimpses,Hx)
            time_end = datetime.datetime.now()
            time_elapsed_lstm += time_end - time_start
            #print('updated gates: %f' % (torch.sum(update_gate)/update_gate.shape[0]))
            #print('H[2][-1].sum(): %f' % (Hx[2].sum()))
            #print('H[3][-1].sum(): %f' % (Hx[3].sum()))
            if 'lstm' in str(type(self.controller)).lower():
                if self.controller.num_layers == 1:
                    c_prev, h_prev, update_prob_prev, cum_update_prob_prev = Hx
                else:
                    c_prev, h_prev, _, _ = Hx[-1]
            if 'gru' in str(type(self.controller)).lower():
                if self.controller.num_layers == 1:
                    h_prev, _, _= Hx
                else:
                    h_prev, _, _= Hx[-1]

            # append this hidden state to all states
            all_hidden.append(output)
            all_updated_states.append(update_gate)

        #print ("time glimpse: %02ds:%03dms, time lstm: %02ds:%03dms" % (
        #                time_elapsed_glimpse.seconds / 2*self.num_glimpses, time_elapsed_glimpse.microseconds / (1000 * 2*self.num_glimpses),
        #                time_elapsed_lstm.seconds / 2 * self.num_glimpses,time_elapsed_lstm.microseconds / (1000 * 2 * self.num_glimpses)))

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)
        all_updated_states = torch.stack(all_updated_states)

        # return a batch of all hidden states.
        return all_hidden, all_updated_states


class ArcBinaryClassifier(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, channels = 3,
                 controller_out = 128):
        super(ArcBinaryClassifier,self).__init__()
        '''
        self.arc = ARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            channels = channels,
            controller_out=controller_out)
        '''

        self.arc = SkipARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            channels=channels,
            controller_out=controller_out)


        # two dense layers, which take the hidden state from the controller of ARC and
        # classify the images as belonging to the same class or not.
        self.dense1 = nn.Linear(controller_out, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image_pairs, return_arc_out = False):
        arc_out, updated_states = self.arc(image_pairs)
        if return_arc_out:
            return arc_out, updated_states

        d1 = F.elu(self.dense1(arc_out))
        decision = torch.sigmoid(self.dense2(d1))
        #decision = self.dense2(d1) # To call BCEWithLogitsLoss

        return decision, updated_states

    def save_to_file(self, file_path):
        torch.save(self.state_dict(), file_path)


