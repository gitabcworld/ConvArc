import torch
import torch.nn as nn
from torch.autograd import Variable
from models.arc.glimpse import GlimpseWindow

class ARC_unroll(nn.Module):
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
        super(ARC_unroll,self).__init__()

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

        # Two dense layers to get the label
        self.dense1 = nn.Linear(controller_out, 64)
        self.dense2 = nn.Linear(64, 1)

        # loss_fn
        self.loss_fn = torch.nn.BCELoss()


    def forward(self, image_pairs, labels):
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

        use_cuda = [i for i in self.glimpser.parameters()][0].is_cuda

        # an array for collecting hidden states from each time step.
        all_hidden = []

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)
        Cx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)

        if use_cuda:
            Hx, Cx = Hx.cuda(), Cx.cuda()

        # take `num_glimpses` glimpses for both images, alternatively.
        decision = None
        total_loss = None
        if labels is not None:
            total_loss = Variable(torch.zeros(1),requires_grad = True)
            if use_cuda:
                total_loss = total_loss.cuda()

        lst_losses_turn = []
        lst_acc_turn = []
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

            d1 = F.elu(self.dense1(Hx))
            decision = torch.sigmoid(self.dense2(d1))
            if labels is not None:
                lst_losses_turn.append(self.loss_fn(decision,labels.float()).data.cpu().numpy())
                lst_acc_turn.append(((decision>0.5).long().squeeze() == labels).float().mean())
                total_loss += self.loss_fn(decision,labels.float()) * pow((turn+1),2)

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)

        # return only the last hidden state, decision and loss if needed
        return all_hidden[-1, :, :], decision, total_loss, lst_losses_turn, lst_acc_turn
