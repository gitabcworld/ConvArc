import torch
import torch.nn as nn
from torch.autograd import Variable
from models.arc.glimpse import GlimpseWindow
from skiprnn_pytorch.rnn_cells.custom_cells import CMultiSkipLSTMCell,CSkipLSTMCell, \
                                                    CMultiSkipGRUCell, CSkipGRUCell, \
                                                    CBasicLSTMCell
import datetime


class SkipARC(nn.Module):
    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, channels = 3,
                 controller_out=128):
        super(SkipARC,self).__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out
        self.controller = CSkipLSTMCell(input_size=(glimpse_h * glimpse_w * channels),
                                             hidden_size=self.controller_out, batch_first=True)
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
        Hx = self.controller.init_hidden(batch_size)

        use_cuda = [i for i in self.glimpser.parameters()][0].is_cuda

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
            output, Hx, update_gate = self.controller(flattened_glimpses,Hx)
            time_end = datetime.datetime.now()
            time_elapsed_lstm += time_end - time_start
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

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)
        all_updated_states = torch.stack(all_updated_states)

        # return a batch of all hidden states.
        return all_hidden, all_updated_states
