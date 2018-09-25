import torch
import torch.nn as nn
import torch.nn.functional as F
from arc.arc import ARC
from arc.arc_unroll import ARC_unroll
from arc.arc_skip import SkipARC
from arc.arc_skip_dense import SkipARC_dense

use_cuda = False

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
    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, channels=3, controller_out=128, attn_type='LSTM',
                 attn_unroll=False, attn_dense=False):
        super(ArcBinaryClassifier, self).__init__()

        if attn_type == 'LSTM':
            if attn_unroll:
                attn_model = ARC_unroll
            else:
                attn_model = ARC
        else:
            if attn_dense:
                attn_model = SkipARC_dense
            else:
                attn_model = SkipARC

        self.arc = attn_model(
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


