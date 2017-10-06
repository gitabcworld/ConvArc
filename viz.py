import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.omniglot import OmniglotOS
from dataset.omniglot import OmniglotVerif
from dataset.banknote import BanknoteVerif
from models.models import ArcBinaryClassifier
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=4, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=512, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=4, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')
parser.add_argument('--same', action='store_true', help='whether to generate same character pairs or not')

opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

# make directory for storing images.
images_path = os.path.join("visualization", opt.name)
if not os.path.isdir(images_path):
	os.makedirs(images_path)


# initialise the batcher
#batcher = BanknoteVerif(batch_size=opt.batchSize)
#batcher = OmniglotVerif(batch_size=opt.batchSize)
batcher = OmniglotOS(batch_size=opt.batchSize)


def display(image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.numpy()/255, cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.numpy()/255, cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.savefig(os.path.join(images_path, name))


def get_sample(discriminator):

    # size of the set to choose sample from from
    sample_size = 30
    X, Y = batcher.fetch_batch("train", batch_size=sample_size)
    pred = discriminator(X)

    if opt.same:
        same_pred = pred[sample_size // 2:].data.numpy()[:, 0]
        mx = same_pred.argsort()[len(same_pred) // 2]  # choose the sample with median confidence
        index = mx + sample_size // 2
    else:
        diff_pred = pred[:sample_size // 2].data.numpy()[:, 0]
        mx = diff_pred.argsort()[len(diff_pred) // 2]  # choose the sample with median confidence
        index = mx

    return X[index]


def visualize():

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))

    arc = discriminator.arc

    sample = get_sample(discriminator)

    all_hidden = arc._forward(sample[None, :, :])[:, 0, :]  # (2*numGlimpses, controller_out)
    glimpse_params = torch.tanh(arc.glimpser(all_hidden))
    _, channels, height, witdth = sample.shape
    masks = arc.glimpse_window.get_attention_mask(glimpse_params, mask_h=height, mask_w=witdth)

    # separate the masks of each image.
    masks1 = []
    masks2 = []
    for i, mask in enumerate(masks):
        if i % 2 == 1:  # the first image outputs the hidden state for the next image
            masks1.append(mask)
        else:
            masks2.append(mask)

    channels = 3
    for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
        sample_0 = (sample[0].transpose(0, 1).transpose(1, 2) + \
                    Variable(torch.from_numpy(batcher.mean_pixel)).float()) * 255
        sample_1 = (sample[1].transpose(0, 1).transpose(1, 2) + \
                    Variable(torch.from_numpy(batcher.mean_pixel)).float()) * 255

        display(sample_0, mask1, sample_1, mask2,"img_{}".format(i))


if __name__ == "__main__":
    visualize()
