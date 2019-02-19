import torch
from torch.autograd import Variable
import math

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
        use_cuda = delta_caps.is_cuda

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
            F_h = F_h.transpose(1, 2).unsqueeze(1).repeat(1,channels,1,1).contiguous() 
            F_h = F_h.view(batch_size*channels,F_h.shape[2],F_h.shape[3])
            F_w = F_w.unsqueeze(1).repeat(1,channels,1,1).contiguous()
            F_w = F_w.view(batch_size*channels,F_w.shape[2],F_w.shape[3])
            glimpses = glimpses.contiguous().view(batch_size*channels,image_h,image_w)
            glimpses = torch.bmm(F_h, glimpses)
            glimpses = torch.bmm(glimpses, F_w)
            glimpses = glimpses.view(batch_size, channels, self.glimpse_h, self.glimpse_w) # (B, c, glimpse_h, glimpse_w)
        else:
            glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
            glimpses = torch.bmm(glimpses, F_w) # (B, glimpse_h, glimpse_w)

        return glimpses
