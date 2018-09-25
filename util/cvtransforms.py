""" OpenCV-based transforms
    Operate on np.ndarrays only, no PIL or torch dependency
"""
from __future__ import division
import math
import random
import numpy as np
import numbers
import cv2
from util.imageAugmenter import ImageAugmenter

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the np.ndarray, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class Scale(object):
    """Rescales the input np.ndarray to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(float(self.size) * h / w)
        else:
            oh = self.size
            ow = int(float(self.size) * w / h)
        return cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)


class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1+th, x1:x1+tw, :]


class Pad(object):
    """Pads the given np.ndarray on all sides with the given "pad" value."""

    def __init__(self, padding, borderType=cv2.BORDER_CONSTANT, borderValue=0):
        assert isinstance(padding, numbers.Number)
        self.padding = padding
        self.borderType = borderType
        self.borderValue = borderValue

    def __call__(self, img):
        if self.padding == 0:
            return img
        p = self.padding
        res = cv2.copyMakeBorder(img, p, p, p, p,
                                 borderType=self.borderType,
                                 value=self.borderValue)
        return res[:, :, np.newaxis] if np.ndim(res) == 2 else res


class RandomCrop(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1+th, x1:x1+tw, :]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            return cv2.flip(img, 1).reshape(img.shape)
        return img


class RandomSizedCrop(object):
    """Random crop the given np.ndarray to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_CUBIC
    """
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)

                img = img[y1:y1+h, x1:x1+w, :]
                assert img.shape[0] == h and img.shape[1] == w

                return cv2.resize(img, (self.size, self.size),
                                  interpolation=self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class AugmentationAleju(object):
    """Wrapper of the imageAugmenter.py by Aleju
    Args:
        img_width_px: The intended width of each image in pixels.
        img_height_px: The intended height of each image in pixels.
        channel_is_first_axis: Whether the channel (e.g. RGB) is the first
            axis of each image (True) or the last axis (False).
            False matches the scipy and PIL implementation and is the
            default. If your images are 2D-grayscale then you can ignore
            this setting (as the augmenter will ignore it too).
        hflip: Whether to randomly flip images horizontally (on the y-axis).
            You may choose either False (no horizontal flipping),
            True (flip with probability 0.5) or use a float
            value (probability) between 0.0 and 1.0. Default is False.
        vflip: Whether to randomly flip images vertically (on the x-axis).
            You may choose either False (no vertical flipping),
            True (flip with probability 0.5) or use a float
            value (probability) between 0.0 and 1.0. Default is False.
        scale_to_percent: Up to which percentage the images may be
            scaled/zoomed. The negative scaling is automatically derived
            from this value. A value of 1.1 allows scaling by any value
            between -10% and +10%. You may set min and max values yourself
            by using a tuple instead, like (1.1, 1.2) to scale between
            +10% and +20%. Default is 1.0 (no scaling).
        scale_axis_equally: Whether to always scale both axis (x and y)
            in the same way. If set to False, then e.g. the Augmenter
            might scale the x-axis by 20% and the y-axis by -5%.
            Default is False.
        rotation_deg: By how much the image may be rotated around its
            center (in degrees). The negative rotation will automatically
            be derived from this value. E.g. a value of 20 allows any
            rotation between -20 degrees and +20 degrees. You may set min
            and max values yourself by using a tuple instead, e.g. (5, 20)
            to rotate between +5 und +20 degrees. Default is 0 (no
            rotation).
        shear_deg: By how much the image may be sheared (in degrees). The
            negative value will automatically be derived from this value.
            E.g. a value of 20 allows any shear between -20 degrees and
            +20 degrees. You may set min and max values yourself by using a
            tuple instead, e.g. (5, 20) to shear between +5 und +20
            degrees. Default is 0 (no shear).
        translation_x_px: By up to how many pixels the image may be
            translated (moved) on the x-axis. The negative value will
            automatically be derived from this value. E.g. a value of +7
            allows any translation between -7 and +7 pixels on the x-axis.
            You may set min and max values yourself by using a tuple
            instead, e.g. (5, 20) to translate between +5 und +20 pixels.
            Default is 0 (no translation on the x-axis).
        translation_y_px: See translation_x_px, just for the y-axis.
        transform_channels_equally: Whether to apply the exactly same
            transformations to each channel of an image (True). Setting
            it to False allows different transformations per channel,
            e.g. the red-channel might be rotated by +20 degrees, while
            the blue channel (of the same image) might be rotated
            by -5 degrees. If you don't have any channels (2D grayscale),
            you can simply ignore this setting.
            Default is True (transform all equally).
    """
    def __init__(self, channel_is_first_axis = False,
                        hflip = False, vflip = False,
                        scale_to_percent = 1.0, scale_axis_equally = False,
                        rotation_deg = 0, shear_deg = 0,
                        translation_x_px = 0, translation_y_px = 0,
                        transform_channels_equally = True):

        self.seed = None
        self.augmenter = ImageAugmenter(None, None,
                                        channel_is_first_axis,
                                        hflip, vflip,
                                        scale_to_percent, scale_axis_equally,
                                        rotation_deg, shear_deg,
                                        translation_x_px, translation_y_px,
                                        transform_channels_equally)

    def __call__(self, img):
        return self.augmenter.augment_image(img,seed=self.seed)
