#!/usr/bin/python
import numpy as np


class BNF:
    'Bilateral Noise Filtering'

    def __init__(self, img, dw, rw, rthres, clip):
        self.img = img
        self.dw = dw
        self.rw = rw
        self.rthres = rthres
        self.clip = clip

    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        img_pad = self.padding()
        img_pad = img_pad.astype(np.uint16)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        bnf_img = np.empty((raw_h, raw_w), np.uint16)
        rdiff = np.zeros((5, 5), dtype='uint16')
        for y in range(img_pad.shape[0] - 4):
            for x in range(img_pad.shape[1] - 4):
                print("[x,y]:[" + str(x) + ',' + str(y) + ']')
                for i in range(5):
                    for j in range(5):
                        rdiff[i, j] = abs(img_pad[y + i, x + j].astype(int) - img_pad[y + 2, x + 2].astype(int))
                        # rdiff[i,j] = abs(img_pad[y+i,x+j] - img_pad[y+2, x+2])
                        if rdiff[i, j] >= self.rthres[0]:
                            rdiff[i, j] = self.rw[0]
                        elif rdiff[i, j] < self.rthres[0] and rdiff[i, j] >= self.rthres[1]:
                            rdiff[i, j] = self.rw[1]
                        elif rdiff[i, j] < self.rthres[1] and rdiff[i, j] >= self.rthres[2]:
                            rdiff[i, j] = self.rw[2]
                        elif rdiff[i, j] < self.rthres[2]:
                            rdiff[i, j] = self.rw[3]
                weights = np.multiply(rdiff, self.dw)
                bnf_img[y, x] = np.sum(np.multiply(img_pad[y:y + 5, x:x + 5], weights[:, :])) / np.sum(weights)
        self.img = bnf_img
        return self.clipping()


class BNFVectorized:
    """
    Source: https://github.com/QiuJueqin/fast-openISP/blob/master/modules/bnf.py
    """

    def __init__(self, intensity_sigma, spatial_sigma):
        self.intensity_weights_lut = self.get_intensity_weights_lut(intensity_sigma)  # x1024
        spatial_weights = gen_gaussian_kernel(kernel_size=5, sigma=spatial_sigma)
        self.spatial_weights = (1024 * spatial_weights / spatial_weights.max()).astype(np.int32)  # x1024

    def execute(self, img):
        img = img.astype(np.int32)
        bf_y_image = bilateral_filter(img, self.spatial_weights, self.intensity_weights_lut, right_shift=10)
        return bf_y_image.astype(np.uint8)

    @staticmethod
    def get_intensity_weights_lut(intensity_sigma):
        intensity_diff = np.arange(255 ** 2)
        exp_lut = 1024 * np.exp(-intensity_diff / (2.0 * (255 * intensity_sigma) ** 2))
        return exp_lut.astype(np.int32)  # x1024


def gen_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, (list, tuple)):
        assert len(kernel_size) == 2
        wy, wx = kernel_size
    else:
        wy = wx = kernel_size

    x = np.arange(wx) - wx // 2
    if wx % 2 == 0:
        x += 0.5

    y = np.arange(wy) - wy // 2
    if wy % 2 == 0:
        y += 0.5

    y, x = np.meshgrid(y, x)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def bilateral_filter(array, spatial_weights, intensity_weights_lut, right_shift=0):
    """
    A faster reimplementation of the bilateral filter
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param spatial_weights: np.ndarray(h, w): predefined spatial gaussian kernel, where h and w are
        kernel height and width respectively
    :param intensity_weights_lut: a predefined exponential LUT that maps intensity distance to the weight
    :param right_shift: shift the multiplication result of the spatial- and intensity weights to the
        right to avoid integer overflow when multiply this result to the input array
    :return: filtered array: np.ndarray(H, W, ...)
    """
    filter_height, filter_width = spatial_weights.shape[:2]
    spatial_weights = spatial_weights.flatten()

    padded_array = pad(array, pads=(filter_height // 2, filter_width // 2))
    shifted_arrays = shift_array(padded_array, window_size=(filter_height, filter_width))

    bf_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        intensity_diff = (shifted_array - array) ** 2
        weight = intensity_weights_lut[intensity_diff] * spatial_weights[i]
        weight = np.right_shift(weight, right_shift)  # to avoid overflow

        bf_array += weight * shifted_array
        weights += weight

    bf_array = (bf_array / weights).astype(array.dtype)

    return bf_array


def pad(array, pads, mode='reflect'):
    """
    Pad an array with given margins
    :param array: np.ndarray(H, W, ...)
    :param pads: {int, sequence}
        if int, pad top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction pad, x-direction pad)
        if 4-element sequence: (top pad, bottom pad, left pad, right pad)
    :param mode: padding mode, see np.pad
    :return: padded array: np.ndarray(H', W', ...)
    """
    if isinstance(pads, (list, tuple, np.ndarray)):
        if len(pads) == 2:
            pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
        elif len(pads) == 4:
            pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
        else:
            raise NotImplementedError

    return np.pad(array, pads, mode)


def shift_array(padded_array, window_size):
    """
    Shift an array within a window and generate window_size**2 shifted arrays
    :param padded_array: np.ndarray(H+2r, W+2r)
    :param window_size: 2r+1
    :return: a generator of length (2r+1)*(2r+1), each is an np.ndarray(H, W), and the original
        array before padding locates in the middle of the generator
    """
    wy, wx = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wy % 2 == 1 and wx % 2 == 1, 'only odd window size is valid'

    height = padded_array.shape[0] - wy + 1
    width = padded_array.shape[1] - wx + 1

    for y0 in range(wy):
        for x0 in range(wx):
            yield padded_array[y0:y0 + height, x0:x0 + width, ...]
