#!/usr/bin/python
import numpy as np

from src.param_optimization.openISP.model.helpers import bilateral_filter, gen_gaussian_kernel


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
