#!/usr/bin/python
import numpy as np

from src.param_optimization.openISP.model.helpers import gen_gaussian_kernel, generic_filter


class EE:
    'Edge Enhancement'

    def __init__(self, img, edge_filter, gain, thres, emclip):
        self.img = img
        self.edge_filter = edge_filter
        self.gain = gain
        self.thres = thres
        self.emclip = emclip

    def padding(self):
        img_pad = np.pad(self.img, ((1, 1), (2, 2)), 'reflect')
        return img_pad

    def clipping(self):
        np.clip(self.img, 0, 255, out=self.img)
        return self.img

    def emlut(self, val, thres, gain, clip):
        lut = 0
        if val < -thres[1]:
            lut = gain[1] * val
        elif val < -thres[0] and val > -thres[1]:
            lut = 0
        elif val < thres[0] and val > -thres[1]:
            lut = gain[0] * val
        elif val > thres[0] and val < thres[1]:
            lut = 0
        elif val > thres[1]:
            lut = gain[1] * val
        # np.clip(lut, clip[0], clip[1], out=lut)
        lut = max(clip[0], min(lut / 256, clip[1]))
        return lut

    def execute(self):
        img_pad = self.padding()
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        ee_img = np.empty((img_h, img_w), np.int16)
        em_img = np.empty((img_h, img_w), np.int16)
        for y in range(img_pad.shape[0] - 2):
            for x in range(img_pad.shape[1] - 4):
                em_img[y, x] = np.sum(np.multiply(img_pad[y:y + 3, x:x + 5], self.edge_filter[:, :])) / 8
                ee_img[y, x] = img_pad[y + 1, x + 2] + self.emlut(em_img[y, x], self.thres, self.gain, self.emclip)
        self.img = ee_img
        return self.clipping(), em_img


class EEHVectorized:
    """
    Source: https://github.com/QiuJueqin/fast-openISP/blob/master/modules/eeh.py
    Adapted.
    """

    def __init__(self, edge_gain, edge_threshold, flat_threshold, delta_threshold, clip):
        self.flat_threshold = flat_threshold
        self.edge_threshold = edge_threshold
        self.delta_threshold = delta_threshold
        self.clip = clip

        kernel = gen_gaussian_kernel(kernel_size=5, sigma=1.2)
        self.gaussian = (1024 * kernel / kernel.max()).astype(np.int32)  # x1024

        t1, t2 = flat_threshold, edge_threshold
        threshold_delta = np.clip(t2 - t1, 1E-6, None)
        self.middle_slope = np.array(edge_gain * t2 / threshold_delta, dtype=np.int32)  # x256
        self.middle_intercept = -np.array(edge_gain * t1 * t2 / threshold_delta, dtype=np.int32)  # x256
        self.edge_gain = np.array(edge_gain, dtype=np.int32)  # x256

    def execute(self, img):
        y_image = img.astype(np.int32)

        delta = y_image - generic_filter(y_image, self.gaussian)
        sign_map = np.sign(delta)
        abs_delta = np.abs(delta)

        middle_delta = np.right_shift(self.middle_slope * abs_delta + self.middle_intercept, 8)
        edge_delta = np.right_shift(self.edge_gain * abs_delta, 8)
        enhanced_delta = (
                (abs_delta > self.flat_threshold) * (abs_delta <= self.edge_threshold) * middle_delta +
                (abs_delta > self.edge_threshold) * edge_delta
        )

        enhanced_delta = sign_map * np.clip(enhanced_delta, 0, self.delta_threshold)
        eeh_y_image = np.clip(y_image + enhanced_delta, 0, self.clip)

        y_image = eeh_y_image.astype(np.uint8)
        edge_map = delta
        return y_image, edge_map
