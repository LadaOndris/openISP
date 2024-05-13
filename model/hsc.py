#!/usr/bin/python
import numpy as np


class HSC:
    'Hue Saturation Control'

    def __init__(self, img, hue, saturation, clip):
        self.img = img
        self.hue = hue
        self.saturation = saturation
        self.clip = clip

        hue_offset = np.pi * self.hue / 180
        self.sin_hue = (256 * np.sin(hue_offset)).astype(np.int32)  # x256
        self.cos_hue = (256 * np.cos(hue_offset)).astype(np.int32)  # x256
        self.saturation_gain = np.array(self.saturation, dtype=np.int32)  # x256

    def execute(self):
        cbcr_image = self.img.astype(np.int32)

        cb_image, cr_image = np.split(cbcr_image, 2, axis=2)

        hsc_cb_image = np.right_shift(self.cos_hue * (cb_image - 128) - self.sin_hue * (cr_image - 128), 8)  # x256
        hsc_cb_image = np.right_shift(self.saturation_gain * hsc_cb_image, 8) + 128

        hsc_cr_image = np.right_shift(self.sin_hue * (cb_image - 128) + self.cos_hue * (cr_image - 128), 8)  # x256
        hsc_cr_image = np.right_shift(self.saturation_gain * hsc_cr_image, 8) + 128

        hsc_cbcr_image = np.dstack([hsc_cr_image, hsc_cb_image])
        hsc_cbcr_image = np.clip(hsc_cbcr_image, 0, self.clip)

        return hsc_cbcr_image.astype(np.uint8)
