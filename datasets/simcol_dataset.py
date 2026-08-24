from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SimcolDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SimcolDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[227.60416 / 475, 0, 227.60416 / 475, 0],
                           [0, 237.5 / 475, 237.5 / 475, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SimcolRAWDataset(SimcolDataset):
    def __init__(self, *args, **kwargs):
        super(SimcolRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "FrameBuffer_{:04d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)

        return image_path
