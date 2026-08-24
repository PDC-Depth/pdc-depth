from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
import json

from .mono_dataset import MonoDataset


class C3VDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(C3VDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[802.319/1350, 0, 668.286/1350, 0],
                           [0, 801.885/1080, 547.733/1080, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class C3VDRAWDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(C3VDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:04d}_color{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "images", f_str)

        return image_path
