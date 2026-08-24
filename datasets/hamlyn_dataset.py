from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
import time

from .mono_dataset import MonoDataset


class HamlynDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(HamlynDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[417.9036255/410 ,  0, (373.208288192749-180)/410, 0],
                           [0, 417.9036255/288, 158.1358108520508/288, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"1": 1, "2": 2, "l": 1, "r": 2}

    def get_K(self, folder):
        seq_num = int(folder[-2:])

        if seq_num in [1]:
            return np.array([
                [5.98734617e-01, 0.00000000e+00, 2.43696794e-01, 0.00000000e+00],
                [0.00000000e+00, 7.98312783e-01, 2.59028316e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype=np.float32)

        if seq_num in [4, 5]:
            return np.array([
                [1.60849142e+00, 0.00000000e+00, 3.88698876e-01, 0.00000000e+00],
                [0.00000000e+00, 2.01061440e+00, 5.52149296e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype=np.float32)

        if seq_num in [6, 8, 9]:
            return np.array([
                [1.19659948e+00, 0.00000000e+00, 4.31988716e-01, 0.00000000e+00],
                [0.00000000e+00, 1.59546602e+00, 5.28490186e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype=np.float32)

        if seq_num in [11, 12]:
            return np.array([
                [1.18481112e+00, 0.00000000e+00, 4.86689210e-01, 0.00000000e+00],
                [0.00000000e+00, 1.48101389e+00, 5.31811833e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ], dtype=np.float32)

        return self.K.copy()

    def check_depth(self):
        
        return False

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class HamlynRAWDataset(HamlynDataset):
    def __init__(self, *args, **kwargs):
        super(HamlynRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image0{}".format(self.side_map[side]), f_str)

        return image_path
