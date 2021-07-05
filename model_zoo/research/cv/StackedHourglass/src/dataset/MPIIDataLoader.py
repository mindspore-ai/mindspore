# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MPII dataset loader
"""
import os
import time

import h5py
import numpy as np
from imageio import imread

from src.config import parse_args

args = parse_args()


class MPII:
    """
    MPII dataset loader
    """

    def __init__(self):
        print("loading data...")
        tic = time.time()

        train_f = h5py.File(os.path.join(args.annot_dir, "train.h5"), "r")
        val_f = h5py.File(os.path.join(args.annot_dir, "valid.h5"), "r")

        self.t_center = train_f["center"][()]
        t_scale = train_f["scale"][()]
        t_part = train_f["part"][()]
        t_visible = train_f["visible"][()]
        t_normalize = train_f["normalize"][()]
        t_imgname = [None] * len(self.t_center)
        for i in range(len(self.t_center)):
            t_imgname[i] = train_f["imgname"][i].decode("UTF-8")

        self.v_center = val_f["center"][()]
        v_scale = val_f["scale"][()]
        v_part = val_f["part"][()]
        v_visible = val_f["visible"][()]
        v_normalize = val_f["normalize"][()]
        v_imgname = [None] * len(self.v_center)
        for i in range(len(self.v_center)):
            v_imgname[i] = val_f["imgname"][i].decode("UTF-8")

        self.center = np.append(self.t_center, self.v_center, axis=0)
        self.scale = np.append(t_scale, v_scale)
        self.part = np.append(t_part, v_part, axis=0)
        self.visible = np.append(t_visible, v_visible, axis=0)
        self.normalize = np.append(t_normalize, v_normalize)
        self.imgname = t_imgname + v_imgname

        print("Done (t={:0.2f}s)".format(time.time() - tic))

        self.num_examples_train, self.num_examples_val = self.getLength()

    def getLength(self):
        """
        get dataset length
        """
        return len(self.t_center), len(self.v_center)

    def setup_val_split(self):
        """
        get index for train and validation imgs
        index for validation images starts after that of train images
        so that loadImage can tell them apart
        """
        valid = [i + self.num_examples_train for i in range(self.num_examples_val)]
        train = [i for i in range(self.num_examples_train)]
        return np.array(train), np.array(valid)

    def get_img(self, idx):
        """
        get image
        """
        imgname = self.imgname[idx]
        path = os.path.join(args.img_dir, imgname)
        img = imread(path)
        return img

    def get_path(self, idx):
        """
        get image path
        """
        imgname = self.imgname[idx]
        path = os.path.join(args.img_dir, imgname)
        return path

    def get_kps(self, idx):
        """
        get key points
        """
        part = self.part[idx]
        visible = self.visible[idx]
        kp2 = np.insert(part, 2, visible, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2
        return kps

    def get_normalized(self, idx):
        """
        get normalized value
        """
        n = self.normalize[idx]
        return n

    def get_center(self, idx):
        """
        get center of the person
        """
        c = self.center[idx]
        return c

    def get_scale(self, idx):
        """
        get scale of the person
        """
        s = self.scale[idx]
        return s


# Part reference
parts = {
    "mpii": [
        "rank",
        "rkne",
        "rhip",
        "lhip",
        "lkne",
        "lank",
        "pelv",
        "thrx",
        "neck",
        "head",
        "rwri",
        "relb",
        "rsho",
        "lsho",
        "lelb",
        "lwri",
    ]
}

flipped_parts = {"mpii": [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {"mpii": [[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

pair_names = {"mpii": ["ankle", "knee", "hip", "pelvis", "thorax", "neck", "head", "wrist", "elbow", "shoulder"]}
