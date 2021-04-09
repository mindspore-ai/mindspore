"""
MIT License

Copyright (c) 2019 Xingyi Zhou
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import pycocotools.coco as coco

class CenterfaceDataset():
    """
    Centerface dataset definition.
    """
    def __init__(self, config, split='train'):
        self.split = split
        self.config = config
        self.max_objs = config.max_objs
        self.img_dir = self.config.img_dir
        self.annot_path = self.config.annot_path

        print('==> getting centerface key point {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if idxs:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))  # Loaded train 12671 samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (image, target) (tuple): target is index of the target class.
        """
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = len(anns)
        if num_objs > self.max_objs:
            num_objs = self.max_objs
            anns = np.random.choice(anns, num_objs)
        # dataType ERROR —— to_list
        target = []
        for ann in anns:
            tmp = []
            tmp.extend(ann['bbox'])
            tmp.extend(ann['keypoints'])
            target.append(tmp)

        img = cv2.imread(img_path)
        return img, target

    def __len__(self):
        return self.num_samples
