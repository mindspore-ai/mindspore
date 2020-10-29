#EXTD: Extremely Tiny Face Detector via Iterative Filter Reuse
# MIT license

# Copyright (c) 2019-present NAVER Corp.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""Augmentations"""

import random
import numpy as np
import cv2

def anchor_crop_image_sampling(image, anns):
    """
    Crop anchors.
    """
    max_size = 12000
    inf_distance = 9999999

    boxes = []
    for ann in anns:
        boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)

    height, width, _ = image.shape

    box_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(box_area) - 1)
    rand_side = box_area[rand_idx] ** 0.5

    anchors = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128, 256, 512]
    distance = inf_distance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_side) < distance:
            distance = abs(anchor - rand_side)
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 11)])
    ratio = float(target_anchor) / rand_side
    ratio = ratio * (2 ** random.uniform(-1, 1))

    if int(height * ratio * width * ratio) > max_size * max_size:
        ratio = (max_size * max_size / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    for i, _ in enumerate(anns):
        anns[i]['bbox'] = [boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]]
        for j in range(5):
            anns[i]['keypoints'][j * 3] *= ratio
            anns[i]['keypoints'][j * 3 + 1] *= ratio

    return image, anns
