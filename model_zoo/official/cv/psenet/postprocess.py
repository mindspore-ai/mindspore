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


import os
import math
import operator
from functools import reduce
import numpy as np
import cv2
from src.model_utils.config import config
from src.PSENET.pse import pse


def sort_to_clockwise(points):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    clockwise_points = sorted(points, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return clockwise_points


def write_result_as_txt(image_name, img_bboxes, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = os.path.join(path, 'res_{}.txt'.format(os.path.splitext(image_name)[0]))
    lines = []
    for _, img_bbox in enumerate(img_bboxes):
        img_bbox = img_bbox.reshape(-1, 2)
        img_bbox = np.array(list(sort_to_clockwise(img_bbox)))[[3, 0, 1, 2]].copy().reshape(-1)
        values = [int(v) for v in img_bbox]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)

def get_img(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == "__main__":
    if not os.path.isdir('./res/submit_ic15/'):
        os.makedirs('./res/submit_ic15/')
    if not os.path.isdir('./res/vis_ic15/'):
        os.makedirs('./res/vis_ic15/')

    file_list = os.listdir(config.img_path)
    for k in file_list:
        if os.path.splitext(k)[-1].lower() in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(config.img_path, k)
            img = get_img(img_path).astype(np.uint8).copy()
            img_name = os.path.split(img_path)[-1]

            score = np.fromfile(os.path.join(config.result_path, k.split('.')[0] + '_0.bin'), np.float32)
            score = score.reshape(1, 1, config.INFER_LONG_SIZE, config.INFER_LONG_SIZE)
            kernels = np.fromfile(os.path.join(config.result_path, k.split('.')[0] + '_1.bin'), bool)
            kernels = kernels.reshape(1, config.KERNEL_NUM, config.INFER_LONG_SIZE, config.INFER_LONG_SIZE)
            score = np.squeeze(score)
            kernels = np.squeeze(kernels)

            # post-process
            pred = pse(kernels, 5.0)
            scale = max(img.shape[:2]) * 1.0 / config.INFER_LONG_SIZE
            label = pred
            label_num = np.max(label) + 1
            bboxes = []

            for i in range(1, label_num):
                pot = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
                if pot.shape[0] < 600:
                    continue

                score_i = np.mean(score[label == i])
                if score_i < 0.93:
                    continue

                rect = cv2.minAreaRect(pot)
                bbox = cv2.boxPoints(rect) * scale
                bbox = bbox.astype('int32')
                cv2.drawContours(img, [bbox], 0, (0, 255, 0), 3)
                bboxes.append(bbox)

            # save res
            cv2.imwrite('./res/vis_ic15/{}'.format(img_name), img[:, :, [2, 1, 0]].copy())
            write_result_as_txt(img_name, bboxes, './res/submit_ic15/')
