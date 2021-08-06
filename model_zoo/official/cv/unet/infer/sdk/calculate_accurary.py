# coding=utf-8
#
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

import os.path
import argparse
import numpy as np
from multiclass_loader import MultiClassLoader


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="path of multiclass")
    parser.add_argument("--infer_dir", type=str, required=True,
                        help="path of infer npy result")
    return parser.parse_args()


def _calculate_accuracy(infer_image, mask_image):
    """
    calculate dice and iou

    Args:
        infer_image (array): The result of an inference, of type array
        mask_image (array): The label of an image, of type array

    Returns:
        float, float

    """
    mask_image = (mask_image / 255.0).astype(int)
    mask_image = (np.arange(2) == mask_image[..., None]).astype(int)

    inter = np.dot(infer_image.flatten(), mask_image.flatten())
    union = np.dot(infer_image.flatten(), infer_image.flatten()) + \
            np.dot(mask_image.flatten(), mask_image.flatten())

    single_dice = 2 * float(inter) / float(union + 1e-6)
    single_iou = single_dice / (2 - single_dice)
    return single_dice, single_iou


def _load_infer_result(infer_result_dir):
    infer_result_dir = os.path.expanduser(infer_result_dir)
    npy_files = os.listdir(infer_result_dir)
    infer = np.array()
    for f in npy_files:
        data = np.load(f)
        infer.append(data)

    return infer


def calculate_origin_accuracy(multiclass_dir, infer_result_dir):
    data_loader = MultiClassLoader(multiclass_dir)
    dice_sum = 0.0
    iou_sum = 0.0
    cnt = 0
    for image_id, _, mask in data_loader.iter_dataset():
        infer = np.load(os.path.join(infer_result_dir, f"{image_id}.npy"))
        dice, iou = _calculate_accuracy(infer, mask)
        print(f"single dice is: {dice}, iou is {iou}")
        dice_sum += dice
        iou_sum += iou
        cnt += 1

    print(f"========== Cross Valid dice coeff is: {dice_sum / cnt}")
    print(f"========== Cross Valid IOU is: {iou_sum / cnt}")


if __name__ == '__main__':
    args = _parser_args()
    calculate_origin_accuracy(args.dataset_dir, args.infer_dir)
