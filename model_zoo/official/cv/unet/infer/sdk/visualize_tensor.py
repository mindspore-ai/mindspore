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
"""
visualize tensor
"""

import argparse
import os

import cv2 as cv
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", type=str, required=True,
                        help='tensor directory')
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory")
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = args.output_dir or args.tensor_dir
    os.makedirs(output_dir, exist_ok=True)
    for tensor_file in os.listdir(args.tensor_dir):
        if not tensor_file.endswith('.npy'):
            continue

        tensor = np.load(os.path.join(args.tensor_dir, tensor_file))
        argmax_tensor = np.argmax(tensor, axis=2)
        num_classes = max(argmax_tensor.flatten())
        filename = os.path.join(output_dir,
                                tensor_file.replace('.npy', '.png'))
        cv.imwrite(filename,
                   (argmax_tensor * 255 // num_classes).astype(np.uint8))


if __name__ == "__main__":
    main()
