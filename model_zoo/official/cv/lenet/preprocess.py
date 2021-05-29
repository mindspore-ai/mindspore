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
"""pre process for 310 inference"""
import os
import argparse
import cv2
import numpy as np

batch_size = 1
parser = argparse.ArgumentParser(description="lenet preprocess data")
parser.add_argument("--dataset_path", type=str, required=True, help="dataset path.")
parser.add_argument("--output_path", type=str, required=True, help="output path.")
args = parser.parse_args()


def calcul_acc(labels, preds):
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)


def save_mnist_to_jpg(dataset_path, output_path):
    files = os.listdir(dataset_path)
    mnist_image_file = os.path.join(dataset_path, [f for f in files if "image" in f][0])
    mnist_label_file = os.path.join(dataset_path, [f for f in files if "label" in f][0])
    save_dir = output_path
    num_file = 10000
    height, width = 28, 28
    size = height * width
    prefix = 'test'
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    for i in range(num_file):
        label = label_file[i]
        image_list = [item for item in image_file[i * size:i * size + size]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(height, width)
        save_name = os.path.join(save_dir, '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, image_np)
    print("=" * 20, "preprocess data finished", "=" * 20)


if __name__ == '__main__':
    save_mnist_to_jpg(args.dataset_path, args.output_path)
