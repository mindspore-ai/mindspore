# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Testing RandomCropDecodeResize op in DE
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(a, mse, original):
    """
    visualizes the image using DE op and Numpy Op
    """
    plt.subplot(141)
    plt.imshow(original)
    plt.title("Original image")

    plt.subplot(142)
    plt.imshow(a)
    plt.title("DE random_crop_decode_resize image")

    plt.subplot(143)
    plt.imshow(a - original)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def test_random_crop_decode_resize_op():
    """
    Test RandomCropDecodeResize op
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data1 = data1.map(input_columns=["image"], operations=random_crop_decode_resize_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):

        if num_iter > 0:
            break
        crop_and_resize = item1["image"]
        original = item2["image"]
        original = cv2.resize(original, (512, 256))
        diff = crop_and_resize - original
        mse = np.sum(np.power(diff, 2))
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        # Uncomment below line if you want to visualize images
        # visualize(crop_and_resize, mse, original)
        num_iter += 1


if __name__ == "__main__":
    test_random_crop_decode_resize_op()
