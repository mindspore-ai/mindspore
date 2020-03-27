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
Testing RandomRotation op in DE
"""
import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(image_1, image_2):
    """
    visualizes the image using RandomErasing and Cutout
    """
    plt.subplot(141)
    plt.imshow(image_1)
    plt.title("RandomErasing")

    plt.subplot(142)
    plt.imshow(image_2)
    plt.title("Cutout")

    plt.subplot(143)
    plt.imshow(image_1 - image_2)
    plt.title("Difference image")
    plt.show()


def test_random_erasing_op():
    """
    Test RandomErasing and Cutout
    """
    logger.info("test_random_erasing")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = vision.ComposeOp(transforms_1)
    data1 = data1.map(input_columns=["image"], operations=transform_1())

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.Cutout(80)
    ]
    transform_2 = vision.ComposeOp(transforms_2)
    data2 = data2.map(input_columns=["image"], operations=transform_2())

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        # visualize(image_1, image_2)


if __name__ == "__main__":
    test_random_erasing_op()
