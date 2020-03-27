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
import mindspore.dataset.transforms.vision.c_transforms as vision
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(image_original, image_cropped):
    """
    visualizes the image using DE op and Numpy op
    """
    num = len(image_cropped)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_original[i])
        plt.title("Original image")

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_cropped[i])
        plt.title("DE center_crop image")

    plt.show()


def test_center_crop_op(height=375, width=375, plot=False):
    """
    Test random_vertical
    """
    logger.info("Test CenterCrop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()
    # 3 images [375, 500] [600, 500] [512, 512]
    center_crop_op = vision.CenterCrop(height, width)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=center_crop_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    image_cropped = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image_cropped.append(item1["image"].copy())
        image.append(item2["image"].copy())
    if plot:
        visualize(image, image_cropped)


if __name__ == "__main__":
    test_center_crop_op()
    test_center_crop_op(600, 600)
    test_center_crop_op(300, 600)
    test_center_crop_op(600, 300)
