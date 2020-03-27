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
Testing RandomCropAndResize op in DE
"""
import matplotlib.pyplot as plt
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

import mindspore.dataset as ds

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(a, mse, original):
    """
    visualizes the image using DE op and Numpy op
    """
    plt.subplot(141)
    plt.imshow(original)
    plt.title("Original image")

    plt.subplot(142)
    plt.imshow(a)
    plt.title("DE random_crop image")

    plt.subplot(143)
    plt.imshow(a - original)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def test_random_crop_op():
    """
    Test RandomCropAndResize op
    """
    logger.info("test_random_crop_and_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_crop_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image1 = item1["image"]
        image2 = item2["image"]


if __name__ == "__main__":
    test_random_crop_op()
