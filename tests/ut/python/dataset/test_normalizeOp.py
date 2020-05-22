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

import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def normalize_np(image):
    """
    Apply the normalization
    """
    #  DE decodes the image in RGB by deafult, hence
    #  the values here are in RGB
    image = np.array(image, np.float32)
    image = image - np.array([121.0, 115.0, 100.0])
    image = image * (1.0 / np.array([70.0, 68.0, 71.0]))
    return image


# pylint: disable=inconsistent-return-statements
def get_normalized(image_id):
    """
    Reads the image using DE ops and then normalizes using Numpy
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    num_iter = 0
    for item in data1.create_dict_iterator():
        image = item["image"]
        if num_iter == image_id:
            return normalize_np(image)
        num_iter += 1


def test_normalize_op():
    """
    Test Normalize
    """
    logger.info("Test Normalize")

    # define map operations
    decode_op = vision.Decode()
    normalize_op = vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0])

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=normalize_op)

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image_de_normalized = item1["image"]
        image_np_normalized = normalize_np(item2["image"])
        diff = image_de_normalized - image_np_normalized
        mse = np.sum(np.power(diff, 2))
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        # Uncomment these blocks to see visual results
        # plt.subplot(131)
        # plt.imshow(image_de_normalized)
        # plt.title("DE normalize image")
        #
        # plt.subplot(132)
        # plt.imshow(image_np_normalized)
        # plt.title("Numpy normalized image")
        #
        # plt.subplot(133)
        # plt.imshow(diff)
        # plt.title("Difference image, mse : {}".format(mse))
        #
        # plt.show()
        num_iter += 1


def test_decode_op():
    logger.info("Test Decode")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=decode_op)

    num_iter = 0
    image = None
    for item in data1.create_dict_iterator():
        logger.info("Looping inside iterator {}".format(num_iter))
        image = item["image"]
        # plt.subplot(131)
        # plt.imshow(image)
        # plt.title("DE image")
        # plt.show()
        num_iter += 1


def test_decode_normalize_op():
    logger.info("Test [Decode, Normalize] in one Map")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    normalize_op = vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0])

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=[decode_op, normalize_op])

    num_iter = 0
    image = None
    for item in data1.create_dict_iterator():
        logger.info("Looping inside iterator {}".format(num_iter))
        image = item["image"]
        # plt.subplot(131)
        # plt.imshow(image)
        # plt.title("DE image")
        # plt.show()
        num_iter += 1


if __name__ == "__main__":
    test_decode_op()
    test_decode_normalize_op()
    test_normalize_op()
