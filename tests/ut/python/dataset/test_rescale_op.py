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
Testing the rescale op in DE
"""
import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def rescale_np(image):
    """
    Apply the rescale
    """
    image = image / 255.0
    image = image - 1.0
    return image


def get_rescaled(image_id):
    """
    Reads the image using DE ops and then rescales using Numpy
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    num_iter = 0
    for item in data1.create_dict_iterator():
        image = item["image"]
        if num_iter == image_id:
            return rescale_np(image)
        num_iter += 1

    return None


def visualize(image_de_rescaled, image_np_rescaled, mse):
    """
    visualizes the image using DE op and Numpy op
    """
    plt.subplot(131)
    plt.imshow(image_de_rescaled)
    plt.title("DE rescale image")

    plt.subplot(132)
    plt.imshow(image_np_rescaled)
    plt.title("Numpy rescaled image")

    plt.subplot(133)
    plt.imshow(image_de_rescaled - image_np_rescaled)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def test_rescale_op():
    """
    Test rescale
    """
    logger.info("Test rescale")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(1.0 / 255.0, -1.0)

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=rescale_op)

    num_iter = 0
    for item in data1.create_dict_iterator():
        image_de_rescaled = item["image"]
        image_np_rescaled = get_rescaled(num_iter)
        diff = image_de_rescaled - image_np_rescaled
        mse = np.sum(np.power(diff, 2))
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        # Uncomment below line if you want to visualize images
        # visualize(image_de_rescaled, image_np_rescaled, mse)
        num_iter += 1


if __name__ == "__main__":
    test_rescale_op()
