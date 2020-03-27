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
Testing the random vertical flip op in DE
"""
import matplotlib.pyplot as plt
import mindspore.dataset.transforms.vision.c_transforms as vision
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def v_flip(image):
    """
    Apply the random_vertical
    """

    # with the seed provided in this test case, it will always flip.
    # that's why we flip here too
    image = image[::-1, :, :]
    return image


def visualize(image_de_random_vertical, image_pil_random_vertical, mse, image_original):
    """
    visualizes the image using DE op and Numpy op
    """
    plt.subplot(141)
    plt.imshow(image_original)
    plt.title("Original image")

    plt.subplot(142)
    plt.imshow(image_de_random_vertical)
    plt.title("DE random_vertical image")

    plt.subplot(143)
    plt.imshow(image_pil_random_vertical)
    plt.title("vertically flipped image")

    plt.subplot(144)
    plt.imshow(image_de_random_vertical - image_pil_random_vertical)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def test_random_vertical_op():
    """
    Test random_vertical
    """
    logger.info("Test random_vertical")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_vertical_op = vision.RandomVerticalFlip()
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_vertical_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):

        # with the seed value, we can only guarantee the first number generated
        if num_iter > 0:
            break

        image_v_flipped = item1["image"]

        image = item2["image"]
        image_v_flipped_2 = v_flip(image)

        diff = image_v_flipped - image_v_flipped_2
        mse = np.sum(np.power(diff, 2))
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        # Uncomment below line if you want to visualize images
        # visualize(image_v_flipped, image_v_flipped_2, mse, image)
        num_iter += 1


if __name__ == "__main__":
    test_random_vertical_op()
