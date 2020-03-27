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
Testing the resize op in DE
"""
import matplotlib.pyplot as plt
import mindspore.dataset.transforms.vision.c_transforms as vision

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(image_de_resized, image_np_resized, mse):
    """
    visualizes the image using DE op and Numpy op
    """
    plt.subplot(131)
    plt.imshow(image_de_resized)
    plt.title("DE resize image")

    plt.subplot(132)
    plt.imshow(image_np_resized)
    plt.title("Numpy resized image")

    plt.subplot(133)
    plt.imshow(image_de_resized - image_np_resized)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def test_random_resize_op():
    """
    Test random_resize_op
    """
    logger.info("Test resize")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=resize_op)

    num_iter = 0
    for item in data1.create_dict_iterator():
        image_de_resized = item["image"]
        # Uncomment below line if you want to visualize images
        # visualize(image_de_resized, image_np_resized, mse)
        num_iter += 1


if __name__ == "__main__":
    test_random_resize_op()
