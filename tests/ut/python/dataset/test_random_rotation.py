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
import cv2
import matplotlib.pyplot as plt
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(a, mse, original):
    """
    visualizes the image using DE op and enCV
    """
    plt.subplot(141)
    plt.imshow(original)
    plt.title("Original image")

    plt.subplot(142)
    plt.imshow(a)
    plt.title("DE random_crop_and_resize image")

    plt.subplot(143)
    plt.imshow(a - original)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def test_random_rotation_op():
    """
    Test RandomRotation op
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = c_vision.Decode()
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    random_rotation_op = c_vision.RandomRotation((90, 90), expand=True)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_rotation_op)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=decode_op)

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        if num_iter > 0:
            break
        rotation = item1["image"]
        original = item2["image"]
        logger.info("shape before rotate: {}".format(original.shape))
        original = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        diff = rotation - original
        mse = np.sum(np.power(diff, 2))
        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse == 0
        # Uncomment below line if you want to visualize images
        # visualize(rotation, mse, original)
        num_iter += 1


def test_random_rotation_expand():
    """
    Test RandomRotation op
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    # use [90, 90] to force rotate 90 degrees, expand is set to be True to match output size
    random_rotation_op = c_vision.RandomRotation((0, 90), expand=True)
    data1 = data1.map(input_columns=["image"], operations=decode_op)
    data1 = data1.map(input_columns=["image"], operations=random_rotation_op)

    num_iter = 0
    for item in data1.create_dict_iterator():
        rotation = item["image"]
        logger.info("shape after rotate: {}".format(rotation.shape))
        num_iter += 1


def test_rotation_diff():
    """
    Test Rotation op
    """
    logger.info("test_random_rotation_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    rotation_op = c_vision.RandomRotation((45, 45), expand=True)
    ctrans = [decode_op,
              rotation_op
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomRotation((45, 45), expand=True),
        py_vision.ToTensor(),
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform())

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))


if __name__ == "__main__":
    test_random_rotation_op()
    test_random_rotation_expand()
    test_rotation_diff()
