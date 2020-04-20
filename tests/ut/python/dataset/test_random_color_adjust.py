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
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def visualize(first, mse, second):
    """
    visualizes the image using DE op and enCV
    """
    plt.subplot(141)
    plt.imshow(first)
    plt.title("c transformed image")

    plt.subplot(142)
    plt.imshow(second)
    plt.title("py random_color_jitter image")

    plt.subplot(143)
    plt.imshow(first - second)
    plt.title("Difference image, mse : {}".format(mse))
    plt.show()


def diff_mse(in1, in2):
    mse = (np.square(in1.astype(float) / 255 - in2.astype(float) / 255)).mean()
    return mse * 100


def test_random_color_jitter_op_brightness():
    """
    Test RandomColorAdjust op
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    random_jitter_op = c_vision.RandomColorAdjust((0.8, 0.8), (1, 1), (1, 1), (0, 0))

    ctrans = [decode_op,
              random_jitter_op,
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust((0.8, 0.8), (1, 1), (1, 1), (0, 0)),
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

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01
        # logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        # if mse != 0:
        #     logger.info("mse is: {}".format(mse))
        # Uncomment below line if you want to visualize images
        # visualize(c_image, mse, py_image)


def test_random_color_jitter_op_contrast():
    """
    Test RandomColorAdjust op
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    random_jitter_op = c_vision.RandomColorAdjust((1, 1), (0.5, 0.5), (1, 1), (0, 0))

    ctrans = [decode_op,
              random_jitter_op
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust((1, 1), (0.5, 0.5), (1, 1), (0, 0)),
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

        diff = c_image - py_image
        logger.info("contrast difference c is : {}".format(c_image[0][0]))
        logger.info("contrast difference  py is : {}".format(py_image[0][0]))

        logger.info("contrast difference is : {}".format(diff[0][0]))
        # mse = (np.sum(np.power(diff, 2))) / (c_image.shape[0] * c_image.shape[1])
        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        # assert mse < 0.01
        # logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        # if mse != 0:
        #     logger.info("mse is: {}".format(mse))
        # Uncomment below line if you want to visualize images
        # visualize(c_image, mse, py_image)


def test_random_color_jitter_op_saturation():
    """
    Test RandomColorAdjust op
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    random_jitter_op = c_vision.RandomColorAdjust((1, 1), (1, 1), (0.5, 0.5), (0, 0))

    ctrans = [decode_op,
              random_jitter_op
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust((1, 1), (1, 1), (0.5, 0.5), (0, 0)),
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

        diff = c_image - py_image

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01
        # logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        # if mse != 0:
        #     logger.info("mse is: {}".format(mse))
        # Uncomment below line if you want to visualize images
        # visualize(c_image, mse, py_image)


def test_random_color_jitter_op_hue():
    """
    Test RandomColorAdjust op
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    random_jitter_op = c_vision.RandomColorAdjust((1, 1), (1, 1), (1, 1), (0.2, 0.2))

    ctrans = [decode_op,
              random_jitter_op,
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust((1, 1), (1, 1), (1, 1), (0.2, 0.2)),
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

        # logger.info("shape of img: {}".format(img.shape))
        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))
        # logger.info("dtype of img: {}".format(img.dtype))

        diff = c_image - py_image
        # mse = (np.sum(np.power(diff, 2))) / (c_image.shape[0] * c_image.shape[1])
        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01
        # logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        # if mse != 0:
        #     logger.info("mse is: {}".format(mse))
        # Uncomment below line if you want to visualize images
        # visualize(c_image, mse, py_image)


if __name__ == "__main__":
    test_random_color_jitter_op_brightness()
    test_random_color_jitter_op_contrast()
    test_random_color_jitter_op_saturation()
    test_random_color_jitter_op_hue()
