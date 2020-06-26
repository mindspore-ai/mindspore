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
Testing Pad op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_pad_op():
    """
    Test Pad op
    """
    logger.info("test_random_color_jitter_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    pad_op = c_vision.Pad((100, 100, 100, 100))
    ctrans = [decode_op,
              pad_op,
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.Pad(100),
        py_vision.ToTensor(),
    ]
    transform = py_vision.ComposeOp(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform())

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01



def test_pad_grayscale():
    """
    Tests that the pad works for grayscale images
    """

    # Note: image.transpose performs channel swap to allow py transforms to
    # work with c transforms
    transforms = [
        py_vision.Decode(),
        py_vision.Grayscale(1),
        py_vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = py_vision.ComposeOp(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(input_columns=["image"], operations=transform())

    # if input is grayscale, the output dimensions should be single channel
    pad_gray = c_vision.Pad(100, fill_value=(20, 20, 20))
    data1 = data1.map(input_columns=["image"], operations=pad_gray)
    dataset_shape_1 = []
    for item1 in data1.create_dict_iterator():
        c_image = item1["image"]
        dataset_shape_1.append(c_image.shape)

    # Dataset for comparison
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    # we use the same padding logic
    ctrans = [decode_op, pad_gray]
    dataset_shape_2 = []

    data2 = data2.map(input_columns=["image"], operations=ctrans)

    for item2 in data2.create_dict_iterator():
        c_image = item2["image"]
        dataset_shape_2.append(c_image.shape)

    for shape1, shape2 in zip(dataset_shape_1, dataset_shape_2):
        # validate that the first two dimensions are the same
        # we have a little inconsistency here because the third dimension is 1 after py_vision.Grayscale
        assert shape1[0:1] == shape2[0:1]


if __name__ == "__main__":
    test_pad_op()
    test_pad_grayscale()
