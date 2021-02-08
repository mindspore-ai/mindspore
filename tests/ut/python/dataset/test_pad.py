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
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse, save_and_check_md5

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False

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

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.Pad(100),
        py_vision.ToTensor(),
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))
        assert mse < 0.01


def test_pad_op2():
    """
    Test Pad op2
    """
    logger.info("test padding parameter with size 2")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    resize_op = c_vision.Resize([90, 90])
    pad_op = c_vision.Pad((100, 9,))
    ctrans = [decode_op, resize_op, pad_op]

    data1 = data1.map(operations=ctrans, input_columns=["image"])
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(data["image"].shape)
        # It pads left, top with 100 and right, bottom with 9,
        # so the final size of image is 90 + 100 + 9 = 199
        assert data["image"].shape[0] == 199
        assert data["image"].shape[1] == 199


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

    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # if input is grayscale, the output dimensions should be single channel
    pad_gray = c_vision.Pad(100, fill_value=(20, 20, 20))
    data1 = data1.map(operations=pad_gray, input_columns=["image"])
    dataset_shape_1 = []
    for item1 in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item1["image"]
        dataset_shape_1.append(c_image.shape)

    # Dataset for comparison
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    # we use the same padding logic
    ctrans = [decode_op, pad_gray]
    dataset_shape_2 = []

    data2 = data2.map(operations=ctrans, input_columns=["image"])

    for item2 in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        c_image = item2["image"]
        dataset_shape_2.append(c_image.shape)

    for shape1, shape2 in zip(dataset_shape_1, dataset_shape_2):
        # validate that the first two dimensions are the same
        # we have a little inconsistency here because the third dimension is 1 after py_vision.Grayscale
        assert shape1[0:1] == shape2[0:1]


def test_pad_md5():
    """
    Test Pad with md5 check
    """
    logger.info("test_pad_md5")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    pad_op = c_vision.Pad(150)
    ctrans = [decode_op,
              pad_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    pytrans = [
        py_vision.Decode(),
        py_vision.Pad(150),
        py_vision.ToTensor(),
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(pytrans)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename1 = "pad_01_c_result.npz"
    save_and_check_md5(data1, filename1, generate_golden=GENERATE_GOLDEN)
    filename2 = "pad_01_py_result.npz"
    save_and_check_md5(data2, filename2, generate_golden=GENERATE_GOLDEN)


if __name__ == "__main__":
    test_pad_op()
    test_pad_grayscale()
    test_pad_md5()
