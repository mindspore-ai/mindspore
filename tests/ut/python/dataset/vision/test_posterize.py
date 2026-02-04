# Copyright 2022 Huawei Technologies Co., Ltd
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
# =========================================================================
"""
Testing Posterize op in DE
"""
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image, ImageOps

import mindspore
import mindspore.dataset as ds
from mindspore.dataset import vision
import mindspore.log as logger


DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_posterize_op():
    """
    Feature: Posterize op
    Description: Test eager support for Posterize Cpp implementation
    Expectation: Receive correct output image from op
    """
    logger.info("test_posterize_op_c")
    for i in range(1, 9):
        posterize_op = vision.Posterize(i)

        img_in = Image.open("../data/dataset/apple.jpg")
        img_ms = posterize_op(img_in)
        img_cv = np.array(ImageOps.posterize(img_in, i))
        assert_allclose(img_ms.flatten(),
                        img_cv.flatten(),
                        rtol=1e-5,
                        atol=0)


def test_posterize_exception_bit():
    """
    Feature: Posterize op
    Description: Test Posterize with out of range or invalid type of input bits
    Expectation: Errors and logs are as expected
    """
    logger.info("test_posterize_exception_bit")
    # Test max > 8
    try:
        _ = vision.Posterize(9)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input bits is not within the required interval of [0, 8]."
    # Test min < 1
    try:
        _ = vision.Posterize(-1)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input bits is not within the required interval of [0, 8]."
    # Test wrong type (not uint8)
    try:
        _ = vision.Posterize(1.1)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument bits with value 1.1 is not of type [<class 'int'>], but got <class 'float'>."
    # Test wrong number of bits
    try:
        _ = vision.Posterize((1, 1, 1))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument bits with value (1, 1, 1) is not of type [<class 'int'>], but got <class 'tuple'>."


def test_data_type_with_posterize():
    """
    Feature: Posterize op
    Description: Test Posterize only support type CV_8S/CV_8U
    Expectation: Errors and logs are as expected
    """
    logger.info("test_data_type_with_posterize")

    data_dir_10 = "../data/dataset/testCifar10Data"
    dataset = ds.Cifar10Dataset(data_dir_10)

    rescale_op = vision.Rescale((1.0 / 255.0), 0.0)
    dataset = dataset.map(operations=rescale_op, input_columns=["image"])

    posterize_op = vision.Posterize(4)
    dataset = dataset.map(operations=posterize_op, input_columns=["image"], num_parallel_workers=1)

    try:
        _ = dataset.output_shapes()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "data type of input image should be int" in str(e)


def test_posterize_pipeline():
    """
    Feature: Posterize op
    Description: Test Posterize C implementation Pipeline
    Expectation: Pass without error
    """
    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64])]
    transforms1 = mindspore.dataset.transforms.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.Posterize(8)
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


if __name__ == "__main__":
    test_posterize_op()
    test_posterize_exception_bit()
    test_data_type_with_posterize()
    test_posterize_pipeline()
