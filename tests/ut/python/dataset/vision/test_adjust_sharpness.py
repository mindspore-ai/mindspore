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
# ==============================================================================
"""
Testing AdjustSharpness op in DE
"""
import numpy as np
from numpy.testing import assert_allclose
import PIL
from PIL import Image, ImageEnhance

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
from mindspore.dataset import vision
from mindspore import log as logger


DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

IMAGE_FILE = "../data/dataset/apple.jpg"


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_sharpness_eager():
    """
    Feature: AdjustSharpness op
    Description: Test eager support for AdjustSharpness C implementation
    Expectation: Output is the same as expected output
    """
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.uint8)
    img_in = rgb_flat.reshape((8, 8, 3))
    img_pil = Image.fromarray(img_in)

    adjustsharpness_op = vision.AdjustSharpness(0.0)
    img_out = adjustsharpness_op(img_in)
    pil_out = ImageEnhance.Sharpness(img_pil).enhance(0)
    pil_out = np.array(pil_out)
    assert_allclose(pil_out.flatten(),
                    img_out.flatten(),
                    rtol=1e-5,
                    atol=0)

    img_in2 = PIL.Image.open("../data/dataset/apple.jpg").convert("RGB")

    adjustsharpness_op2 = vision.AdjustSharpness(1.0)
    img_out2 = adjustsharpness_op2(img_in2)
    img_out2 = np.array(img_out2)
    pil_out2 = ImageEnhance.Sharpness(img_in2).enhance(1)
    pil_out2 = np.array(pil_out2)
    assert_allclose(pil_out2.flatten(),
                    img_out2.flatten(),
                    rtol=1e-5,
                    atol=0)


def test_adjust_sharpness_invalid_sharpnessfactor_param():
    """
    Feature: AdjustSharpness op
    Description: Test AdjustSharpness Cpp implementation with invalid ignore parameter
    Expectation: Correct error is raised as expected
    """
    logger.info("Test AdjustSharpness C implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid alpha
        data_set = data_set.map(operations=vision.AdjustSharpness(sharpness_factor=-10.0),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in AdjustSharpness: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid alpha
        data_set = data_set.map(operations=vision.AdjustSharpness(sharpness_factor=[1.0, 2.0]),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustSharpness: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_sharpness_pipeline():
    """
    Feature: AdjustSharpness op
    Description: Test AdjustSharpness Cpp implementation Pipeline
    Expectation: Output is the same as expected output
    """
    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64])]
    transforms1 = mindspore.dataset.transforms.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.AdjustSharpness(1.0)
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
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
    test_adjust_sharpness_eager()
    test_adjust_sharpness_invalid_sharpnessfactor_param()
    test_adjust_sharpness_pipeline()
