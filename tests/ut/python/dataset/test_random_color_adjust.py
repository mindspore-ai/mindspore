# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
Testing RandomColorAdjust in DE
"""
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def util_test_random_color_adjust_error(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
    """
    Util function that tests the error message in case of grayscale images
    """

    transforms = [
        vision.Decode(True),
        vision.Grayscale(1),
        vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # if input is grayscale, the output dimensions should be single channel, the following should fail
    random_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)
    with pytest.raises(RuntimeError) as info:
        data1 = data1.map(operations=random_adjust_op, input_columns=["image"])
        dataset_shape_1 = []
        for item1 in data1.create_dict_iterator(num_epochs=1):
            c_image = item1["image"]
            dataset_shape_1.append(c_image.shape)

    error_msg = "Expecting tensor in channel of (3)"

    assert error_msg in str(info.value)


def util_test_random_color_adjust_op(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0), plot=False):
    """
    Util function that tests RandomColorAdjust for a specific argument
    """

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()

    random_adjust_op = vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)

    ctrans = [decode_op,
              random_adjust_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                 hue=hue),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))

        mse = diff_mse(c_image, py_image)
        logger.info("mse is {}".format(mse))

        logger.info("random_rotation_op_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01

        if plot:
            visualize_image(c_image, py_image, mse)


def test_random_color_adjust_op_brightness(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for brightness
    Expectation: The dataset is processed as expected
    """

    logger.info("test_random_color_adjust_op_brightness")

    util_test_random_color_adjust_op(brightness=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_brightness_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for brightness input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_brightness_error")

    util_test_random_color_adjust_error(brightness=(0.5, 0.5))


def test_random_color_adjust_op_contrast(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for contrast
    Expectation: The dataset is processed as expected
    """

    logger.info("test_random_color_adjust_op_contrast")

    util_test_random_color_adjust_op(contrast=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_contrast_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for contrast input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_contrast_error")

    util_test_random_color_adjust_error(contrast=(0.5, 0.5))


def test_random_color_adjust_op_saturation(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for saturation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_color_adjust_op_saturation")

    util_test_random_color_adjust_op(saturation=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_saturation_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for saturation input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_saturation_error")

    util_test_random_color_adjust_error(saturation=(0.5, 0.5))


def test_random_color_adjust_op_hue(plot=False):
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for hue
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_color_adjust_op_hue")

    util_test_random_color_adjust_op(hue=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_hue_error():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust for hue input in case of grayscale image
    Expectation: Correct error is thrown and error message is printed as expected
    """

    logger.info("test_random_color_adjust_op_hue_error")

    util_test_random_color_adjust_error(hue=(0.5, 0.5))


def test_random_color_adjust_md5():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("Test RandomColorAdjust with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_adjust_op = vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_adjust_op, input_columns=["image"])

    # Second dataset
    transforms = [
        vision.Decode(True),
        vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_color_adjust_01_c_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)
    filename = "random_color_adjust_01_py_result.npz"
    save_and_check_md5_pil(data2, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_color_adjust_eager():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with eager mode
    Expectation: Test runs successfully
    """
    image = np.random.random((28, 28, 3)).astype(np.float32)
    random_color_adjust = vision.RandomColorAdjust(contrast=0.5)
    out = random_color_adjust(image)
    assert out.shape == (28, 28, 3)


def test_random_color_adjust_invalid_dtype():
    """
    Feature: RandomColorAdjust
    Description: Test RandomColorAdjust with invalid image dtype
    Expectation: RuntimeError raised
    """
    image = np.random.random((28, 28, 3)).astype(np.float64)

    # test AdjustContrast
    with pytest.raises(RuntimeError) as error_info:
        adjust_contrast = vision.RandomColorAdjust(contrast=0.5)
        _ = adjust_contrast(image)
    assert "Expecting tensor in type of (uint8, uint16, float32)" in str(error_info.value)

    # test AdjustSaturation
    with pytest.raises(RuntimeError) as error_info:
        image = np.random.random((28, 28, 3)).astype(np.float64)
        adjust_saturation = vision.RandomColorAdjust(saturation=2.0)
        _ = adjust_saturation(image)
    assert "Expecting tensor in type of (uint8, uint16, float32)" in str(error_info.value)


if __name__ == "__main__":
    test_random_color_adjust_op_brightness(plot=True)
    test_random_color_adjust_op_brightness_error()
    test_random_color_adjust_op_contrast(plot=True)
    test_random_color_adjust_op_contrast_error()
    test_random_color_adjust_op_saturation(plot=True)
    test_random_color_adjust_op_saturation_error()
    test_random_color_adjust_op_hue(plot=True)
    test_random_color_adjust_op_hue_error()
    test_random_color_adjust_md5()
    test_random_color_adjust_eager()
    test_random_color_adjust_invalid_dtype()
