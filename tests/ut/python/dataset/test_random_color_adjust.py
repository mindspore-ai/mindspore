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
Testing RandomColorAdjust op in DE
"""
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def util_test_random_color_adjust_error(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
    """
    Util function that tests the error message in case of grayscale images
    """

    transforms = [
        py_vision.Decode(),
        py_vision.Grayscale(1),
        py_vision.ToTensor(),
        (lambda image: (image.transpose(1, 2, 0) * 255).astype(np.uint8))
    ]

    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # if input is grayscale, the output dimensions should be single channel, the following should fail
    random_adjust_op = c_vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                  hue=hue)
    with pytest.raises(RuntimeError) as info:
        data1 = data1.map(operations=random_adjust_op, input_columns=["image"])
        dataset_shape_1 = []
        for item1 in data1.create_dict_iterator(num_epochs=1):
            c_image = item1["image"]
            dataset_shape_1.append(c_image.shape)

    error_msg = "image shape is not <H,W,C>"

    assert error_msg in str(info.value)


def util_test_random_color_adjust_op(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0), plot=False):
    """
    Util function that tests RandomColorAdjust for a specific argument
    """

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    random_adjust_op = c_vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                                  hue=hue)

    ctrans = [decode_op,
              random_adjust_op,
              ]

    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation,
                                    hue=hue),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
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
    Test RandomColorAdjust op for brightness
    """

    logger.info("test_random_color_adjust_op_brightness")

    util_test_random_color_adjust_op(brightness=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_brightness_error():
    """
    Test RandomColorAdjust error message with brightness input in case of grayscale image
    """

    logger.info("test_random_color_adjust_op_brightness_error")

    util_test_random_color_adjust_error(brightness=(0.5, 0.5))


def test_random_color_adjust_op_contrast(plot=False):
    """
    Test RandomColorAdjust op for contrast
    """

    logger.info("test_random_color_adjust_op_contrast")

    util_test_random_color_adjust_op(contrast=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_contrast_error():
    """
    Test RandomColorAdjust error message with contrast input in case of grayscale image
    """

    logger.info("test_random_color_adjust_op_contrast_error")

    util_test_random_color_adjust_error(contrast=(0.5, 0.5))


def test_random_color_adjust_op_saturation(plot=False):
    """
    Test RandomColorAdjust op for saturation
    """
    logger.info("test_random_color_adjust_op_saturation")

    util_test_random_color_adjust_op(saturation=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_saturation_error():
    """
    Test RandomColorAdjust error message with saturation input in case of grayscale image
    """

    logger.info("test_random_color_adjust_op_saturation_error")

    util_test_random_color_adjust_error(saturation=(0.5, 0.5))


def test_random_color_adjust_op_hue(plot=False):
    """
    Test RandomColorAdjust op for hue
    """
    logger.info("test_random_color_adjust_op_hue")

    util_test_random_color_adjust_op(hue=(0.5, 0.5), plot=plot)


def test_random_color_adjust_op_hue_error():
    """
    Test RandomColorAdjust error message with hue input in case of grayscale image
    """

    logger.info("test_random_color_adjust_op_hue_error")

    util_test_random_color_adjust_error(hue=(0.5, 0.5))


def test_random_color_adjust_md5():
    """
    Test RandomColorAdjust with md5 check
    """
    logger.info("Test RandomColorAdjust with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()
    random_adjust_op = c_vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_adjust_op, input_columns=["image"])

    # Second dataset
    transforms = [
        py_vision.Decode(),
        py_vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_color_adjust_01_c_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)
    filename = "random_color_adjust_01_py_result.npz"
    save_and_check_md5(data2, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


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
