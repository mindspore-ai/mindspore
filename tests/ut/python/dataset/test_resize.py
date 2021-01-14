# Copyright 2020 Huawei Technologies Co., Ltd
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
Testing Resize op in DE
"""
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_resize_op(plot=False):
    def test_resize_op_parameters(test_name, size, plot):
        """
        Test resize_op
        """
        logger.info("Test resize: {0}".format(test_name))
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

        # define map operations
        decode_op = vision.Decode()
        resize_op = vision.Resize(size)

        # apply map operations on images
        data1 = data1.map(operations=decode_op, input_columns=["image"])

        data2 = data1.map(operations=resize_op, input_columns=["image"])
        image_original = []
        image_resized = []
        for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_1 = item1["image"]
            image_2 = item2["image"]
            image_original.append(image_1)
            image_resized.append(image_2)
        if plot:
            visualize_list(image_original, image_resized)

    test_resize_op_parameters("Test single int for size", 10, plot=False)
    test_resize_op_parameters("Test tuple for size", (10, 15), plot=False)

def test_resize_op_ANTIALIAS():
    """
    Test resize_op
    """
    logger.info("Test resize for ANTIALIAS")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = py_vision.Decode()
    resize_op = py_vision.Resize(20, Inter.ANTIALIAS)

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, resize_op, py_vision.ToTensor()], input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("use Resize by Inter.ANTIALIAS process {} images.".format(num_iter))

def test_resize_md5(plot=False):
    def test_resize_md5_parameters(test_name, size, filename, seed, plot):
        """
        Test Resize with md5 check
        """
        logger.info("Test Resize with md5 check: {0}".format(test_name))
        original_seed = config_get_set_seed(seed)
        original_num_parallel_workers = config_get_set_num_parallel_workers(1)

        # Generate dataset
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        decode_op = vision.Decode()
        resize_op = vision.Resize(size)
        data1 = data1.map(operations=decode_op, input_columns=["image"])
        data2 = data1.map(operations=resize_op, input_columns=["image"])
        image_original = []
        image_resized = []
        # Compare with expected md5 from images
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

        for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_1 = item1["image"]
            image_2 = item2["image"]
            image_original.append(image_1)
            image_resized.append(image_2)
        if plot:
            visualize_list(image_original, image_resized)

        # Restore configuration
        ds.config.set_seed(original_seed)
        ds.config.set_num_parallel_workers(original_num_parallel_workers)

    test_resize_md5_parameters("Test single int for size", 5, "resize_01_result.npz", 5, plot)
    test_resize_md5_parameters("Test tuple for size", (5, 7), "resize_02_result.npz", 7, plot)


def test_resize_op_invalid_input():
    def test_invalid_input(test_name, size, interpolation, error, error_msg):
        logger.info("Test Resize with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vision.Resize(size, interpolation)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid size parameter type as a single number", 4.5, Inter.LINEAR, TypeError,
                       "Size should be a single integer or a list/tuple (h, w) of length 2.")
    test_invalid_input("invalid size parameter shape", (2, 3, 4), Inter.LINEAR, TypeError,
                       "Size should be a single integer or a list/tuple (h, w) of length 2.")
    test_invalid_input("invalid size parameter type in a tuple", (2.3, 3), Inter.LINEAR, TypeError,
                       "Argument size at dim 0 with value 2.3 is not of type (<class 'int'>,)")
    test_invalid_input("invalid Interpolation value", (2.3, 3), None, KeyError, "None")


if __name__ == "__main__":
    test_resize_op(plot=True)
    test_resize_op_ANTIALIAS()
    test_resize_md5(plot=True)
    test_resize_op_invalid_input()
