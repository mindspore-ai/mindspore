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
Testing RandomAffine op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_affine_op(plot=False):
    """
    Test RandomAffine in python transformations
    """
    logger.info("test_random_affine_op")
    # define map operations
    transforms1 = [
        py_vision.Decode(),
        py_vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        py_vision.ToTensor()
    ]
    transform1 = py_vision.ComposeOp(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = py_vision.ComposeOp(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(input_columns=["image"], operations=transform1())
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform2())

    image_affine = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_affine.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_affine)


def test_random_affine_md5():
    """
    Test RandomAffine with md5 comparison
    """
    logger.info("test_random_affine_md5")
    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        py_vision.Decode(),
        py_vision.RandomAffine(degrees=(-5, 15), translate=(0.1, 0.3),
                               scale=(0.9, 1.1), shear=(-10, 10, -5, 5)),
        py_vision.ToTensor()
    ]
    transform = py_vision.ComposeOp(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(input_columns=["image"], operations=transform())

    # check results with md5 comparison
    filename = "random_affine_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_exception_negative_degrees():
    """
    Test RandomAffine: input degrees in negative, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_negative_degrees")
    try:
        _ = py_vision.RandomAffine(degrees=-15)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "If degrees is a single number, it cannot be negative."


def test_random_affine_exception_translation_range():
    """
    Test RandomAffine: translation value is not in [0, 1], expected to raise ValueError
    """
    logger.info("test_random_affine_exception_translation_range")
    try:
        _ = py_vision.RandomAffine(degrees=15, translate=(0.1, 1.5))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "translation values should be between 0 and 1"


def test_random_affine_exception_scale_value():
    """
    Test RandomAffine: scale is not positive, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_scale_value")
    try:
        _ = py_vision.RandomAffine(degrees=15, scale=(0.0, 1.1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "scale values should be positive"


def test_random_affine_exception_shear_value():
    """
    Test RandomAffine: shear is a number but is not positive, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_shear_value")
    try:
        _ = py_vision.RandomAffine(degrees=15, shear=-5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "If shear is a single number, it must be positive."


def test_random_affine_exception_degrees_size():
    """
    Test RandomAffine: degrees is a list or tuple and its length is not 2,
    expected to raise TypeError
    """
    logger.info("test_random_affine_exception_degrees_size")
    try:
        _ = py_vision.RandomAffine(degrees=[15])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "If degrees is a sequence, the length must be 2."


def test_random_affine_exception_translate_size():
    """
    Test RandomAffine: translate is not list or a tuple of length 2,
    expected to raise TypeError
    """
    logger.info("test_random_affine_exception_translate_size")
    try:
        _ = py_vision.RandomAffine(degrees=15, translate=(0.1))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "translate should be a list or tuple of length 2."


def test_random_affine_exception_scale_size():
    """
    Test RandomAffine: scale is not a list or tuple of length 2,
    expected to raise TypeError
    """
    logger.info("test_random_affine_exception_scale_size")
    try:
        _ = py_vision.RandomAffine(degrees=15, scale=(0.5))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "scale should be a list or tuple of length 2."


def test_random_affine_exception_shear_size():
    """
    Test RandomAffine: shear is not a list or tuple of length 2 or 4,
    expected to raise TypeError
    """
    logger.info("test_random_affine_exception_shear_size")
    try:
        _ = py_vision.RandomAffine(degrees=15, shear=(-5, 5, 10))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "shear should be a list or tuple and it must be of length 2 or 4."


if __name__ == "__main__":
    test_random_affine_op(plot=True)
    test_random_affine_md5()
    test_random_affine_exception_negative_degrees()
    test_random_affine_exception_translation_range()
    test_random_affine_exception_scale_value()
    test_random_affine_exception_shear_value()
    test_random_affine_exception_degrees_size()
    test_random_affine_exception_translate_size()
    test_random_affine_exception_scale_size()
    test_random_affine_exception_shear_size()
