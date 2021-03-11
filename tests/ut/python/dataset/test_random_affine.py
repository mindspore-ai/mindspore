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
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"


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
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_affine = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_affine.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_affine)


def test_random_affine_op_c(plot=False):
    """
    Test RandomAffine in C transformations
    """
    logger.info("test_random_affine_op_c")
    # define map operations
    transforms1 = [
        c_vision.Decode(),
        c_vision.RandomAffine(degrees=0, translate=(0.5, 0.5, 0, 0))
    ]

    transforms2 = [
        c_vision.Decode()
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transforms2, input_columns=["image"])

    image_affine = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
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
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_c_md5():
    """
    Test RandomAffine C Op with md5 comparison
    """
    logger.info("test_random_affine_c_md5")
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        c_vision.Decode(),
        c_vision.RandomAffine(degrees=(-5, 15), translate=(-0.1, 0.1, -0.3, 0.3),
                              scale=(0.9, 1.1), shear=(-10, 10, -5, 5))
    ]

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transforms, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_default_c_md5():
    """
    Test RandomAffine C Op (default params) with md5 comparison
    """
    logger.info("test_random_affine_default_c_md5")
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        c_vision.Decode(),
        c_vision.RandomAffine(degrees=0)
    ]

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transforms, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_default_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_py_exception_non_pil_images():
    """
    Test RandomAffine: input img is ndarray and not PIL, expected to raise RuntimeError
    """
    logger.info("test_random_affine_exception_negative_degrees")
    dataset = ds.MnistDataset(MNIST_DATA_DIR, num_samples=3, num_parallel_workers=3)
    try:
        transform = mindspore.dataset.transforms.py_transforms.Compose([py_vision.ToTensor(),
                                                                        py_vision.RandomAffine(degrees=(15, 15))])
        dataset = dataset.map(operations=transform, input_columns=["image"], num_parallel_workers=3)
        for _ in dataset.create_dict_iterator(num_epochs=1):
            pass
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Pillow image" in str(e)


def test_random_affine_exception_negative_degrees():
    """
    Test RandomAffine: input degrees in negative, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_negative_degrees")
    try:
        _ = py_vision.RandomAffine(degrees=-15)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input degrees is not within the required interval of [0, 16777216]."


def test_random_affine_exception_translation_range():
    """
    Test RandomAffine: translation value is not in [-1, 1], expected to raise ValueError
    """
    logger.info("test_random_affine_exception_translation_range")
    try:
        _ = c_vision.RandomAffine(degrees=15, translate=(0.1, 1.5))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input translate at 1 is not within the required interval of [-1.0, 1.0]."
    logger.info("test_random_affine_exception_translation_range")
    try:
        _ = c_vision.RandomAffine(degrees=15, translate=(-2, 1.5))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input translate at 0 is not within the required interval of [-1.0, 1.0]."


def test_random_affine_exception_scale_value():
    """
    Test RandomAffine: scale is not valid, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_scale_value")
    try:
        _ = py_vision.RandomAffine(degrees=15, scale=(0.0, 0.0))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale[1] must be greater than 0."

    try:
        _ = py_vision.RandomAffine(degrees=15, scale=(2.0, 1.1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale[1] must be equal to or greater than scale[0]."


def test_random_affine_exception_shear_value():
    """
    Test RandomAffine: shear is a number but is not positive, expected to raise ValueError
    """
    logger.info("test_random_affine_exception_shear_value")
    try:
        _ = py_vision.RandomAffine(degrees=15, shear=-5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear must be greater than 0."

    try:
        _ = py_vision.RandomAffine(degrees=15, shear=(5, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0]"

    try:
        _ = py_vision.RandomAffine(degrees=15, shear=(5, 1, 2, 8))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0] and " \
                         "shear[3] must be equal to or greater than shear[2]."

    try:
        _ = py_vision.RandomAffine(degrees=15, shear=(5, 9, 2, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0] and " \
                         "shear[3] must be equal to or greater than shear[2]."


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
        assert str(
            e) == "Argument translate with value 0.1 is not of type (<class 'list'>," \
                  " <class 'tuple'>)."


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
        assert str(e) == "Argument scale with value 0.5 is not of type (<class 'tuple'>," \
                         " <class 'list'>)."


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
        assert str(e) == "shear must be of length 2 or 4."


if __name__ == "__main__":
    test_random_affine_op(plot=True)
    test_random_affine_op_c(plot=True)
    test_random_affine_md5()
    test_random_affine_c_md5()
    test_random_affine_default_c_md5()
    test_random_affine_py_exception_non_pil_images()
    test_random_affine_exception_negative_degrees()
    test_random_affine_exception_translation_range()
    test_random_affine_exception_scale_value()
    test_random_affine_exception_shear_value()
    test_random_affine_exception_degrees_size()
    test_random_affine_exception_translate_size()
    test_random_affine_exception_scale_size()
    test_random_affine_exception_shear_size()
