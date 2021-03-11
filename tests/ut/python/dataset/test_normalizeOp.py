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
Testing Normalize op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse, save_and_check_md5, visualize_image

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def normalize_np(image, mean, std):
    """
    Apply the normalization
    """
    #  DE decodes the image in RGB by default, hence
    #  the values here are in RGB
    image = np.array(image, np.float32)
    image = image - np.array(mean)
    image = image * (1.0 / np.array(std))
    return image


def util_test_normalize(mean, std, op_type):
    """
    Utility function for testing Normalize. Input arguments are given by other tests
    """
    if op_type == "cpp":
        # define map operations
        decode_op = c_vision.Decode()
        normalize_op = c_vision.Normalize(mean, std)
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        data = data.map(operations=decode_op, input_columns=["image"])
        data = data.map(operations=normalize_op, input_columns=["image"])
    elif op_type == "python":
        # define map operations
        transforms = [
            py_vision.Decode(),
            py_vision.ToTensor(),
            py_vision.Normalize(mean, std)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        data = data.map(operations=transform, input_columns=["image"])
    else:
        raise ValueError("Wrong parameter value")
    return data


def util_test_normalize_grayscale(num_output_channels, mean, std):
    """
    Utility function for testing Normalize. Input arguments are given by other tests
    """
    transforms = [
        py_vision.Decode(),
        py_vision.Grayscale(num_output_channels),
        py_vision.ToTensor(),
        py_vision.Normalize(mean, std)
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])
    return data


def test_normalize_op_c(plot=False):
    """
    Test Normalize in cpp transformations
    """
    logger.info("Test Normalize in cpp")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    # define map operations
    decode_op = c_vision.Decode()
    normalize_op = c_vision.Normalize(mean, std)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=normalize_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_original = item2["image"]
        image_np_normalized = normalize_np(image_original, mean, std)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalize_op_py(plot=False):
    """
    Test Normalize in python transformations
    """
    logger.info("Test Normalize in python")
    mean = [0.475, 0.45, 0.392]
    std = [0.275, 0.267, 0.278]
    # define map operations
    transforms = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    normalize_op = py_vision.Normalize(mean, std)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalize_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_np_normalized = (normalize_np(item2["image"].transpose(1, 2, 0), mean, std) * 255).astype(np.uint8)
        image_original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_decode_op():
    """
    Test Decode op
    """
    logger.info("Test Decode")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = c_vision.Decode()

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info("Looping inside iterator {}".format(num_iter))
        _ = item["image"]
        num_iter += 1


def test_decode_normalize_op():
    """
    Test Decode op followed by Normalize op
    """
    logger.info("Test [Decode, Normalize] in one Map")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = c_vision.Decode()
    normalize_op = c_vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0])

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, normalize_op], input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        logger.info("Looping inside iterator {}".format(num_iter))
        _ = item["image"]
        num_iter += 1


def test_normalize_md5_01():
    """
    Test Normalize with md5 check: valid mean and std
    expected to pass
    """
    logger.info("test_normalize_md5_01")
    data_c = util_test_normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], "cpp")
    data_py = util_test_normalize([0.475, 0.45, 0.392], [0.275, 0.267, 0.278], "python")

    # check results with md5 comparison
    filename1 = "normalize_01_c_result.npz"
    filename2 = "normalize_01_py_result.npz"
    save_and_check_md5(data_c, filename1, generate_golden=GENERATE_GOLDEN)
    save_and_check_md5(data_py, filename2, generate_golden=GENERATE_GOLDEN)


def test_normalize_md5_02():
    """
    Test Normalize with md5 check: len(mean)=len(std)=1 with RGB images
    expected to pass
    """
    logger.info("test_normalize_md5_02")
    data_py = util_test_normalize([0.475], [0.275], "python")

    # check results with md5 comparison
    filename2 = "normalize_02_py_result.npz"
    save_and_check_md5(data_py, filename2, generate_golden=GENERATE_GOLDEN)


def test_normalize_exception_unequal_size_c():
    """
    Test Normalize in c transformation: len(mean) != len(std)
    expected to raise ValueError
    """
    logger.info("test_normalize_exception_unequal_size_c")
    try:
        _ = c_vision.Normalize([100, 250, 125], [50, 50, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."


def test_normalize_exception_out_of_range_c():
    """
    Test Normalize in c transformation: mean, std out of range
    expected to raise ValueError
    """
    logger.info("test_normalize_exception_out_of_range_c")
    try:
        _ = c_vision.Normalize([256, 250, 125], [50, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "not within the required interval" in str(e)
    try:
        _ = c_vision.Normalize([255, 250, 125], [0, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "not within the required interval" in str(e)


def test_normalize_exception_unequal_size_py():
    """
    Test Normalize in python transformation: len(mean) != len(std)
    expected to raise ValueError
    """
    logger.info("test_normalize_exception_unequal_size_py")
    try:
        _ = py_vision.Normalize([0.50, 0.30, 0.75], [0.18, 0.32, 0.71, 0.72])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."


def test_normalize_exception_invalid_size_py():
    """
    Test Normalize in python transformation: len(mean)=len(std)=2
    expected to raise RuntimeError
    """
    logger.info("test_normalize_exception_invalid_size_py")
    data = util_test_normalize([0.75, 0.25], [0.18, 0.32], "python")
    try:
        _ = data.create_dict_iterator(num_epochs=1).__next__()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Length of mean and std must both be 1 or" in str(e)


def test_normalize_exception_invalid_range_py():
    """
    Test Normalize in python transformation: value is not in range [0,1]
    expected to raise ValueError
    """
    logger.info("test_normalize_exception_invalid_range_py")
    try:
        _ = py_vision.Normalize([0.75, 1.25, 0.5], [0.1, 0.18, 1.32])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input mean_value is not within the required interval of [0.0, 1.0]." in str(e)


def test_normalize_grayscale_md5_01():
    """
    Test Normalize with md5 check: len(mean)=len(std)=1 with 1 channel grayscale images
    expected to pass
    """
    logger.info("test_normalize_grayscale_md5_01")
    data = util_test_normalize_grayscale(1, [0.5], [0.175])
    # check results with md5 comparison
    filename = "normalize_03_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_normalize_grayscale_md5_02():
    """
    Test Normalize with md5 check: len(mean)=len(std)=3 with 3 channel grayscale images
    expected to pass
    """
    logger.info("test_normalize_grayscale_md5_02")
    data = util_test_normalize_grayscale(3, [0.5, 0.5, 0.5], [0.175, 0.235, 0.512])
    # check results with md5 comparison
    filename = "normalize_04_py_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_normalize_grayscale_exception():
    """
    Test Normalize: len(mean)=len(std)=3 with 1 channel grayscale images
    expected to raise RuntimeError
    """
    logger.info("test_normalize_grayscale_exception")
    try:
        _ = util_test_normalize_grayscale(1, [0.5, 0.5, 0.5], [0.175, 0.235, 0.512])
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input is not within the required range" in str(e)


if __name__ == "__main__":
    test_decode_op()
    test_decode_normalize_op()
    test_normalize_op_c(plot=True)
    test_normalize_op_py(plot=True)
    test_normalize_md5_01()
    test_normalize_md5_02()
    test_normalize_exception_unequal_size_c()
    test_normalize_exception_unequal_size_py()
    test_normalize_exception_invalid_size_py()
    test_normalize_exception_invalid_range_py()
    test_normalize_grayscale_md5_01()
    test_normalize_grayscale_md5_02()
    test_normalize_grayscale_exception()
