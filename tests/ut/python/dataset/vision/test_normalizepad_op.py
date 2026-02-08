# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False


def normalize_pad_np(image, mean, std):
    """
    Apply the normalize+pad
    """
    #  DE decodes the image in RGB by default, hence
    #  the values here are in RGB
    image = np.array(image, np.float32)
    image = image - np.array(mean)
    image = image * (1.0 / np.array(std))
    zeros = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.float32)
    output = np.concatenate((image, zeros), axis=2)
    return output


def test_normalize_pad_op_hwc(plot=False):
    """
    Feature: NormalizePad
    Description: Test NormalizePad with Decode versus NumPy comparison
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test NormalizePad with hwc")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    # define map operations
    decode_op = vision.Decode()
    normalize_pad_op = vision.NormalizePad(mean, std, is_hwc=True)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=normalize_pad_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_original = item2["image"]
        image_np_normalized = normalize_pad_np(image_original, mean, std)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalize_pad_op_chw(plot=False):
    """
    Feature: NormalizePad
    Description: Test NormalizePad with CHW input, Decode(to_pil=True) & ToTensor versus NumPy comparison
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test NormalizePad with chw")
    mean = [0.475, 0.45, 0.392]
    std = [0.275, 0.267, 0.278]
    # define map operations
    transforms = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    normalize_pad_op = vision.NormalizePad(mean, std, is_hwc=False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalize_pad_op, input_columns=["image"])

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(transforms2)

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_np_normalized = (normalize_pad_np(item2["image"].transpose(1, 2, 0), mean, std) * 255).astype(np.uint8)
        image_original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalize_pad_op_comp_chw():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with CHW input, Decode(to_pil=True) & ToTensor versus Decode(to_pil=False) & HWC2CHW
                 comparison.
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test NormalizePad with CHW input")
    mean = [0.475, 0.45, 0.392]
    std = [0.275, 0.267, 0.278]
    # define map operations
    transforms = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    normalize_pad_op = vision.NormalizePad(mean, std, is_hwc=False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalize_pad_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=vision.Decode(), input_columns=["image"])
    data2 = data2.map(operations=vision.HWC2CHW(), input_columns=["image"])
    data2 = data2.map(operations=vision.NormalizePad(mean, std, is_hwc=False), input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_np_normalized = item2["image"] / 255
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01


def test_decode_normalize_pad_op():
    """
    Feature: NormalizePad
    Description: Test Decode op followed by NormalizePad op
    Expectation: Passes the md5 check test
    """
    logger.info("Test [Decode, Normalize] in one Map")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    normalize_pad_op = vision.NormalizePad([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], "float16")

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, normalize_pad_op], input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("Looping inside iterator {}".format(num_iter))
        assert item["image"].dtype == np.float16
        num_iter += 1


def test_multi_channel_normalize_pad_chw():
    """
    Feature: NormalizePad
    Description: Test NormalizePad Op with multi-channel CHW input
    Expectation: Test succeeds.
    """
    mean = [0.475, 0.45, 0.392, 0.5]
    std = [0.275, 0.267, 0.278, 0.3]
    image = np.random.randn(4, 102, 85).astype(np.uint8)
    op = vision.NormalizePad(mean, std, is_hwc=False)
    op(image)


def test_multi_channel_normalize_pad_hwc():
    """
    Feature: NormalizePad
    Description: Test NormalizePad Op with multi-channel HWC input
    Expectation: Test succeeds.
    """
    mean = [0.475, 0.45, 0.392, 0.5]
    std = [0.275, 0.267, 0.278, 0.3]
    image = np.random.randn(102, 85, 4).astype(np.uint8)
    op = vision.NormalizePad(mean, std, is_hwc=True)
    op(image)


def test_normalize_pad_op_1channel(plot=False):
    """
    Feature: NormalizePad
    Description: Test NormalizePad Op with single channel input
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test NormalizePad Single Channel with HWC")
    mean = [121.0]
    std = [70.0]
    normalize_pad_op = vision.NormalizePad(mean, std, is_hwc=True)

    #  First dataset
    data2 = ds.MnistDataset(MNIST_DATA_DIR, shuffle=False)
    data1 = data2.map(operations=normalize_pad_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_original = item2["image"]
        image_np_normalized = normalize_pad_np(image_original, mean, std)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1
    assert num_iter == 10000


def test_normalize_pad_exception_unequal_size_1():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalize_pad_exception_unequal_size_1")
    try:
        _ = vision.NormalizePad([100, 250, 125], [50, 50, 75, 75])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."

    try:
        _ = vision.NormalizePad([100, 250, 125], [50, 50, 75], 1)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "dtype should be string."

    try:
        _ = vision.NormalizePad([100, 250, 125], [50, 50, 75], "")
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "dtype only supports float32 or float16."


def test_normalize_pad_exception_unequal_size_2():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalize_pad_exception_unequal_size_2")
    try:
        _ = vision.NormalizePad([0.50, 0.30, 0.75], [0.18, 0.32, 0.71, 0.72], is_hwc=False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Length of mean and std must be equal."

    try:
        _ = vision.NormalizePad([0.50, 0.30, 0.75], [0.18, 0.32, 0.71], 1, is_hwc=False)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "dtype should be string."

    try:
        _ = vision.NormalizePad([0.50, 0.30, 0.75], [0.18, 0.32, 0.71], "", is_hwc=False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "dtype only supports float32 or float16."


def test_normalize_pad_exception_invalid_range():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with error input: value is not in range [0,1]
    Expectation: ValueError raised
    """
    logger.info("test_normalize_pad_exception_invalid_range")
    try:
        _ = vision.NormalizePad([0.75, 1.25, 0.5], [0.1, 0.18, 1.32], is_hwc=False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input mean_value is not within the required interval of [0.0, 1.0]." in str(e)


def test_normalize_pad_runtime_error():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with error input image
    Expectation: RuntimeError raised
    """
    logger.info("test_normalize_pad_runtime_error")
    try:
        mean = [0.25, 0.65, 0.39]
        std = [0.75, 0.27, 0.28]
        image = np.random.randn(128, 128, 3, 3).astype(np.uint8)
        _ = vision.NormalizePad(mean, std, is_hwc=True)(image)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "input tensor is not in shape of <H,W> or <H,W,C>" in str(e)

    try:
        mean = [0.25, 0.65, 0.39]
        std = [0.75, 0.27, 0.28]
        image = np.random.randn(3, 10, 10).astype(np.float32)
        _ = vision.NormalizePad(mean, std, dtype="float32", is_hwc=True)(image)
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "number of channels does not match the size of mean and std vectors" in str(e)


def test_normalize_pad_invalid_param():
    """
    Feature: NormalizePad
    Description: Test NormalizePad with invalid param
    Expectation: TypeError raised
    """
    logger.info("test_normalize_pad_invalid_param")

    with pytest.raises(TypeError) as error_info:
        _ = vision.NormalizePad([0.75, 1.25, 0.5], [0.1, 0.18, "0.22"])
    assert "Argument std[2] with value 0.22 is not of type [<class 'int'>, <class 'float'>]" in str(error_info.value)


def test_normalize_pad_operation_01():
    """
    Feature: NormalizePad operation
    Description: Testing the normal functionality of the NormalizePad operator
    Expectation: The Output is equal to the expected output
    """
    # NormalizePad operator: Normal testing, numpy image, <H,W,C>
    image = np.random.randint(0, 255, (128, 128, 3))
    op = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], is_hwc=True)
    out = op(image)
    assert out.shape == (128, 128, 4)
    assert out.dtype == 'float32'

    # NormalizePad operator: Normal testing, numpy image, <C,H,W>
    image = np.random.randint(0, 255, (3, 128, 128))
    op = vision.NormalizePad(mean=(120, 150, 200), std=(20, 30, 40), is_hwc=False)
    out = op(image)
    assert out.shape == (4, 128, 128)
    assert out.dtype == 'float32'

    # NormalizePad operator: Normal testing, PIL image, <H,W,C>
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    op = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], is_hwc=True)
    out = op(image)
    assert out.shape == (128, 128, 4)
    assert out.dtype == 'float32'

    # NormalizePad operator: Normal testing, PIL image, <C,H,W>
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    image = vision.HWC2CHW()(image)
    op = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], is_hwc=False)
    out = op(image)
    assert out.shape == (4, 128, 128)
    assert out.dtype == 'float32'

    # NormalizePad operator: Normal testing, pipeline: PIL image <H,W,C>, output float32
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    transforms = [vision.ToPIL(),
                  vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float32', is_hwc=True)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape[2] == 4
        assert data["image"].dtype == 'float32'

    # NormalizePad operator: Normal testing, pipeline: PIL image <C,H,W>, output float16
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False)
    transforms = [vision.Decode(to_pil=True),
                  vision.HWC2CHW(),
                  vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float16', is_hwc=False)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape[0] == 4
        assert data["image"].dtype == 'float16'

    # NormalizePad operator: Normal testing, pipeline: numpy image <H,W,C>, output float32
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False)
    transforms = [vision.Decode(to_pil=False),
                  vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float32', is_hwc=True)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape[2] == 4
        assert data["image"].dtype == 'float32'

    # NormalizePad operator: Normal testing, pipeline: numpy image <C,H,W>, output float16
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False)
    transforms = [vision.Decode(to_pil=False),
                  vision.HWC2CHW(),
                  vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float16', is_hwc=False)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape[0] == 4
        assert data["image"].dtype == 'float16'

    # NormalizePad operator: Normal testing, <C,H,W>, output float16
    image = np.random.randint(0, 255, (3, 128, 128))
    op = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float16', is_hwc=False)
    out = op(image)
    assert out.shape == (4, 128, 128)
    assert out.dtype == 'float16'

    # NormalizePad operator: Normal testing, default parameter
    image = np.random.randint(0, 255, (20, 40, 3))
    op1 = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40])
    op2 = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float32', is_hwc=True)
    out1 = op1(image)
    out2 = op2(image)
    assert out1.shape == (20, 40, 4) == out2.shape
    assert out1.dtype == 'float32' == out2.dtype
    assert (out1 == out2).all()

    # NormalizePad operator: Normal testing, PIL image channels 4
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.uint8)
    image = vision.ToPIL()(image)
    op = vision.NormalizePad(mean=[120, 150, 200, 200], std=[20, 30, 20, 30], is_hwc=True)
    out = op(image)
    assert out.shape == (128, 128, 5)


def test_normalize_pad_operation_02():
    """
    Feature: NormalizePad operation
    Description: Testing the normal functionality of the NormalizePad operator
    Expectation: The Output is equal to the expected output
    """
    # NormalizePad operator: Normal testing, Input data is two-dimensional NumPy data.
    image = np.random.randint(0, 255, (128, 128))
    op = vision.NormalizePad(mean=[0.120], std=[0.20])
    out = op(image)
    assert out.shape == (128, 128, 2)

    # NormalizePad operator: Normal testing, numpy image channels 4, is_hwc=False
    image = np.random.randn(4, 128, 128)
    op = vision.NormalizePad(mean=[0.120, 0.160, 0.150, 0.180], std=[0.20, 0.40, 0.30, 0.60], is_hwc=False)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 128, 128)

    # NormalizePad operator: Normal testing, PIL image channels 4, is_hwc=True
    image = np.random.randn(128, 128, 4).astype(np.uint8)
    image = vision.ToPIL()(image)
    op = vision.NormalizePad(mean=[0.120, 0.160, 0.150, 0.180], std=[0.20, 0.40, 0.30, 0.60], is_hwc=True)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (128, 128, 5)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 3, is_hwc=False
    image = np.random.randn(3, 128, 128).astype(np.float32)
    op = vision.NormalizePad(mean=[0.120, 0.160, 0.150], std=[0.20, 0.40, 0.30], is_hwc=False)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 128, 128)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 3, is_hwc=True
    image = np.random.randn(128, 128, 3)
    op = vision.NormalizePad(mean=[0.120, 0.160, 0.150], std=[0.20, 0.40, 0.30], is_hwc=True)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (128, 128, 4)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 2, is_hwc=False
    image = np.random.randn(2, 128, 128)
    op = vision.NormalizePad(mean=[0.120, 0.160], std=[0.20, 0.40], is_hwc=False)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 128, 128)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 2, is_hwc=True
    image = np.random.randn(128, 128, 2)
    op = vision.NormalizePad(mean=[0.120, 0.160], std=[0.20, 0.40], is_hwc=True)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 1, is_hwc=False
    image = np.random.randn(1, 128, 128)
    op = vision.NormalizePad(mean=[0.120], std=[0.20], is_hwc=False)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 128, 128)
    assert out.dtype == np.float32

    # NormalizePad operator: Normal testing, numpy image channels 1, is_hwc=True
    image = np.random.randn(128, 128, 1)
    op = vision.NormalizePad(mean=[0.120], std=[0.20], is_hwc=True)
    out = op(image)
    assert isinstance(out, np.ndarray)
    assert out.shape == (128, 128, 2)
    assert out.dtype == np.float32


def test_normalize_pad_exception_01():
    """
    Feature: NormalizePad operation
    Description: Testing the NormalizePad Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # NormalizePad Operator: Anomaly Testing, input data is int
    image = 10
    op = vision.NormalizePad(mean=[120, 160, 120], std=[20, 40, 20])
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        op(image)

    # NormalizePad Operator: Anomaly Testing, input data is str
    image = "abc"
    op = vision.NormalizePad(mean=[120, 160, 120], std=[20, 40, 20])
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'str'>."):
        op(image)

    # NormalizePad Operator: Anomaly Testing, input data is list
    image = np.random.randint(0, 255, (128, 128, 3)).tolist()
    op = vision.NormalizePad(mean=[120, 160, 120], std=[20, 40, 20], is_hwc=True)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(image)

    # NormalizePad Operator: Anomaly Testing, The input data is a one-dimensional NumPy array.
    image = np.random.randint(0, 255, (128,))
    op = vision.NormalizePad(mean=[120, 120, 100], std=[20, 30, 40], is_hwc=True)
    with pytest.raises(RuntimeError, match="NormalizePad: input tensor is not in shape of <H,W> "
                                           "or <H,W,C>, but got rank: 1. You may need to perform Decode first."):
        op(image)

    # NormalizePad Operator: Anomaly Testing, The input data is a four-dimensional NumPy array.
    image = np.random.randint(0, 255, (128, 128, 3, 3))
    op = vision.NormalizePad(mean=[120, 120, 100], std=[20, 30, 40], is_hwc=True)
    with pytest.raises(RuntimeError, match="NormalizePad: input tensor is not in shape of <H,W> "
                                           "or <H,W,C>, but got rank: 4"):
        op(image)

    # NormalizePad Operator: Anomaly Testing, Input image is a numpy string data
    image = np.random.randn(128, 128, 1).astype("S")
    op = vision.NormalizePad(mean=[0.120], std=[0.20], is_hwc=True)
    with pytest.raises(RuntimeError, match=r"NormalizePad: unsupported type, currently supported "
                                           r"types include \[bool,int8_t,uint8_t,int16_t,uint16_t,int32_t,"
                                           r"uint32_t,int64_t,uint64_t,float16,float,double\]."):
        op(image)

    # NormalizePad Operator: Anomaly Testing, dtype=1
    with pytest.raises(TypeError, match='dtype should be string.'):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype=1, is_hwc=True)

    # NormalizePad Operator: Anomaly Testing, dtype=True
    with pytest.raises(TypeError, match='dtype should be string.'):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype=True, is_hwc=True)

    # NormalizePad Operator: Anomaly Testing, dtype='aaa'
    with pytest.raises(ValueError, match='dtype only supports float32 or float16.'):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='aaa', is_hwc=True)

    # NormalizePad Operator: Anomaly Testing, is_hwc='aaa'
    with pytest.raises(TypeError, match=r"Argument is_hwc with value aaa is not of type \[<class 'bool'>\], "
                                        r"but got <class 'str'>."):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float32', is_hwc='aaa')

    # NormalizePad Operator: Anomaly Testing, is_hwc=True but input image <C,H,W>
    image = np.random.randint(0, 255, (3, 10, 10))
    op = vision.NormalizePad(mean=[120, 150, 200], std=[20, 30, 40], dtype='float32', is_hwc=True)
    with pytest.raises(RuntimeError, match=r"NormalizePad: number of channels does not match "
                                           r"the size of mean and std vectors, got channels: 10, size of mean: 3"):
        op(image)

    # NormalizePad Operator: Anomaly Testing, the length of std is 2, while  mean is 3
    with pytest.raises(ValueError, match="Length of mean and std must be equal."):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 30])

    # NormalizePad Operator: Anomaly Testing, the length of std is 1, while  mean is 3
    with pytest.raises(ValueError, match="Length of mean and std must be equal."):
        vision.NormalizePad(mean=[120, 150, 200], std=[20])

    # NormalizePad Operator: Anomaly Testing, std TypeError str
    with pytest.raises(TypeError, match=r"Argument std with value 20 is not of type \[<class 'list'>, "
                                        r"<class 'tuple'>\], but got <class 'str'>."):
        vision.NormalizePad(mean=[120, 150, 200], std='20')

    # NormalizePad Operator: Anomaly Testing, std TypeError int
    with pytest.raises(TypeError, match=r"Argument std with value 20 is not of type \[<class 'list'>, "
                                        r"<class 'tuple'>\], but got <class 'int'>."):
        vision.NormalizePad(mean=[120, 150, 200], std=20)

    # NormalizePad Operator: Anomaly Testing, std ValueError 0
    with pytest.raises(ValueError, match=r"Input std\[2\] is not within the required interval of \(0, 255\]."):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 20, 0])

    # NormalizePad Operator: Anomaly Testing, std ValueError 256
    with pytest.raises(ValueError, match=r"Input std\[2\] is not within the required interval of \(0, 255\]."):
        vision.NormalizePad(mean=[120, 150, 200], std=[20, 20, 256])

    # NormalizePad Operator: Anomaly Testing, element in std TypeError
    with pytest.raises(TypeError, match=r"Argument std\[2\] with value 20 is not of type \[<class 'int'>, "
                                        r"<class 'float'>\], but got <class 'str'>."):
        vision.NormalizePad(mean=[0.120, 0.150, 0.200], std=[0.20, 0.20, '20'])

    # NormalizePad Operator: Anomaly Testing, mean TypeError str
    with pytest.raises(TypeError, match=r"Argument mean with value 20 is not of type \[<class 'list'>, "
                                        r"<class 'tuple'>\], but got <class 'str'>."):
        vision.NormalizePad(mean='20', std=[120, 150, 200])


def test_normalize_pad_exception_02():
    """
    Feature: NormalizePad operation
    Description: Testing the NormalizePad Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # NormalizePad Operator: Anomaly Testing, mean TypeError int
    with pytest.raises(TypeError, match=r"Argument mean with value 20 is not of type \[<class 'list'>, "
                                        r"<class 'tuple'>\], but got <class 'int'>."):
        vision.NormalizePad(mean=20, std=[120, 150, 200])

    # NormalizePad Operator: Anomaly Testing, mean ValueError -1
    with pytest.raises(ValueError, match=r"Input mean\[2\] is not within the required interval of \[0, 255\]."):
        vision.NormalizePad(mean=[20, 20, -1], std=[120, 150, 200])

    # NormalizePad Operator: Anomaly Testing, mean ValueError 256
    with pytest.raises(ValueError, match=r"Input mean\[2\] is not within the required interval of \[0, 255\]."):
        vision.NormalizePad(mean=[20, 20, 256], std=[120, 150, 200])

    # NormalizePad Operator: Anomaly Testing, element in mean TypeError
    with pytest.raises(TypeError, match=r"Argument mean\[2\] with value 20 is not of type \[<class 'int'>, "
                                        r"<class 'float'>\], but got <class 'str'>."):
        vision.NormalizePad(mean=[20, 20, '20'], std=[120, 150, 200])



if __name__ == "__main__":
    test_normalize_pad_op_hwc(plot=True)
    test_normalize_pad_op_chw(plot=True)
    test_normalize_pad_op_comp_chw()
    test_decode_normalize_pad_op()
    test_multi_channel_normalize_pad_chw()
    test_multi_channel_normalize_pad_hwc()
    test_normalize_pad_exception_unequal_size_1()
    test_normalize_pad_exception_unequal_size_2()
    test_normalize_pad_exception_invalid_range()
    test_normalize_pad_op_1channel()
    test_normalize_pad_runtime_error()
    test_normalize_pad_invalid_param()
    test_normalize_pad_operation_01()
    test_normalize_pad_operation_02()
    test_normalize_pad_exception_01()
    test_normalize_pad_exception_02()
