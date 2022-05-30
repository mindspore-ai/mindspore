# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse, visualize_image

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def normalizepad_np(image, mean, std):
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


def test_normalizepad_op_hwc(plot=False):
    """
    Feature: NormalizePad op
    Description: Test NormalizePad with Decode versus NumPy comparison
    Expectation: Test succeeds. MSE difference is negligible.
    """
    logger.info("Test NormalizePad with hwc")
    mean = [121.0, 115.0, 100.0]
    std = [70.0, 68.0, 71.0]
    # define map operations
    decode_op = vision.Decode()
    normalizepad_op = vision.NormalizePad(mean, std, is_hwc=True)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=normalizepad_op, input_columns=["image"])

    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_de_normalized = item1["image"]
        image_original = item2["image"]
        image_np_normalized = normalizepad_np(image_original, mean, std)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalizepad_op_chw(plot=False):
    """
    Feature: NormalizePad op
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
    normalizepad_op = vision.NormalizePad(mean, std, is_hwc=False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalizepad_op, input_columns=["image"])

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
        image_np_normalized = (normalizepad_np(item2["image"].transpose(1, 2, 0), mean, std) * 255).astype(np.uint8)
        image_original = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        mse = diff_mse(image_de_normalized, image_np_normalized)
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        assert mse < 0.01
        if plot:
            visualize_image(image_original, image_de_normalized, mse, image_np_normalized)
        num_iter += 1


def test_normalizepad_op_comp_chw():
    """
    Feature: NormalizePad op
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
    normalizepad_op = vision.NormalizePad(mean, std, is_hwc=False)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1 = data1.map(operations=normalizepad_op, input_columns=["image"])

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


def test_decode_normalizepad_op():
    """
    Feature: NormalizePad op
    Description: Test Decode op followed by NormalizePad op
    Expectation: Passes the md5 check test
    """
    logger.info("Test [Decode, Normalize] in one Map")

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image", "label"], num_parallel_workers=1,
                               shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    normalizepad_op = vision.NormalizePad([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], "float16")

    # apply map operations on images
    data1 = data1.map(operations=[decode_op, normalizepad_op], input_columns=["image"])

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("Looping inside iterator {}".format(num_iter))
        assert item["image"].dtype == np.float16
        num_iter += 1


def test_normalizepad_exception_unequal_size_1():
    """
    Feature: NormalizePad op
    Description: Test NormalizePad with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalizepad_exception_unequal_size_1")
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


def test_normalizepad_exception_unequal_size_2():
    """
    Feature: NormalizePad op
    Description: Test NormalizePad with error input: len(mean) != len(std)
    Expectation: ValueError raised
    """
    logger.info("test_normalizepad_exception_unequal_size_2")
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


def test_normalizepad_exception_invalid_range():
    """
    Feature: NormalizePad op
    Description: Test NormalizePad with error input: value is not in range [0,1]
    Expectation: ValueError raised
    """
    logger.info("test_normalizepad_exception_invalid_range")
    try:
        _ = vision.NormalizePad([0.75, 1.25, 0.5], [0.1, 0.18, 1.32], is_hwc=False)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input mean_value is not within the required interval of [0.0, 1.0]." in str(e)


if __name__ == "__main__":
    test_normalizepad_op_hwc(plot=True)
    test_normalizepad_op_chw(plot=True)
    test_normalizepad_op_comp_chw()
    test_decode_normalizepad_op()
    test_normalizepad_exception_unequal_size_1()
    test_normalizepad_exception_unequal_size_2()
    test_normalizepad_exception_invalid_range()
